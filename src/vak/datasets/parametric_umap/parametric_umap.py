from __future__ import annotations

import pathlib
import warnings
from typing import Callable

import numpy as np
import numpy.typing as npt
import pandas as pd
from pynndescent import NNDescent
import scipy.sparse._coo
from sklearn.utils import check_random_state
from torch.utils.data import Dataset

# isort: off
# Ignore warnings from Numba deprecation:
# https://numba.readthedocs.io/en/stable/reference/deprecation.html#deprecation-of-object-mode-fall-back-behaviour-when-using-jit
# Numba is required by UMAP.
from numba.core.errors import NumbaDeprecationWarning

warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
from umap.umap_ import fuzzy_simplicial_set  # noqa: E402
# isort: on


def get_umap_graph(
    X: npt.NDArray,
    n_neighbors: int = 10,
    metric: str = "euclidean",
    random_state: np.random.RandomState | None = None,
    max_candidates: int = 60,
    verbose: bool = True,
) -> scipy.sparse._coo.coo_matrix:
    r"""Get graph used by UMAP,
    the fuzzy topological representation.

    Parameters
    ----------
    X : numpy.ndarray
        Data from which to build the graph.
    n_neighbors : int
        Number of nearest neighbors to use
        when computing approximate nearest neighbors.
        Parameter passed to :class:`pynndescent.NNDescent`
        and :func:`umap._umap.fuzzy_simplicial_set`.
    metric : str
        Distance metric. Default is "cosine".
        Parameter passed to :class:`pynndescent.NNDescent`
        and :func:`umap._umap.fuzzy_simplicial_set`.
    random_state : numpy.random.RandomState
        Either a numpy.random.RandomState instance,
        or None.
    max_candidates : int
        Default is 60.
        Parameter passed to :class:`pynndescent.NNDescent`.
    verbose : bool
        Whether :class:`pynndescent.NNDescent` should log
        finding the approximate nearest neighbors.
        Default is True.

    Returns
    -------
    graph : scipy.sparse.csr_matrix

    Notes
    -----
    Adapted from https://github.com/timsainb/ParametricUMAP_paper

    The graph returned is a graph of the probabilities of an edge exists between points.

    Local, one-directional, probabilities (:math:`P^{UMAP}_{i|j}`)
    are computed between a point and its neighbors to determine
    the probability with which an edge (or simplex) exists,
    based upon an assumption that data is uniformly distributed
    across a manifold in a warped dataspace.
    Under this assumption, a local notion of distance
    is set by the distance to the :math:`k^{th}` nearest neighbor
    and the local probability is scaled by that local notion of distance.

    Where :math:`\rho_{i}` is a local connectivity parameter set
    to the distance from :math:`x_i` to its nearest neighbor,
    and :math:`\sigma_{i}` is a local connectivity parameter
    set to match the local distance around :math:`x_i` upon its :math:`k` nearest neighbors
    (where :math:`k` is a hyperparameter).
    In the UMAP package, these are calculated using :func:`umap._umap.smooth_knn_dist`.
    """
    random_state = (
        check_random_state(None) if random_state is None else random_state
    )

    # number of trees in random projection forest
    n_trees = 5 + int(round((X.shape[0]) ** 0.5 / 20.0))

    # max number of nearest neighbor iters to perform
    n_iters = max(5, int(round(np.log2(X.shape[0]))))

    # get nearest neighbors
    nnd = NNDescent(
        X.reshape((len(X), np.product(np.shape(X)[1:]))),
        n_neighbors=n_neighbors,
        metric=metric,
        n_trees=n_trees,
        n_iters=n_iters,
        max_candidates=max_candidates,
        verbose=verbose,
    )

    # get indices and distances for 10 nearest neighbors of every point in dataset
    knn_indices, knn_dists = nnd.neighbor_graph

    # build fuzzy simplicial complex
    umap_graph, sigmas, rhos = fuzzy_simplicial_set(
        X=X,
        n_neighbors=n_neighbors,
        metric=metric,
        random_state=random_state,
        knn_indices=knn_indices,
        knn_dists=knn_dists,
    )

    return umap_graph


def get_graph_elements(
    graph: scipy.sparse._coo.coo_matrix, n_epochs: int
) -> tuple[
    scipy.sparse._coo.coo_matrix,
    npt.NDArray,
    npt.NDArray,
    npt.NDArray,
    npt.NDArray,
    int,
]:
    """Get graph elements for Parametric UMAP Dataset.

    Parameters
    ----------
    graph : scipy.sparse.csr_matrix
        The graph returned by :func:`get_umap_graph`.
    n_epochs : int
        Number of epochs model will be trained

    Returns
    -------
    graph : scipy.sparse._coo.coo_matrix
        The graph, now in COOrdinate format.
    epochs_per_sample : int
    head : numpy.ndarray
        Graph rows.
    tail : numpy.ndarray
        Graph columns.
    weight : numpy.ndarray
        Graph data.
    n_vertices : int
        Number of vertices in dataset.
    """
    graph = graph.tocoo()

    # eliminate duplicate entries by summing them together
    graph.sum_duplicates()

    # number of vertices in dataset
    n_vertices = graph.shape[1]

    # get the number of epochs based on the size of the dataset
    if n_epochs is None:
        # For smaller datasets we can use more epochs
        if graph.shape[0] <= 10000:
            n_epochs = 500
        else:
            n_epochs = 200

    # remove elements with very low probability
    graph.data[graph.data < (graph.data.max() / float(n_epochs))] = 0.0
    graph.eliminate_zeros()

    # get epochs per sample based upon edge probability
    epochs_per_sample = n_epochs * graph.data

    head = graph.row
    tail = graph.col
    weight = graph.data

    return graph, epochs_per_sample, head, tail, weight, n_vertices


class ParametricUMAPDataset(Dataset):
    """Dataset used for training Parametric UMAP models"""

    def __init__(
        self,
        data: npt.NDArray,
        graph,
        dataset_df: pd.DataFrame,
        n_epochs: int = 200,
        transform: Callable | None = None,
    ):
        (
            graph,
            epochs_per_sample,
            head,
            tail,
            weight,
            n_vertices,
        ) = get_graph_elements(graph, n_epochs)

        # we repeat each sample in (head, tail) a certain number of times depending on its probability
        self.edges_to_exp, self.edges_from_exp = (
            np.repeat(head, epochs_per_sample.astype("int")),
            np.repeat(tail, epochs_per_sample.astype("int")),
        )
        # we then shuffle -- not sure this is necessary if the dataset is shuffled during training?
        shuffle_mask = np.random.permutation(np.arange(len(self.edges_to_exp)))
        self.edges_to_exp = self.edges_to_exp[shuffle_mask].astype(np.int64)
        self.edges_from_exp = self.edges_from_exp[shuffle_mask].astype(
            np.int64
        )

        self.data = data
        self.dataset_df = dataset_df
        self.transform = transform

    @property
    def duration(self):
        return self.dataset_df["duration"].sum()

    def __len__(self):
        return self.edges_to_exp.shape[0]

    @property
    def shape(self):
        tmp_x_ind = 0
        tmp_item = self.__getitem__(tmp_x_ind)
        return tmp_item[0].shape

    def __getitem__(self, index):
        edges_to_exp = self.data[self.edges_to_exp[index]]
        edges_from_exp = self.data[self.edges_from_exp[index]]
        if self.transform:
            edges_to_exp = self.transform(edges_to_exp)
            edges_from_exp = self.transform(edges_from_exp)
        return (edges_to_exp, edges_from_exp)

    @classmethod
    def from_dataset_path(
        cls,
        dataset_path: str | pathlib.Path,
        split: str,
        n_neighbors: int = 10,
        metric: str = "euclidean",
        random_state: int | None = None,
        n_epochs: int = 200,
        transform: Callable | None = None,
    ):
        """

        Parameters
        ----------
        dataset_path : str, pathlib.Path
            Path to a directory that represents a dataset.
        split
        n_neighbors
        metric
        random_state
        n_epochs
        transform

        Returns
        -------

        """
        import vak.datasets  # import here just to make classmethod more explicit

        dataset_path = pathlib.Path(dataset_path)
        metadata = vak.datasets.parametric_umap.Metadata.from_dataset_path(
            dataset_path
        )

        dataset_csv_path = dataset_path / metadata.dataset_csv_filename
        dataset_df = pd.read_csv(dataset_csv_path)
        split_df = dataset_df[dataset_df.split == split]

        data = np.stack(
            [
                np.load(dataset_path / spect_path)
                for spect_path in split_df.spect_path.values
            ]
        )
        graph = get_umap_graph(
            data,
            n_neighbors=n_neighbors,
            metric=metric,
            random_state=random_state,
        )

        return cls(
            data,
            graph,
            split_df,
            n_epochs,
            transform=transform,
        )


class ParametricUMAPInferenceDataset(Dataset):
    def __init__(
        self,
        data: npt.NDArray,
        dataset_df: pd.DataFrame,
        transform: Callable | None = None,
    ):
        self.data = data
        self.dataset_df = dataset_df
        self.transform = transform

    @property
    def duration(self):
        return self.dataset_df["duration"].sum()

    def __len__(self):
        return self.data.shape[0]

    @property
    def shape(self):
        tmp_x_ind = 0
        tmp_item = self.__getitem__(tmp_x_ind)
        return tmp_item[0].shape

    def __getitem__(self, index):
        x = self.data[index]
        df_index = self.dataset_df.index[index]
        if self.transform:
            x = self.transform(x)
        return {"x": x, "df_index": df_index}

    @classmethod
    def from_dataset_path(
        cls,
        dataset_path: str | pathlib.Path,
        split: str,
        n_neighbors: int = 10,
        metric: str = "euclidean",
        random_state: int | None = None,
        n_epochs: int = 200,
        transform: Callable | None = None,
    ):
        """

        Parameters
        ----------
        dataset_path : str, pathlib.Path
            Path to a directory that represents a dataset.
        split
        n_neighbors
        metric
        random_state
        n_epochs
        transform

        Returns
        -------

        """
        import vak.datasets  # import here just to make classmethod more explicit

        dataset_path = pathlib.Path(dataset_path)
        metadata = vak.datasets.parametric_umap.Metadata.from_dataset_path(
            dataset_path
        )

        dataset_csv_path = dataset_path / metadata.dataset_csv_filename
        dataset_df = pd.read_csv(dataset_csv_path)
        split_df = dataset_df[dataset_df.split == split]

        data = np.stack(
            [
                np.load(dataset_path / spect_path)
                for spect_path in split_df.spect_path.values
            ]
        )
        return cls(
            data,
            split_df,
            transform=transform,
        )
