import pathlib

import numpy as np
import pandas as pd
from pynndescent import NNDescent
from sklearn.utils import check_random_state
from umap.umap_ import fuzzy_simplicial_set
from torch.utils.data import Dataset


def get_umap_graph(X, n_neighbors: int = 10, metric: str= "cosine", random_state: int | None = None, max_candidates=60, verbose=True):
    random_state = check_random_state(None) if random_state == None else random_state

    # number of trees in random projection forest
    n_trees = 5 + int(round((X.shape[0]) ** 0.5 / 20.0))

    # max number of nearest neighbor iters to perform
    n_iters = max(5, int(round(np.log2(X.shape[0]))))
    # distance metric

    # get nearest neighbors
    nnd = NNDescent(
        X.reshape((len(X), np.product(np.shape(X)[1:]))),
        n_neighbors=n_neighbors,
        metric=metric,
        n_trees=n_trees,
        n_iters=n_iters,
        max_candidates=max_candidates,
        verbose=verbose
    )

    # get indices and distances
    knn_indices, knn_dists = nnd.neighbor_graph

    # build fuzzy_simplicial_set
    umap_graph, sigmas, rhos = fuzzy_simplicial_set(
        X=X,
        n_neighbors=n_neighbors,
        metric=metric,
        random_state=random_state,
        knn_indices=knn_indices,
        knn_dists=knn_dists,
    )

    return umap_graph


def get_graph_elements(graph, n_epochs):
    """Get graph elements for UMAP Dataset"""

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
    def __init__(self, data, graph, n_epochs=200, transform=None):
        graph, epochs_per_sample, head, tail, weight, n_vertices = get_graph_elements(graph, n_epochs)

        self.edges_to_exp, self.edges_from_exp = (
            np.repeat(head, epochs_per_sample.astype("int")),
            np.repeat(tail, epochs_per_sample.astype("int")),
        )
        shuffle_mask = np.random.permutation(np.arange(len(self.edges_to_exp)))
        self.edges_to_exp = self.edges_to_exp[shuffle_mask].astype(np.int64)
        self.edges_from_exp = self.edges_from_exp[shuffle_mask].astype(np.int64)
        self.data = data
        self.transform = transform

    def __len__(self):
        return int(self.data.shape[0])

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
    def from_dataset_path(cls,
                          dataset_path,
                          split,
                          n_neighbors=10,
                          metric='euclidean',
                          random_state=None,
                          n_epochs=200,
                          transform=None):
        import vak.datasets  # import here just to make classmethod more explicit

        dataset_path = pathlib.Path(dataset_path)
        metadata = vak.datasets.dimensionality_reduction.Metadata.from_dataset_path(dataset_path)

        dataset_csv_path = dataset_path / metadata.dataset_csv_filename
        dataset_df = pd.read_csv(dataset_csv_path)
        split_df = dataset_df[dataset_df.split == split]

        data = np.stack(
            [
                np.load(dataset_path / spect_path) for spect_path in split_df.spect_path.values
            ]
        )
        graph = get_umap_graph(data, n_neighbors=n_neighbors, metric=metric, random_state=random_state)

        return cls(
            data,
            graph,
            n_epochs,
            transform=transform,
        )
