"""Dataset class for VAE models that operate on segments.

Segments are typically found with a segmenting algorithm
that thresholds audio signal energy,
e.g., syllables from birdsong or mouse USVs."""
from __future__ import annotations

import pathlib
from typing import Callable

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch.utils.data


class SegmentDataset(torch.utils.data.Dataset):
    """Dataset class for VAE models that operate on segments.

    Segments are typically found with a segmenting algorithm
    that thresholds audio signal energy,
    e.g., syllables from birdsong or mouse USVs."""

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
        return tmp_item["x"].shape

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
            subset: str | None = None,
            transform: Callable | None = None,
    ):
        """Make a :class:`SegmentDataset` instance,
        given the path to a VAE segment dataset.

        Parameters
        ----------
        dataset_path : pathlib.Path
            Path to directory that represents a
            frame classification dataset,
            as created by
            :func:`vak.prep.prep_frame_classification_dataset`.
        split : str
            The name of a split from the dataset,
            one of {'train', 'val', 'test'}.
        subset : str, optional
            Name of subset to use.
            If specified, this takes precedence over split.
            Subsets are typically taken from the training data
            for use when generating a learning curve.
        transform : callable
            The transform applied to the input to the neural network :math:`x`.

        Returns
        -------
        dataset : vak.datasets.vae.SegmentDataset
        """
        import vak.datasets  # import here just to make classmethod more explicit

        dataset_path = pathlib.Path(dataset_path)
        metadata = vak.datasets.vae.Metadata.from_dataset_path(
            dataset_path
        )

        dataset_csv_path = dataset_path / metadata.dataset_csv_filename
        dataset_df = pd.read_csv(dataset_csv_path)
        # subset takes precedence over split, if specified
        if subset:
            dataset_df = dataset_df[dataset_df.subset == subset].copy()
        else:
            dataset_df = dataset_df[dataset_df.split == split].copy()

        data = np.stack(
            [
                np.load(dataset_path / spect_path)
                for spect_path in dataset_df.spect_path.values
            ]
        )
        return cls(
            data,
            dataset_df,
            transform=transform,
        )
