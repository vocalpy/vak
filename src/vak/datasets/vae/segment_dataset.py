from __future__ import annotations

import pathlib
from typing import Callable

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch.utils.data


class SegmentDataset(torch.utils.data.Dataset):
    """Pipeline for loading samples from a dataset of spectrograms

    This is a simplified version of
    :class:`vak.datasets.parametric_umap.ParametricUmapInferenceDataset`.
    """

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
            transform: Callable | None = None,
    ):
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
