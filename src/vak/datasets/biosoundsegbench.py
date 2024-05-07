from __future__ import annotations

from typing import Callable, Literal, Mapping

import collections
import dataclasses
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import vak

from biosoundsegbench import transforms


VALID_TARGET_TYPES = (
    'boundary_onehot',
    'multi_frame_labels',
    'binary_frame_labels',
    ('boundary_onehot', 'binary_frame_labels'),
    ('binary_frame_labels', 'boundary_onehot'),
    ('boundary_onehot', 'multi_frame_labels'),
    ('multi_frame_labels', 'boundary_onehot'),
    'None',
)


FRAMES_PATH_COL_NAME = "frames_path"
MULTI_FRAME_LABELS_PATH_COL_NAME = "multi_frame_labels_path"
BINARY_FRAME_LABELS_PATH_COL_NAME = "binary_frame_labels_path"
BOUNDARY_ONEHOT_PATH_COL_NAME = "boundary_onehot_path"


@dataclasses.dataclass
class SampleIDVectorPaths:
    train: pathlib.Path
    val: pathlib.Path
    test: pathlib.Path


@dataclasses.dataclass
class IndsInSampleVectorPaths:
    train: pathlib.Path
    val: pathlib.Path
    test: pathlib.Path


@dataclasses.dataclass
class SplitsMetadata:
    """Dataclass that represents metadata about dataset splits,
    loaded from a json file"""
    splits_csv_path: pathlib.Path
    sample_id_vector_paths: SampleIDVectorPaths
    inds_in_sample_vector_paths: IndsInSampleVectorPaths

    @classmethod
    def from_paths(cls, json_path, dataset_path):
        json_path = pathlib.Path(json_path)
        with json_path.open('r') as fp:
            splits_json = json.load(fp)

        dataset_path = pathlib.Path(dataset_path)
        if not dataset_path.exists() or not dataset_path.is_dir():
            raise NotADirectoryError(
                f"`dataset_path` not found or not a directory: {dataset_path}"
            )

        splits_csv_path = pathlib.Path(
            dataset_path / splits_json['splits_csv_path']
        )
        if not splits_csv_path.exists():
            raise FileNotFoundError(
                f"`splits_csv_path` not found: {splits_csv_path}"
            )

        sample_id_vector_paths = {
            split: dataset_path / path
            for split, path in splits_json['sample_id_vec_path'].items()
        }
        for split, vec_path in sample_id_vector_paths.items():
            if not vec_path.exists():
                raise FileNotFoundError(
                    f"`sample_id_vector_path` for split '{split}' not found: {vec_path}"
                )
        sample_id_vector_paths = SampleIDVectorPaths(
            **sample_id_vector_paths
        )

        inds_in_sample_vector_paths = {
            split: dataset_path / path
            for split, path in splits_json['inds_in_sample_vec_path'].items()
        }
        for split, vec_path in inds_in_sample_vector_paths.items():
            if not vec_path.exists():
                raise FileNotFoundError(
                    f"`inds_in_sample_vec_path` for split '{split}' not found: {vec_path}"
                )
        inds_in_sample_vector_paths = IndsInSampleVectorPaths(
            **inds_in_sample_vector_paths
        )

        return cls(
            splits_csv_path,
            sample_id_vector_paths,
            inds_in_sample_vector_paths
        )


class BioSoundSegBench:
    def __init__(
        self,
        root: str | pathlib.Path,
        splits_path: str | pathlib.Path,
        split: Literal["train", "val", "test"],
        item_transform: Callable,
        target_type: str | list[str] | tuple[str] | None = None,
        window_size: int | None = None,
        stride: int = 1,
    ):
        """BioSoundSegBench dataset."""
        if split == 'train' and window_size is None:
            raise ValueError(
                "Must specify `window_size` if split is 'train'`, but "
                "`window_size` is None."
            )
        if split == 'train' and target_type is None:
            raise ValueError(
                "Must specify `target_type` if split is 'train'`, but "
                "`target_type` is None."
            )

        root = pathlib.Path(root)
        if not root.exists() or not root.is_dir():
            raise NotADirectoryError(
                f"`root` for dataset not found, or not a directory: {root}"
            )
        self.root = root

        splits_path = pathlib.Path(splits_path)
        if not splits_path.exists():
            raise NotADirectoryError(
                f"`splits_path` not found: {splits_path}"
            )
        self.splits_metadata = SplitsMetadata.from_paths(
            json_path=splits_path, dataset_path=root
        )

        if target_type is None:
            target_type = 'None'
        if target_type not in VALID_TARGET_TYPES:
            raise ValueError(
                f"Invalid `target_type`: {target_type}. "
                f"Valid target types are: {VALID_TARGET_TYPES}"
            )
        if isinstance(target_type, str):
            # make single str a tuple so we can do ``if 'some target' in self.target_type``
            target_type = (target_type,)
        self.target_type = target_type

        self.split = split
        split_df = pd.read_csv(self.splits_metadata.splits_csv_path)
        split_df = split_df[split_df.split == split].copy()
        self.split_df = split_df

        self.frames_paths = self.split_df[
            FRAMES_PATH_COL_NAME
        ].values
        self.target_paths = {}
        if 'multi_frame_labels' in self.target_type:
            self.target_paths['multi_frame_labels'] = self.split_df[
                MULTI_FRAME_LABELS_PATH_COL_NAME
            ].values
        if 'binary_frame_labels' in self.target_type:
            self.target_paths['binary_frame_labels'] = self.split_df[
                BINARY_FRAME_LABELS_PATH_COL_NAME
            ].values
        if 'boundary_onehot' in self.target_type:
            self.target_paths['boundary_onehot'] = self.split_df[
                BOUNDARY_ONEHOT_PATH_COL_NAME
            ].values
        else:
            self.boundary_onehot_paths = None

        self.sample_ids = np.load(
            getattr(self.splits_metadata.sample_id_vector_paths, split)
        )
        self.inds_in_sample = np.load(
            getattr(self.splits_metadata.inds_in_sample_vector_paths, split)
        )
        self.window_size = window_size
        self.stride = stride
        if split == 'train':
            window_inds = vak.datasets.frame_classification.window_dataset.get_window_inds(
                self.sample_ids.shape[-1], window_size, stride
            )
        else:
            window_inds = None
        self.window_inds = window_inds
        self.item_transform = item_transform

    @property
    def input_shape(self):
        tmp_x_ind = 0
        tmp_item = self.__getitem__(tmp_x_ind)
        input_shape = tmp_item["frames"].shape
        if self.split == 'train' and len(input_shape) == 3:
            return input_shape
        elif self.split in ('val', 'test', 'predict') and len(input_shape) == 4:
            # discard windows dimension from shape --
            # it's sample dependent and not what we want
            return input_shape[1:]

    def _getitem_train(self, idx):
        window_idx = self.window_inds[idx]
        sample_ids = self.sample_ids[
            window_idx : window_idx + self.window_size  # noqa: E203
        ]
        uniq_sample_ids = np.unique(sample_ids)
        item = {}
        if len(uniq_sample_ids) == 1:
            # repeat myself to avoid running a loop on one item
            sample_id = uniq_sample_ids[0]
            frames_path = self.root / self.frames_paths[sample_id]
            spect_dict = vak.common.files.spect.load(frames_path)
            item['frames'] = spect_dict[vak.common.constants.SPECT_KEY]
            for target_type in self.target_type:
                item[target_type] = np.load(
                    self.root / self.target_paths[target_type][sample_id]
                )

        elif len(uniq_sample_ids) > 1:
            item['frames'] = []
            for target_type in self.target_type:
                # do this to append instead of using defaultdict
                # so that when we do `'target_type' in item` we don't get empty list
                item[target_type] = []
            for sample_id in sorted(uniq_sample_ids):
                frames_path = self.root / self.frames_paths[sample_id]
                spect_dict = vak.common.files.spect.load(frames_path)
                item['frames'].append(
                    spect_dict[vak.common.constants.SPECT_KEY]
                )
                for target_type in self.target_type:
                    item[target_type].append(
                        np.load(
                            self.root / self.target_paths[target_type][sample_id]
                        )
                    )

            item['frames'] = np.concatenate(item['frames'], axis=1)
            for target_type in self.target_type:
                item[target_type] = np.concatenate(item[target_type])
        else:
            raise ValueError(
                f"Unexpected number of ``uniq_sample_ids``: {uniq_sample_ids}"
            )

        ind_in_sample = self.inds_in_sample[window_idx]
        item['frames'] = item['frames'][
            ...,
            ind_in_sample : ind_in_sample + self.window_size,  # noqa: E203
        ]
        for target_type in self.target_type:
            item[target_type] = item[target_type][
                ind_in_sample : ind_in_sample + self.window_size  # noqa: E203
            ]
        item = self.item_transform(item)
        return item

    def _getitem_val(self, idx):
        item = {}
        frames_path = self.root / self.frames_paths[idx]
        spect_dict = vak.common.files.spect.load(frames_path)
        item['frames'] = spect_dict[vak.common.constants.SPECT_KEY]
        for target_type in self.target_type:
            item[target_type] = np.load(
                self.root / self.target_paths[target_type][idx]
            )
        item = self.item_transform(item)
        return item

    def __getitem__(self, idx):
        if self.split == 'train':
            item = self._getitem_train(idx)
        else:
            item = self._getitem_val(idx)
        return item


# def get_biosoundsegbench(
#     root: str | pathlib.Path,
#     species: str | list[str] | tuple[str],
#     target_type: str | list[str] | tuple[str],
#     unit: str,
#     id: str | None,
#     split: str,
#     window_size: int,
#     stride: int = 1,
#     labelmap: Mapping | None = None
# ):
#     """Get a :class:`DataPipe` instance
#     for loading samples from the BioSoundSegBench.

#     This function determines the correct data to use,
#     according to the `species`, `unit`, and `id`
#     specified.

#     It also determines which `transform` to use,
#     according to the `target_type`.
#     """
#     root = pathlib.Path(root)
#     if not root.exists() or not root.is_dir():
#         raise NotADirectoryError()

#     species_dict = INIT_ARGS_CSV_MAP[species]
#     if id is None:
#         # we ignore individual ID and concatenate all CSVs
#         ids_dict = species_dict[unit]
#         csv_paths = [
#             csv_path for id, csv_path in ids_dict.items()
#         ]
#     else:
#         csv_paths = [species_dict[unit][id]]
#     dataset_df = []
#     for csv_path in csv_paths:
#         dataset_df.append(pd.read_csv(csv_path))
#     dataset_df = pd.concat(dataset_df)

#     # TODO: I think this is a case where we need an "item transform" for train,
#     # to encapsulate the logic of dealing with different target types
#     if split == 'train':
#         # for boundary detection and binary classification, we use target transforms
#         # instead of loading from separate vectors for now
#         # TODO: fix this to load from separate data we prep -- be more frugal at runtime
#         if target_type == 'boundary':
#             target_transform = transforms.FrameLabelsToBoundaryOnehot()
#         elif target_type == 'label-binary':
#             target_transform = transforms.
#         elif target_type == 'label-multi':
#             # all we have to do is load the frame labels vector
#             target_transform = None


#     elif split in ('val', 'test', 'predict'):
