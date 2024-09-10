"""Class representing CMACBench dataset."""
from __future__ import annotations

import json
import pathlib
from typing import TYPE_CHECKING, Callable, Literal

import numpy as np
import pandas as pd
import torch
import torchvision.transforms
from attrs import define

from ... import common, datapipes, transforms

from .helper import metadata_from_splits_json_path, SplitsMetadata
from .transforms import TrainItemTransform, InferItemTransform


if TYPE_CHECKING:
    from ...transforms import FramesStandardizer


VALID_TARGET_TYPES = (
    "boundary_frame_labels",
    "multi_frame_labels",
    "binary_frame_labels",
    ("boundary_frame_labels", "multi_frame_labels"),
    "None",
)


FRAMES_PATH_COL_NAME = "frames_path"
MULTI_FRAME_LABELS_PATH_COL_NAME = "multi_frame_labels_path"
BINARY_FRAME_LABELS_PATH_COL_NAME = "binary_frame_labels_path"
BOUNDARY_FRAME_LABELS_PATH_COL_NAME = "boundary_frame_labels_path"


class CMACBench(torch.utils.data.Dataset):
    """Class representing the CMAC dataset.
    
    Notes
    -----
    For more information about this dataset, please see
    https://github.com/vocalpy/CMACBench
    """

    def __init__(
        self,
        dataset_path: str | pathlib.Path,
        splits_path: str | pathlib.Path,
        split: Literal["train", "val", "test"],
        window_size: int,
        target_type: str | list[str] | tuple[str] | None = None,
        stride: int = 1,
        standardize_frames: bool = False,
        frames_standardizer: transforms.FramesStandardizer | None = None,
        frames_padval: float = 0.0,
        frame_labels_padval: int = -1,
        return_padding_mask: bool = False,
        return_frames_path: bool = False,
        item_transform: Callable | None = None,
    ):
        """BioSoundSegBench dataset."""
        # ---- validate args, roughly in order
        dataset_path = pathlib.Path(dataset_path)
        if not dataset_path.exists() or not dataset_path.is_dir():
            raise NotADirectoryError(
                f"`dataset_path` for dataset not found, or not a directory: {dataset_path}"
            )
        self.dataset_path = dataset_path
        if split not in common.constants.VALID_SPLITS:
            raise ValueError(
                f"Invalid split name: {split}\n"
                f"Valid splits are: {common.constants.VALID_SPLITS}"
            )

        splits_path = pathlib.Path(splits_path)
        if not splits_path.exists():
            tmp_splits_path = (
                dataset_path / "splits" / "splits-jsons" / splits_path
            )
            if not tmp_splits_path.exists():
                raise FileNotFoundError(
                    f"Did not find `splits_path` using either absolute path ({splits_path})"
                    f"or relative to `dataset_path` ({tmp_splits_path})"
                )
            # if tmp_splits_path *does* exist, replace splits_path with it
            splits_path = tmp_splits_path
        self.splits_path = splits_path
        self.splits_metadata = SplitsMetadata.from_paths(
            json_path=splits_path, dataset_path=dataset_path
        )

        if target_type is None and split != "predict":
            raise ValueError(
                f"Must specify `target_type` if split is '{split}', but "
                "`target_type` is None. `target_type` can only be None if split is 'predict'."
            )
        if target_type is None:
            target_type = "None"
        if not isinstance(target_type, (str, list, tuple)):
            raise TypeError(
                f"`target_type` must be string or sequence of strings but type was: {type(target_type)}\n"
                f"Valid `target_type` arguments are: {VALID_TARGET_TYPES}"
            )
        if isinstance(target_type, (list, tuple)):
            if not all(
                [isinstance(target_type_, str) for target_type_ in target_type]
            ):
                types_in_target_types = set(
                    [type(target_type_) for target_type_ in target_type]
                )
                raise TypeError(
                    "A list or tuple of `target_type` must be all strings, "
                    f"but found the following types: {types_in_target_types}\n"
                    f"`target_type` was: {target_type}\n"
                    f"Valid `target_type` arguments are: {VALID_TARGET_TYPES}"
                )
            # alphabetically sort list or tuple, and make sure it's a tuple
            target_type = tuple(sorted(target_type))
        if target_type not in VALID_TARGET_TYPES:
            raise ValueError(
                f"Invalid `target_type`: {target_type}. "
                f"Valid target types are: {VALID_TARGET_TYPES}"
            )
        if isinstance(target_type, str):
            # make single str a tuple so we can do ``if 'some target' in self.target_type``
            target_type = (target_type,)
        self.target_type = target_type

        # this is a bit convoluted: we are setting metadata, to set frame dur,
        # to be able to compute duration in property below
        self.training_replicate_metadata = metadata_from_splits_json_path(
            self.splits_path, self.dataset_path
        )
        self.frame_dur = (
            self.training_replicate_metadata.frame_dur * 1e-3
        )  # convert from ms to s!

        if "multi_frame_labels" in target_type:
            labelmaps_json_path = self.dataset_path / "labelmaps.json"
            if not labelmaps_json_path.exists():
                raise FileNotFoundError(
                    '`target_type` includes "multi_frame_labels" but '
                    "'labelmaps.json' was not found in root of dataset path:\n"
                    f"{labelmaps_json_path}"
                )
            with labelmaps_json_path.open("r") as fp:
                labelmaps = json.load(fp)
                group = self.training_replicate_metadata.biosound_group
                unit = self.training_replicate_metadata.unit
                id_ = self.training_replicate_metadata.id
            if id_ is not None:
                if group == "Mouse-Pup-Call":
                    self.labelmap = labelmaps[group][unit]["all"]
                else:
                    self.labelmap = labelmaps[group][unit][id_]
            else:
                if group == "Human-Speech":
                    self.labelmap = labelmaps[group][unit]["all"]
                else:
                    raise ValueError(
                        "Unable to determine labelmap to use for "
                        f"group '{group}', unit '{unit}', and id '{id}'. "
                        "Please check that splits_json path is correct."
                    )
        elif target_type == ("binary_frame_labels",):
            self.labelmap = {"no segment": 0, "segment": 1}
        elif target_type == ("boundary_frame_labels",):
            self.labelmap = {"no boundary": 0, "boundary": 1}

        self.split = split
        split_df = pd.read_csv(self.splits_metadata.splits_csv_path)
        split_df = split_df[split_df.split == split].copy()
        self.split_df = split_df

        self.frames_paths = self.split_df[FRAMES_PATH_COL_NAME].values
        self.target_paths = {}
        if "multi_frame_labels" in self.target_type:
            self.target_paths["multi_frame_labels"] = self.split_df[
                MULTI_FRAME_LABELS_PATH_COL_NAME
            ].values
        if "binary_frame_labels" in self.target_type:
            self.target_paths["binary_frame_labels"] = self.split_df[
                BINARY_FRAME_LABELS_PATH_COL_NAME
            ].values
        if "boundary_frame_labels" in self.target_type:
            self.target_paths["boundary_frame_labels"] = self.split_df[
                BOUNDARY_FRAME_LABELS_PATH_COL_NAME
            ].values

        self.window_size = window_size
        self.stride = stride

        # we need all these vectors for getting batches of windows during training
        # for other splits, we use these to determine the duration of the dataset
        self.sample_ids = np.load(
            getattr(self.splits_metadata.sample_id_vector_paths, split)
        )
        self.inds_in_sample = np.load(
            getattr(self.splits_metadata.inds_in_sample_vector_paths, split)
        )
        self.window_inds = (
            datapipes.frame_classification.train_datapipe.get_window_inds(
                self.sample_ids.shape[-1], window_size, stride
            )
        )

        if item_transform is None:
            if standardize_frames and frames_standardizer is None:
                from ..transforms import FramesStandardizer

                frames_standardizer = (
                    FramesStandardizer.fit_inputs_targets_csv_path(
                        self.splits_metadata.splits_csv_path, self.dataset_path
                    )
                )
            if split == "train":
                self.item_transform = TrainItemTransform(
                    frames_standardizer=frames_standardizer
                )
            elif split in ("val", "test", "predict"):
                self.item_transform = InferItemTransform(
                    window_size,
                    frames_standardizer,
                    frames_padval,
                    frame_labels_padval,
                    return_padding_mask,
                )
        else:
            if not callable(item_transform):
                raise ValueError(
                    "`item_transform` should be `callable` "
                    f"but value for `item_transform` was not: {item_transform}"
                )
            self.item_transform = item_transform

        self.return_frames_path = return_frames_path

    @property
    def shape(self):
        tmp_x_ind = 0
        tmp_item = self.__getitem__(tmp_x_ind)
        input_shape = tmp_item["frames"].shape
        if self.split == "train" and len(input_shape) == 3:
            return input_shape
        elif (
            self.split in ("val", "test", "predict") and len(input_shape) == 4
        ):
            # discard windows dimension from shape,
            # it's sample dependent and not what we want
            return input_shape[1:]

    @property
    def duration(self):
        return self.sample_ids.shape[-1] * self.frame_dur

    def __len__(self):
        """number of batches"""
        if self.split == "train":
            return len(self.window_inds)
        else:
            return len(np.unique(self.sample_ids))

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
            frames_path = self.dataset_path / self.frames_paths[sample_id]
            spect_dict = common.files.spect.load(frames_path)
            item["frames"] = spect_dict[common.constants.SPECT_KEY]
            for target_type in self.target_type:
                item[target_type] = np.load(
                    self.dataset_path
                    / self.target_paths[target_type][sample_id]
                )

        elif len(uniq_sample_ids) > 1:
            item["frames"] = []
            for target_type in self.target_type:
                # do this to append instead of using defaultdict
                # so that when we do `'target_type' in item` we don't get empty list
                item[target_type] = []
            for sample_id in sorted(uniq_sample_ids):
                frames_path = self.dataset_path / self.frames_paths[sample_id]
                spect_dict = common.files.spect.load(frames_path)
                item["frames"].append(spect_dict[common.constants.SPECT_KEY])
                for target_type in self.target_type:
                    item[target_type].append(
                        np.load(
                            self.dataset_path
                            / self.target_paths[target_type][sample_id]
                        )
                    )

            item["frames"] = np.concatenate(item["frames"], axis=1)
            for target_type in self.target_type:
                item[target_type] = np.concatenate(item[target_type])
        else:
            raise ValueError(
                f"Unexpected number of ``uniq_sample_ids``: {uniq_sample_ids}"
            )

        ind_in_sample = self.inds_in_sample[window_idx]
        item["frames"] = item["frames"][
            ...,
            ind_in_sample : ind_in_sample + self.window_size,  # noqa: E203
        ]
        for target_type in self.target_type:
            item[target_type] = item[target_type][
                ind_in_sample : ind_in_sample + self.window_size  # noqa: E203
            ]
        item = self.item_transform(**item)
        return item

    def _getitem_infer(self, idx):
        item = {}
        frames_path = self.dataset_path / self.frames_paths[idx]
        if self.return_frames_path:
            item["frames_path"] = frames_path
        spect_dict = common.files.spect.load(frames_path)
        item["frames"] = spect_dict[common.constants.SPECT_KEY]
        if self.target_type != "None":  # target_type can be None for predict
            for target_type in self.target_type:
                item[target_type] = np.load(
                    self.dataset_path / self.target_paths[target_type][idx]
                )
        item = self.item_transform(**item)
        return item

    def __getitem__(self, idx):
        if self.split == "train":
            item = self._getitem_train(idx)
        elif self.split in ("val", "test", "predict"):
            item = self._getitem_infer(idx)
        return item
