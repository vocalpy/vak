"""Class representing BioSoundSegBench dataset."""

from __future__ import annotations

import json
import pathlib
from typing import TYPE_CHECKING, Callable, Literal

import numpy as np
import pandas as pd
import torch
import torchvision.transforms
from attrs import define

from .. import common, datapipes, transforms

if TYPE_CHECKING:
    from ..transforms import FramesStandardizer


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


@define
class SampleIDVectorPaths:
    train: pathlib.Path
    val: pathlib.Path
    test: pathlib.Path


@define
class IndsInSampleVectorPaths:
    train: pathlib.Path
    val: pathlib.Path
    test: pathlib.Path


@define
class SplitsMetadata:
    """Class that represents metadata about dataset splits
    in the BioSoundSegBench dataset, loaded from a json file"""

    splits_csv_path: pathlib.Path
    sample_id_vector_paths: SampleIDVectorPaths
    inds_in_sample_vector_paths: IndsInSampleVectorPaths

    @classmethod
    def from_paths(cls, json_path, dataset_path):
        json_path = pathlib.Path(json_path)
        with json_path.open("r") as fp:
            splits_json = json.load(fp)

        dataset_path = pathlib.Path(dataset_path)
        if not dataset_path.exists() or not dataset_path.is_dir():
            raise NotADirectoryError(
                f"`dataset_path` not found or not a directory: {dataset_path}"
            )

        splits_csv_path = pathlib.Path(
            dataset_path / splits_json["splits_csv_path"]
        )
        if not splits_csv_path.exists():
            raise FileNotFoundError(
                f"`splits_csv_path` not found: {splits_csv_path}"
            )

        sample_id_vector_paths = {
            split: dataset_path / path
            for split, path in splits_json["sample_id_vec_path"].items()
        }
        for split, vec_path in sample_id_vector_paths.items():
            if not vec_path.exists():
                raise FileNotFoundError(
                    f"`sample_id_vector_path` for split '{split}' not found: {vec_path}"
                )
        sample_id_vector_paths = SampleIDVectorPaths(**sample_id_vector_paths)

        inds_in_sample_vector_paths = {
            split: dataset_path / path
            for split, path in splits_json["inds_in_sample_vec_path"].items()
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
            inds_in_sample_vector_paths,
        )


@define
class TrainingReplicateMetadata:
    """Class representing metadata for a
    pre-defined training replicate
    in the BioSoundSegBench dataset.
    """

    biosound_group: str
    id: str | None
    frame_dur: float
    unit: str
    data_source: str | None
    train_dur: float
    replicate_num: int


def metadata_from_splits_json_path(
    splits_json_path: pathlib.Path, datset_path: pathlib.Path
) -> TrainingReplicateMetadata:
    name = splits_json_path.name
    try:
        (
            biosound_group,
            unit,
            id_,
            frame_dur_1st_half,
            frame_dur_2nd_half,
            data_source,
            train_dur_1st_half,
            train_dur_2nd_half,
            replicate_num,
            _,
            _,
        ) = name.split(".")
    # Human-Speech doesn't have ID or data source in filename
    # so it will raise a ValueError
    except ValueError:
        name = splits_json_path.name
        (
            biosound_group,
            unit,
            frame_dur_1st_half,
            frame_dur_2nd_half,
            train_dur_1st_half,
            train_dur_2nd_half,
            replicate_num,
            _,
            _,
        ) = name.split(".")
        id_ = None
        data_source = None
    if id_ is not None:
        id_ = id_.split("-")[-1]
    frame_dur = float(
        frame_dur_1st_half.split("-")[-1]
        + "."
        + frame_dur_2nd_half.split("-")[0]
    )
    train_dur = float(
        train_dur_1st_half.split("-")[-1]
        + "."
        + train_dur_2nd_half.split("-")[0]
    )
    replicate_num = int(replicate_num.split("-")[-1])
    return TrainingReplicateMetadata(
        biosound_group,
        id_,
        frame_dur,
        unit,
        data_source,
        train_dur,
        replicate_num,
    )


class TrainItemTransform:
    """Default transform used when training frame classification models
    with :class:`BioSoundSegBench` dataset."""

    def __init__(
        self,
        frames_standardizer: FramesStandardizer | None = None,
    ):
        from ..transforms import FramesStandardizer  # avoid circular import

        if frames_standardizer is not None:
            if isinstance(frames_standardizer, FramesStandardizer):
                frames_transform = [frames_standardizer]
            else:
                raise TypeError(
                    f"invalid type for frames_standardizer: {type(frames_standardizer)}. "
                    "Should be an instance of vak.transforms.StandardizeSpect"
                )
        else:
            frames_transform = []
        # add as an attribute on self so that high-level functions can save this class as needed
        self.frames_standardizer = frames_standardizer

        frames_transform.extend(
            [
                transforms.ToFloatTensor(),
                transforms.AddChannel(),
            ]
        )
        self.frames_transform = torchvision.transforms.Compose(
            frames_transform
        )
        self.frame_labels_transform = transforms.ToLongTensor()

    def __call__(
        self,
        frames: torch.Tensor,
        multi_frame_labels: torch.Tensor | None = None,
        binary_frame_labels: torch.Tensor | None = None,
        boundary_frame_labels: torch.Tensor | None = None,
    ) -> dict:
        frames = self.frames_transform(frames)
        item = {
            "frames": frames,
        }
        if multi_frame_labels is not None:
            item["multi_frame_labels"] = self.frame_labels_transform(
                multi_frame_labels
            )

        if binary_frame_labels is not None:
            item["binary_frame_labels"] = self.frame_labels_transform(
                binary_frame_labels
            )

        if boundary_frame_labels is not None:
            item["boundary_frame_labels"] = self.frame_labels_transform(
                boundary_frame_labels
            )

        return item


class InferItemTransform:
    """Default transform used when running inference on classification models
    with :class:`BioSoundSegBench` dataset, for evaluation or to generate new predictions.

    Returned item includes frames reshaped into a stack of windows,
    with padded added to make reshaping possible.
    Any `frame_labels` are not padded and reshaped,
    but are converted to :class:`torch.LongTensor`.
    If return_padding_mask is True, item includes 'padding_mask' that
    can be used to crop off any predictions made on the padding.

    Attributes
    ----------
    frames_standardizer : vak.transforms.FramesStandardizer
        instance that has already been fit to dataset, using fit_df method.
        Default is None, in which case no standardization transform is applied.
    window_size : int
        width of window in number of elements. Argument to PadToWindow transform.
    frames_padval : float
        Value to pad frames with. Added to end of array, the "right side".
        Argument to PadToWindow transform. Default is 0.0.
    frame_labels_padval : int
        Value to pad frame labels vector with. Added to the end of the array.
        Argument to PadToWindow transform. Default is -1.
        Used with ``ignore_index`` argument of :mod:`torch.nn.CrossEntropyLoss`.
    return_padding_mask : bool
        if True, the dictionary returned by ItemTransform classes will include
        a boolean vector to use for cropping back down to size before padding.
        padding_mask has size equal to width of padded array, i.e. original size
        plus padding at the end, and has values of 1 where
        columns in padded are from the original array,
        and values of 0 where columns were added for padding.
    """

    def __init__(
        self,
        window_size,
        frames_standardizer=None,
        frames_padval=0.0,
        frame_labels_padval=-1,
        return_padding_mask=True,
        channel_dim=1,
    ):
        from ..transforms import FramesStandardizer  # avoid circular import

        self.window_size = window_size
        self.frames_padval = frames_padval
        self.frame_labels_padval = frame_labels_padval
        self.return_padding_mask = return_padding_mask
        self.channel_dim = channel_dim

        if frames_standardizer is not None:
            if not isinstance(frames_standardizer, FramesStandardizer):
                raise TypeError(
                    f"Invalid type for frames_standardizer: {type(frames_standardizer)}. "
                    "Should be an instance of vak.transforms.FramesStandardizer"
                )
        # add as an attribute on self to use inside __call__
        # *and* so that high-level functions can save this class as needed
        self.frames_standardizer = frames_standardizer

        self.pad_to_window = transforms.PadToWindow(
            window_size, frames_padval, return_padding_mask=return_padding_mask
        )

        self.frames_transform_after_pad = torchvision.transforms.Compose(
            [
                transforms.ViewAsWindowBatch(window_size),
                transforms.ToFloatTensor(),
                # below, add channel at first dimension because windows become batch
                transforms.AddChannel(channel_dim=channel_dim),
            ]
        )

        self.frame_labels_padval = frame_labels_padval
        self.frame_labels_transform = transforms.ToLongTensor()

    def __call__(
        self,
        frames: torch.Tensor,
        multi_frame_labels: torch.Tensor | None = None,
        binary_frame_labels: torch.Tensor | None = None,
        boundary_frame_labels: torch.Tensor | None = None,
        frames_path=None,
    ) -> dict:
        if self.frames_standardizer:
            frames = self.frames_standardizer(frames)

        if self.pad_to_window.return_padding_mask:
            frames, padding_mask = self.pad_to_window(frames)
        else:
            frames = self.pad_to_window(frames)
            padding_mask = None
        frames = self.frames_transform_after_pad(frames)

        item = {
            "frames": frames,
        }

        if multi_frame_labels is not None:
            item["multi_frame_labels"] = self.frame_labels_transform(
                multi_frame_labels
            )

        if binary_frame_labels is not None:
            item["binary_frame_labels"] = self.frame_labels_transform(
                binary_frame_labels
            )

        if boundary_frame_labels is not None:
            item["boundary_frame_labels"] = self.frame_labels_transform(
                boundary_frame_labels
            )

        if padding_mask is not None:
            item["padding_mask"] = padding_mask

        if frames_path is not None:
            # make sure frames_path is a str, not a pathlib.Path
            item["frames_path"] = str(frames_path)

        return item


class BioSoundSegBench:
    """Class representing BioSoundSegBench dataset."""

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
            # discard windows dimension from shape --
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
