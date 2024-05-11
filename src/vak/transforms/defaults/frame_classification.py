"""Default transforms for frame classification models.

These are "item" transforms because they apply transforms to input parameters
and then return them in an "item" (dictionary)
that is turn returned by the __getitem__ method of a vak.InferDatapipe.
Having the transform return a dictionary makes it possible to avoid
coupling the InferDatapipe __getitem__ implementation to the transforms
needed for specific neural network models, e.g., whether the returned
output includes a mask to crop off padding that was added.
"""

from __future__ import annotations

import torch
import torchvision.transforms

from .. import transforms as vak_transforms
from ..transforms import FramesStandardizer


class TrainItemTransform:
    """Default transform used when training frame classification models"""

    def __init__(
        self,
        frames_standardizer: FramesStandardizer | None = None,
    ):
        if frames_standardizer is not None:
            if isinstance(
                frames_standardizer, vak_transforms.FramesStandardizer
            ):
                frames_transform = [frames_standardizer]
            else:
                raise TypeError(
                    f"invalid type for frames_standardizer: {type(frames_standardizer)}. "
                    "Should be an instance of vak.transforms.StandardizeSpect"
                )
        else:
            frames_transform = []

        frames_transform.extend(
            [
                vak_transforms.ToFloatTensor(),
                vak_transforms.AddChannel(),
            ]
        )
        self.frames_transform = torchvision.transforms.Compose(
            frames_transform
        )
        self.frame_labels_transform = vak_transforms.ToLongTensor()

    def __call__(self, frames, frame_labels, spect_path=None):
        frames = self.frames_transform(frames)
        frame_labels = self.frame_labels_transform(frame_labels)
        item = {
            "frames": frames,
            "multi_frame_labels": frame_labels,
        }

        if spect_path is not None:
            item["spect_path"] = spect_path

        return item


class InferItemTransform:
    """Default transform used when running inference on frame classification models,
    for evaluation or to generate new predictions.

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
        self.window_size = window_size
        self.frames_padval = frames_padval
        self.frame_labels_padval = frame_labels_padval
        self.return_padding_mask = return_padding_mask
        self.channel_dim = channel_dim

        if frames_standardizer is not None:
            if not isinstance(
                frames_standardizer, vak_transforms.FramesStandardizer
            ):
                raise TypeError(
                    f"Invalid type for frames_standardizer: {type(frames_standardizer)}. "
                    "Should be an instance of vak.transforms.FramesStandardizer"
                )
        self.frames_standardizer = frames_standardizer

        self.pad_to_window = vak_transforms.PadToWindow(
            window_size, frames_padval, return_padding_mask=return_padding_mask
        )

        self.frames_transform_after_pad = torchvision.transforms.Compose(
            [
                vak_transforms.ViewAsWindowBatch(window_size),
                vak_transforms.ToFloatTensor(),
                # below, add channel at first dimension because windows become batch
                vak_transforms.AddChannel(channel_dim=channel_dim),
            ]
        )

        self.frame_labels_padval = frame_labels_padval
        self.frame_labels_transform = vak_transforms.ToLongTensor()

    def __call__(
        self,
        frames: torch.Tensor,
        frame_labels: torch.Tensor | None = None,
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

        if frame_labels is not None:
            frame_labels = self.frame_labels_transform(frame_labels)
            item["multi_frame_labels"] = frame_labels

        if padding_mask is not None:
            item["padding_mask"] = padding_mask

        if frames_path is not None:
            # make sure frames_path is a str, not a pathlib.Path
            item["frames_path"] = str(frames_path)

        return item
