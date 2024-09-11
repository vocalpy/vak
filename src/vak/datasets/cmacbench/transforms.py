"""Default transforms for CMACBench dataset."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Literal

import torch
import torchvision.transforms

from ... import transforms

if TYPE_CHECKING:
    from ...transforms import FramesStandardizer


class TrainItemTransform:
    """Default transform used when training frame classification models
    with :class:`CMACBench` dataset."""

    def __init__(
        self,
        frames_standardizer: FramesStandardizer | None = None,
    ):
        from ...transforms import FramesStandardizer  # avoid circular import

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
    with :class:`CMACBench` dataset, for evaluation or to generate new predictions.

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
        from ...transforms import FramesStandardizer  # avoid circular import

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