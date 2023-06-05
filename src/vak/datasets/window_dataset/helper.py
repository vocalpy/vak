"""Helper functions used to create vectors for WindowDataset.
In a separate module because these are pretty verbose functions.
"""
from __future__ import annotations

import pathlib

import numpy as np
import numpy.typing as npt
import pandas as pd
import random

from ... import transforms
from ...common import (
    annotation,
    files,
    validators
)


def crop_vectors_keep_classes(
        lbl_tb: npt.NDArray,
        source_ids: npt.NDArray,
        source_inds: npt.NDArray,
        window_inds: npt.NDArray,
        crop_dur: float,
        timebin_dur: float,
        labelmap: dict,
        window_size: int,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    r"""Crop vectors representing WindowDataset
    to a target duration.

    This function "crops" a WindowDataset
    by shortening the vectors that represent
    valid windows in a way that
    ensures all classes are still present after cropping.
    It first tries to crop from the end of the dataset,
    then from the front,
    and then finally it tries to remove
    unlabeled periods that are at least equal to
    ``window_size`` + 2 time bins, until
    the total duration reaches the target size.
    If none of those approaches can preserve all classes
    in the dataset, the function raises an error.

    Parameters
    ----------
    lbl_tb : numpy.ndarray
        Vector of labels for time bins,
        where labels are from
        the set of values in ``labelmap``.
    source_ids : numpy.ndarray
        Represents the "ID" of any source file,
        i.e., the index into ``spect_paths``
        that will let us load that file.
        For a dataset with :math:`m` files,
        this will be an array of length :math:`T`,
        the total number of time bins across all files,
        with elements :math:`i in (0, 1, ..., m - 1)`
        indicating which time bins
        correspond to which file :math:`m_i`:
         :math:`(0, 0, 0, ..., 1, 1, ..., m - 1, m -1)`.
    source_inds : numpy.ndarray
        Same length as ``source_ids`` but values represent
        indices of time bins within each audio array or spectrogram.
        For a data set with :math:`T` total time bins across all files,
        where :math:`t_i` indicates the number of time bins
        in each file :math:`m_i`,
        this will look like
        :math:`(0, 1, ..., t_0, 0, 1, ..., t_1, ... t_m)`.
    window_inds : numpy.ndarray
        Starting indices of each valid window in the dataset.
        The value at ``window_inds[0]``
        represents the start index of the first window; using that
        value, we can index into ``source_ids`` to get the path
        of the audio or spectrogram file to load, and
        we can index into ``source_inds``
        to get a window from the audio or spectrogram itself.
        ``window_inds`` will always be strictly shorter than ``source_ids`` and
        ``source_inds``, because the number of valid time bins in
        each file :math:`m_i` will be at most :math:`t_i - \text{window size}`,
        and cropping to a specified duration will remove
        additional time bins.
    crop_dur : float
        Duration to which dataset should be "cropped", in seconds.
    timebin_dur : float
        For a dataset of audio,
        the duration of a single sample,
        i.e., the inverse of the sampling rate given in Hertz.
        For a dataset of spectrograms,
        the duration of a single time bin in the spectrograms.
    labelmap : dict
        Dict that maps labels from dataset
        to a series of consecutive integers.
        To create a label map, pass a set of labels
        to the ``vak.utils.labels.to_map`` function.

    Returns
    -------
    x_id_cropped : numpy.ndarray
        ``source_ids`` after cropping.
    x_inds_cropped : numpy.ndarray
        ``source_inds`` after cropping.
    window_inds_updated : numpy.ndarray
        ``window_inds_vector`` with indices that are invalid
        after cropping now set to the value
        ``WindowDataset.INVALID_WINDOW_VAL``,
        so that they will be removed.
    """
    from .class_ import WindowDataset  # avoid circular import

    # ---- pre-conditions
    lbl_tb = validators.column_or_1d(lbl_tb)
    source_ids = validators.column_or_1d(source_ids)
    source_inds = validators.column_or_1d(source_inds)
    window_inds = validators.column_or_1d(window_inds)

    lens = (
        lbl_tb.shape[-1],
        source_ids.shape[-1],
        source_inds.shape[-1],
        window_inds.shape[-1],
    )
    uniq_lens = set(lens)
    if len(uniq_lens) != 1:
        raise ValueError(
            "lbl_tb, source_ids, source_inds, and window_inds should all "
            "have the same length, but did not find one unique length. "
            "Lengths of lbl_tb, source_ids, source_inds, and window_inds_vector "
            f"were: {lens}"
        )

    # ---- compute target length in number of time bins
    cropped_length = np.round(crop_dur / timebin_dur).astype(int)

    if source_ids.shape[-1] == cropped_length:
        return source_ids, source_inds, window_inds

    elif source_ids.shape[-1] < cropped_length:
        raise ValueError(
            f"arrays have length {source_ids.shape[-1]} "
            f"that is shorter than correct length, {cropped_length}, "
            f"(= target duration {crop_dur} / duration of timebins, {timebin_dur})."
        )

    # ---- Actual cropping logic starts here ----------------------------------------

    classes = np.asarray(sorted(list(labelmap.values())))

    # ---- try cropping off the end first
    lbl_tb_cropped = lbl_tb[:cropped_length]

    if np.array_equal(np.unique(lbl_tb_cropped), classes):
        window_inds[cropped_length:] = WindowDataset.INVALID_WINDOW_VAL
        return (
            source_ids[:cropped_length],
            source_inds[:cropped_length],
            window_inds,
        )

    # ---- try truncating off the front instead
    lbl_tb_cropped = lbl_tb[-cropped_length:]
    if np.array_equal(np.unique(lbl_tb_cropped), classes):
        # set every index *up to but not including* the first valid window start to "invalid"
        window_inds[:-cropped_length] = WindowDataset.INVALID_WINDOW_VAL
        # also need to 'reset' the indexing so it starts at 0. First find current minimum index value
        min_x_ind = window_inds[window_inds != WindowDataset.INVALID_WINDOW_VAL].min()
        # Then set min x ind to 0, min x ind + 1 to 1, min ind + 2 to 2, ...
        window_inds[window_inds != WindowDataset.INVALID_WINDOW_VAL] = (
                window_inds[window_inds != WindowDataset.INVALID_WINDOW_VAL] - min_x_ind
        )
        return (
            source_ids[-cropped_length:],
            source_inds[-cropped_length:],
            window_inds,
        )

    # ---- try cropping silences
    # This is done by seeking segments > (window_size + 2 bins) and removing them.
    # When using this option we do not crop the spect vector sizes

    # Ignored data is defined as data that does not appear in any training window.
    # This means that there are 3 distinct cases:
    # 1. Silence in the beginning of a file
    # 2. Silence in the middle of a file
    # 3. Silence at the end of the file
    #    where 'end' means the segment prior to the last ``window_size`` bins in the file,
    #    because those cannot be the start of a training window (we wouldn't have enough bins)

    # assigning WindowDataset.INVALID_WINDOW_VAL to window_inds segments
    # in these 3 cases will cause data to be ignored with
    # durations that depend on whether the segments touch the ends of files
    # because we do not ignore non-silence segments.

    # first identify all silence segments larger than the window duration + 2
    if "unlabeled" in labelmap:
        unlabeled = labelmap["unlabeled"]
    else:
        raise ValueError(
            "Was not able to crop x vectors to specified duration; "
            "could not crop from start or end, and there are no "
            "unlabeled segments long enough to use to further crop."
        )
    valid_unlabeled = np.logical_and(
        lbl_tb == unlabeled, window_inds != WindowDataset.INVALID_WINDOW_VAL
    )
    unlabeled_diff = np.diff(np.concatenate([[0], valid_unlabeled, [0]]))
    unlabeled_onsets = np.where(unlabeled_diff == 1)[0]
    unlabeled_offsets = np.where(unlabeled_diff == -1)[0]
    unlabeled_durations = unlabeled_offsets - unlabeled_onsets
    N_PAD_BINS = 2
    unlabeled_onsets = unlabeled_onsets[
        unlabeled_durations >= window_size + N_PAD_BINS
        ]
    unlabeled_offsets = unlabeled_offsets[
        unlabeled_durations >= window_size + N_PAD_BINS
        ]
    unlabeled_durations = unlabeled_durations[
        unlabeled_durations >= window_size + N_PAD_BINS
        ]
    # indicate silences in the beginning of files
    border_onsets = (
            np.concatenate([[WindowDataset.INVALID_WINDOW_VAL], window_inds])[
                unlabeled_onsets
            ]
            == WindowDataset.INVALID_WINDOW_VAL
    )
    # indicate silences at the end of files
    border_offsets = (
            np.concatenate([window_inds, [WindowDataset.INVALID_WINDOW_VAL]])[
                unlabeled_offsets + 1
                ]
            == WindowDataset.INVALID_WINDOW_VAL
    )

    # This is how much data can be ignored from each silence segment
    # without ignoring the end of file windows
    num_potential_ignored_data_bins = (
            unlabeled_durations
            - (window_size + N_PAD_BINS)
            + window_size * border_onsets
    )

    num_bins_to_crop = len(lbl_tb) - cropped_length
    if sum(num_potential_ignored_data_bins) < num_bins_to_crop:
        # This is how much data can be ignored from each silence segment including the end of file windows
        num_potential_ignored_data_bins = (
                unlabeled_durations
                - (window_size - N_PAD_BINS)
                + window_size * (border_onsets + border_offsets)
        )
    else:
        border_offsets[:] = False

    # Second we find a ~random combination to remove
    crop_more = 0
    if sum(num_potential_ignored_data_bins) < num_bins_to_crop:
        # if we will still need to crop more we will do so from non-silence segments
        crop_more = num_bins_to_crop - sum(num_potential_ignored_data_bins) + 1
        num_bins_to_crop = sum(num_potential_ignored_data_bins) - 1

    segment_ind = np.arange(len(num_potential_ignored_data_bins))
    random.shuffle(segment_ind)
    last_ind = np.where(
        np.cumsum(num_potential_ignored_data_bins[segment_ind])
        >= num_bins_to_crop
    )[0][0]
    bins_to_ignore = np.array([], dtype=int)
    for cnt in range(last_ind):
        if border_onsets[segment_ind[cnt]]:  # remove silences at file onsets
            bins_to_ignore = np.concatenate(
                [
                    bins_to_ignore,
                    np.arange(
                        unlabeled_onsets[segment_ind[cnt]],
                        unlabeled_offsets[segment_ind[cnt]] - 1,
                    ),
                ]
            )
        elif border_offsets[
            segment_ind[cnt]
        ]:  # remove silences at file offsets
            bins_to_ignore = np.concatenate(
                [
                    bins_to_ignore,
                    np.arange(
                        unlabeled_onsets[segment_ind[cnt]] + 1,
                        unlabeled_offsets[segment_ind[cnt]],
                    ),
                ]
            )
        else:  # remove silences within the files
            bins_to_ignore = np.concatenate(
                [
                    bins_to_ignore,
                    np.arange(
                        unlabeled_onsets[segment_ind[cnt]] + 1,
                        unlabeled_offsets[segment_ind[cnt]] - 1,
                    ),
                ]
            )
    left_to_crop = (
            num_bins_to_crop
            - sum(num_potential_ignored_data_bins[segment_ind[:last_ind]])
            - border_onsets[segment_ind[last_ind]] * window_size
    )
    if border_onsets[segment_ind[last_ind]]:
        bins_to_ignore = np.concatenate(
            [
                bins_to_ignore,
                np.arange(
                    unlabeled_onsets[segment_ind[last_ind]],
                    unlabeled_onsets[segment_ind[last_ind]] + left_to_crop,
                ),
            ]
        )
    elif border_offsets[segment_ind[last_ind]]:
        if (
                left_to_crop
                < num_potential_ignored_data_bins[segment_ind[last_ind]]
                - window_size
        ):
            bins_to_ignore = np.concatenate(
                [
                    bins_to_ignore,
                    np.arange(
                        unlabeled_onsets[segment_ind[last_ind]] + 1,
                        unlabeled_onsets[segment_ind[last_ind]] + left_to_crop,
                    ),
                ]
            )
        else:
            bins_to_ignore = np.concatenate(
                [
                    bins_to_ignore,
                    np.arange(
                        unlabeled_onsets[segment_ind[last_ind]] + 1,
                        unlabeled_onsets[segment_ind[last_ind]]
                        + left_to_crop
                        - window_size,
                    ),
                ]
            )
    else:
        bins_to_ignore = np.concatenate(
            [
                bins_to_ignore,
                np.arange(
                    unlabeled_onsets[segment_ind[last_ind]] + 1,
                    unlabeled_onsets[segment_ind[last_ind]] + left_to_crop,
                ),
            ]
        )

    window_inds[bins_to_ignore] = WindowDataset.INVALID_WINDOW_VAL

    # we may still need to crop. Try doing it from the beginning of the dataset
    if (
            crop_more > 0
    ):  # This addition can lead to imprecision but only in cases where we ask for very small datasets
        if crop_more > sum(window_inds != WindowDataset.INVALID_WINDOW_VAL):
            raise ValueError(
                "was not able to crop spect vectors to specified duration "
                "in a way that maintained all classes in dataset"
            )
        extra_bins = window_inds[window_inds != WindowDataset.INVALID_WINDOW_VAL][
                     :crop_more
                     ]
        bins_to_ignore = np.concatenate([bins_to_ignore, extra_bins])
        window_inds[bins_to_ignore] = WindowDataset.INVALID_WINDOW_VAL

    if np.array_equal(
            np.unique(lbl_tb[np.setdiff1d(np.arange(len(lbl_tb)), bins_to_ignore)]),
            classes,
    ):
        return source_ids, source_inds, window_inds

    raise ValueError(
        "was not able to crop spect vectors to specified duration "
        "in a way that maintained all classes in dataset"
    )


def _vectors_from_df(
        df: pd.DataFrame,
        dataset_path: str | pathlib.Path,
        labelmap: dict,
        crop_to_dur: bool,
        window_size: int,
        spect_key: str = "s",
        timebins_key: str = "t",
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray | None]:
    """Helper function that generates WindowDataset-related vectors
    *without* removing the elements from ``window_inds`` that are set to
    ``WindowDataset.INVALID_WINDOW_VAL``.

    This lets us call this method from within tests,
    e.g. in unit tests for ``crop_vectors_keep_classes``
    instead of essentially copy-pasting the whole
    ``vectors_from_df`` method into a single unit test.

    Note that this ``df`` **only** contains the split we care about,
    e.g., the calling function should pass in
    ``vak_df[vak_df.split == 'split_of_interest']``.
    """
    from .class_ import WindowDataset

    # TODO: when we add `x_source`, use `audio_path` here if `x_source == 'audio'`
    source_paths = df["spect_path"].values

    source_ids = []
    source_inds = []
    window_inds = []
    total_tb = 0
    if crop_to_dur:
        # then we need labeled timebins so we can ensure unique labelset is preserved when cropping
        lbl_tb = []
        annots = annotation.from_df(df, dataset_path)
        annot_format = annotation.format_from_df(df)

        if all([source_path.endswith('.spect.npz') for source_path in source_paths]):
            annotated_ext = '.spect.npz'
        else:
            annotated_ext_set = set([annotated_file.suffix for annotated_file in annotated_files])
            if len(annotated_ext_set) > 1:
                raise ValueError(
                    "Found more than one extension in annotated files, "
                    "unclear which extension to use when mapping to annotations "
                    f"with 'replace' method. Extensions found: {ext_set}"
                )
            annotated_ext = annotated_ext_set.pop()

        source_annot_map = annotation.map_annotated_to_annot(source_paths, annots, annot_format,
                                                             annotated_ext=annotated_ext)
        unlabeled_label = labelmap.get('unlabeled', 0)

    # this avoids repeating the entire for loop twice, for with cropping or without;
    # it's more DRY but a bit confusing to read at first
    if crop_to_dur:
        to_do = enumerate(source_annot_map.items())
    else:
        to_do = enumerate(source_paths)

    for ind, to_do_item in to_do:
        if crop_to_dur:
            source_path, annot = to_do_item
        else:
            source_path = to_do_item

        spect_dict = files.spect.load(dataset_path / source_path)
        n_tb = spect_dict[spect_key].shape[-1]

        source_ids.append(np.ones((n_tb,), dtype=np.int64) * ind)
        source_inds.append(np.arange(n_tb))

        valid_window_inds = np.arange(total_tb, total_tb + n_tb)
        last_valid_window_ind = total_tb + n_tb - window_size
        valid_window_inds[
            valid_window_inds > last_valid_window_ind
            ] = WindowDataset.INVALID_WINDOW_VAL
        window_inds.append(valid_window_inds)

        total_tb += n_tb

        if crop_to_dur:  # get labeled timebins so we can preserve all classes when cropping
            lbls_int = [labelmap[lbl] for lbl in annot.seq.labels]

            timebins = spect_dict[timebins_key]
            lbl_tb.append(
                transforms.labeled_timebins.from_segments(
                    lbls_int,
                    annot.seq.onsets_s,
                    annot.seq.offsets_s,
                    timebins,
                    unlabeled_label=unlabeled_label,
                )
            )

    source_ids = np.concatenate(source_ids)
    source_inds = np.concatenate(source_inds)
    window_inds = np.concatenate(window_inds)

    if crop_to_dur:
        lbl_tb = np.concatenate(lbl_tb)
    else:
        lbl_tb = None

    return source_ids, source_inds, window_inds, lbl_tb


def vectors_from_df(
        df: pd.DataFrame,
        dataset_path: str | pathlib.Path,
        split: str,
        window_size: int,
        spect_key: str = "s",
        timebins_key: str = "t",
        crop_dur: int | float | None = None,
        timebin_dur: float | None = None,
        labelmap: dict | None = None,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    r"""Get source_ids and spect_ind_vector from a dataframe
    that represents a dataset of vocalizations.

    See ``vak.datasets.WindowDataset`` for a
    detailed explanation of these vectors.

    Parameters
    ----------
    df : pandas.DataFrame
        That represents a dataset of vocalizations.
    dataset_path : str
        Path to dataset, a directory generated by running ``vak prep``.
    window_size : int
        Size of the window, in number of time bins,
        that is taken from the audio array
        or spectrogram to become a training sample.
    spect_key : str
        Key to access spectograms in array files.
        Default is "s".
    timebins_key : str
        Key to access time bin vector in array files.
        Default is "t".
    crop_dur : float
        Duration to which dataset should be "cropped". Default is None,
        in which case entire duration of specified split will be used.
    timebin_dur : float
        Duration of a single time bin in spectrograms. Default is None.
        Used when "cropping" dataset with ``crop_dur``, and required if a
        value is specified for that parameter.
    labelmap : dict
        Dict that maps labels from dataset to a series of consecutive integers.
        To create a label map, pass a set of labels to the `vak.utils.labels.to_map` function.
        Used when "cropping" dataset with ``crop_dur``
        to make sure all labels in ``labelmap`` are still
        in the dataset after cropping.
        Required if a  value is specified for ``crop_dur``.

    Returns
    -------
    source_ids : numpy.ndarray
        Represents the "id" of any spectrogram,
        i.e., the index into spect_paths that will let us load it.
    source_inds : numpy.ndarray
        Valid indices of windows we can grab from each
        audio array or spectrogram.
    window_inds : numpy.ndarray
        Vector of all valid starting indices of all windows in the dataset.
        This vector is what is used by PyTorch to determine
        the number of samples in the dataset, via the
        ``WindowDataset.__len__`` method.
        Without cropping, a dataset with ``t`` total time bins
        across all audio arrays or spectrograms will have
        (``t`` - ``window_size``) possible windows
        with indices (0, 1, 2, ..., t-1).
        But cropping with ``crop_dur`` will
        remove some of these indices.
    """
    from .class_ import WindowDataset  # avoid circular import

    dataset_path = pathlib.Path(dataset_path)
    if not dataset_path.exists() or not dataset_path.is_dir():
        raise NotADirectoryError(
            f"`dataset_path` not found or not recognized as a directory: {dataset_path}"
        )

    if crop_dur is not None and timebin_dur is None:
        raise ValueError("Must provide timebin_dur when specifying crop_dur")

    if crop_dur is not None and labelmap is None:
        raise ValueError("Must provide labelmap when specifying crop_dur")

    if split not in WindowDataset.VALID_SPLITS:
        raise ValueError(
            f"Invalid value for split: {split}. Valid split names are: {WindowDataset.VALID_SPLITS}"
        )

    if crop_dur is not None and timebin_dur is not None:
        crop_to_dur = True
        crop_dur = float(crop_dur)
        timebin_dur = float(timebin_dur)
    else:
        crop_to_dur = False

    if "split" == "all":
        pass  # use all rows, don't select by split
    else:
        if split not in df.split.unique().tolist():
            raise ValueError(
                f"Split {split} not found in dataframe. Splits in dataframe are: {df.split.unique().tolist()}"
            )
        df = df[df["split"] == split]

    (source_ids,
     source_inds,
     window_inds,
     lbl_tb) = _vectors_from_df(
        df,
        dataset_path,
        labelmap,
        crop_to_dur,
        window_size,
        spect_key,
        timebins_key,
    )

    if crop_to_dur:
        (
            source_ids,
            source_inds,
            window_inds,
        ) = crop_vectors_keep_classes(
            lbl_tb,
            source_ids,
            source_inds,
            window_inds,
            crop_dur,
            timebin_dur,
            labelmap,
            window_size,
        )

    window_inds = window_inds[window_inds != WindowDataset.INVALID_WINDOW_VAL]
    return source_ids, source_inds, window_inds
