"""Helper functions for frame classification dataset prep."""
from __future__ import annotations

import collections
import copy
import logging
import pathlib

import attrs
import crowsetta
import dask.bag as db
from dask.diagnostics import ProgressBar
import numpy as np
import pandas as pd

from ... import (
    common,
    datasets,
    transforms
)


logger = logging.getLogger(__name__)


def argsort_by_label_freq(
        annots: list[crowsetta.Annotation]
) -> list[int]:
    """Returns indices to sort a list of annotations
    in order of more frequently appearing labels,
    i.e., the first annotation will have the label
    that appears least frequently and the last annotation
    will have the label that appears most frequently.

   Used to sort a dataframe representing a dataset of annotated audio
   or spectrograms before cropping that dataset to a specified duration,
   so that it's less likely that cropping will remove all occurrences
   of any label class from the total dataset.

    Parameters
    ----------
    annots: list
        List of :class:`crowsetta.Annotation` instances.

    Returns
    -------
    sort_inds: list
        Integer values to sort ``annots``.
    """
    all_labels = [
        lbl for annot in annots for lbl in annot.seq.labels
    ]
    label_counts = collections.Counter(all_labels)

    sort_inds = []
    # make indices ahead of time so they stay constant as we remove things from the list
    ind_annot_tuples = list(
        enumerate(copy.deepcopy(annots))
    )
    for label, _ in reversed(label_counts.most_common()):
        # next line, [:] to make a temporary copy to avoid remove bug
        for ind_annot_tuple in ind_annot_tuples[:]:
            ind, annot = ind_annot_tuple
            if label in annot.seq.labels.tolist():
                sort_inds.append(ind)
                ind_annot_tuples.remove(ind_annot_tuple)

    # make sure we got all source_paths + annots
    if len(ind_annot_tuples) > 0:
        for ind_annot_tuple in ind_annot_tuples:
            ind, annot = ind_annot_tuple
            sort_inds.append(ind)
            ind_annot_tuples.remove(ind_annot_tuple)

    if len(ind_annot_tuples) > 0:
        raise ValueError(
            "Not all ``annots`` were used in sorting."
            f"Left over (with indices from list): {ind_annot_tuples}"
        )

    if not (
        sorted(sort_inds) == list(range(len(annots)))
    ):
        raise ValueError(
            "sorted(sort_inds) does not equal range(len(annots)):"
            f"sort_inds: {sort_inds}\nrange(len(annots)): {list(range(len(annots)))}"
        )

    return sort_inds


@attrs.define(frozen=True)
class SplitRecord:
    source_id: int = attrs.field()
    frame_npy_path: str
    frame_labels_npy_path: str
    source_id_vec: np.ndarray
    inds_in_source_vec: np.ndarray


def make_npy_files_for_each_split(
        dataset_df: pd.DataFrame,
        dataset_path: str | pathlib.Path,
        input_type: str,
        purpose: str,
        labelmap: dict,
        audio_format: str,
        spect_key: str = 's',
        timebins_key: str = 't',
):
    dataset_df_out = []
    splits = [
        split
        for split in sorted(dataset_df.split.dropna().unique())
        if split != "None"
    ]
    for split in splits:
        split_subdir = dataset_path / split
        split_subdir.mkdir()

        split_df = dataset_df[dataset_df.split == split].copy()

        if purpose != 'predict':
            annots = common.annotation.from_df(split_df)
        else:
            annots = None

        if annots:
            sort_inds = argsort_by_label_freq(annots)
            split_df['sort_inds'] = sort_inds
            split_df = split_df.sort_values(by='sort_inds').drop(columns='sort_inds').reset_index()

        if input_type == 'audio':
            source_paths = split_df['audio_path'].values
        elif input_type == 'spect':
            source_paths = split_df['spect_path'].values
        else:
            raise ValueError(
                f"Invalid ``input_type``: {input_type}"
            )
        # do this *again* after sorting the dataframe
        if purpose != 'predict':
            annots = common.annotation.from_df(split_df)
        else:
            annots = None

        def _save_dataset_arrays_and_return_index_arrays(
                source_id_path_annot_tup
        ):
            """Function we use with dask to parallelize

            Defined in-line so variables are in scope
            """
            source_id, source_path, annot = source_id_path_annot_tup
            source_path = pathlib.Path(source_path)

            if input_type == 'audio':
                frames, samplefreq = common.constants.AUDIO_FORMAT_FUNC_MAP[audio_format](source_path)
                if annot:
                    frame_times = np.arange(frames.shape[-1]) / samplefreq
            elif input_type == 'spect':
                spect_dict = np.load(source_path)
                frames = spect_dict[spect_key]
                if annot:
                    frame_times = spect_dict[timebins_key]
            frames_npy_path = split_subdir / (source_path.stem + '.frames.npy')
            np.save(frames, frames_npy_path)
            frames_npy_path = str(
                # make sure we save path in csv as relative to dataset root
                frames_npy_path.relative_to(dataset_path)
            )

            n_frames = frames.shape[-1]
            source_id_vec = np.ones((n_frames,)).astype(np.int32) * source_id
            inds_in_source_vec = np.arange(n_frames)

            # add to frame labels
            if annot:
                lbls_int = [labelmap[lbl] for lbl in annot.seq.labels]
                frame_labels = transforms.labeled_timebins.from_segments(
                    lbls_int,
                    annot.seq.onsets_s,
                    annot.seq.offsets_s,
                    frame_times,
                    unlabeled_label=labelmap["unlabeled"],
                )
                frame_labels_npy_path = split_subdir / (source_path.stem + '.frame_labels.npy')
                np.save(frame_labels, frame_labels_npy_path)
                frame_labels_npy_path = str(
                    # make sure we save path in csv as relative to dataset root
                    frame_labels_npy_path.relative_to(dataset_path)
                )
            else:
                frame_labels_npy_path = None

            return SplitRecord(
                source_id,
                frames_npy_path,
                frame_labels_npy_path,
                source_id_vec,
                inds_in_source_vec
            )

        # ---- make npy files for this split, parallelized with dask
        # using nested function just defined
        if annots:
            source_path_annot_tups = [
                (source_id, source_path, annot)
                for source_id, (source_path, annot) in enumerate(zip(source_paths, annots))
            ]
        else:
            source_path_annot_tups = [
                (source_id, source_path, None)
                for source_id, source_path in enumerate(source_paths)
            ]

        source_path_annot_bag = db.from_sequence(source_path_annot_tups)
        # logger.info(
        #     "creating pandas.DataFrame representing dataset from audio files",
        # )
        with ProgressBar():
            split_records = list(source_path_annot_bag.map(
                _save_dataset_arrays_and_return_index_arrays
            ))
        split_records = sorted(split_records, key=lambda record: record.source_id)

        # ---- save indexing vectors in split directory
        source_id_vec = np.concatenate(
            (record.source_id_vec for record in split_records)
        )
        np.save(
            source_id_vec, split_subdir / datasets.frame_classification.constants.SOURCE_IDS_ARRAY_FILENAME
        )
        inds_in_source_vec = np.concatenate(
            (record.inds_in_source_vec for record in split_records)
        )
        np.save(
            inds_in_source_vec, split_subdir / datasets.frame_classification.constants.INDS_IN_SOURCE_NPY_FILE
        )

        frame_npy_paths = [
            str(record.frame_npy_path) for record in split_records
        ]
        split_df['frame_npy_paths'] = frame_npy_paths

        frame_labels_npy_paths = [
            str(record.frame_labels_npy_path) for record in split_records
        ]
        split_df['frame_label_npy_paths'] = frame_labels_npy_paths
        dataset_df_out.append(split_df)

    dataset_df_out = pd.concat(dataset_df_out)
    return dataset_df_out



# TODO: do we want to save annotations as csv still, to be able to get labels?
        # logger.info(
        #     "Saving ``inputs`` vector for frame classification dataset with size "
        #     f"{round(inputs.nbytes * 1e-6, 2)} MB"
        # )
        # np.save(split_dst / datasets.frame_classification.constants.INPUT_ARRAY_FILENAME, inputs)
        # logger.info(
        #     "Saving ``source_id`` vector for frame classification dataset with size "
        #     f"{round(source_id_vec.nbytes * 1e-6, 2)} MB"
        # )
        # np.save(split_dst / datasets.frame_classification.constants.SOURCE_IDS_ARRAY_FILENAME,
        #         source_id_vec)
        # if purpose != 'predict':
        #     logger.info(
        #         "Saving frame labels vector (targets) for frame classification dataset "
        #         f"with size {round(frame_labels.nbytes * 1e-6, 2)} MB"
        #     )
        #     np.save(split_dst / datasets.frame_classification.constants.FRAME_LABELS_ARRAY_FILENAME, frame_labels)
        #     logger.info(
        #         "Saving annotations as csv"
        #     )
        #     generic_seq = crowsetta.formats.seq.GenericSeq(annots=annots)
        #     generic_seq.to_file(split_dst / datasets.frame_classification.constants.ANNOTATION_CSV_FILENAME)
