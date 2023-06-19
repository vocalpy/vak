from __future__ import annotations

import logging
import pathlib

import crowsetta
import dask.bag as db
from dask.diagnostics import ProgressBar
import numpy as np
import pandas as pd

from ..common import constants, files
from ..common.annotation import map_annotated_to_annot
from ..common.converters import labelset_to_set
from .spectrogram_dataset.audio_helper import files_from_dir


logger = logging.getLogger(__name__)


# constant, used for names of columns in DataFrame below
DF_COLUMNS = [
    "audio_path",
    "annot_path",
    "annot_format",
    "duration",
    "samplerate",
]


def prep_audio_dataset(
    audio_format: str,
    data_dir: list | None = None,
    annot_format: str | None = None,
    labelset: set | None = None,
) -> pd.DataFrame:
    """Convert audio files into a dataset of vocalizations
    represented as a Pandas DataFrame.

    Parameters
    ----------
    audio_format : str
        format of files containing spectrograms. One of {'mat', 'npz'}
    data_dir : str
        path to directory of files containing spectrograms as arrays.
        Default is None.
    annot_format : str
        name of annotation format. Added as a column to the DataFrame if specified.
        Used by other functions that open annotation files via their paths from the DataFrame.
        Should be a format that the crowsetta library recognizes.
        Default is None.
    labelset : str, list, set
        of str or int, set of unique labels for vocalizations. Default is None.
        If not None, then files will be skipped where the associated annotation
        contains labels not found in ``labelset``.
        ``labelset`` is converted to a Python ``set`` using ``vak.converters.labelset_to_set``.
        See help for that function for details on how to specify labelset.

    Returns
    -------
    dataset_df : pandas.Dataframe
        Dataframe that represents a dataset of vocalizations.
    """
    # pre-conditions ---------------------------------------------------------------------------------------------------
    if audio_format not in constants.VALID_AUDIO_FORMATS:
        raise ValueError(
            f"audio format must be one of '{constants.VALID_AUDIO_FORMATS}'; "
            f"format '{audio_format}' not recognized."
        )

    if labelset is not None:
        labelset = labelset_to_set(labelset)

    data_dir = expanded_user_path(data_dir)
    if not data_dir.is_dir():
        raise NotADirectoryError(f"data_dir not found: {data_dir}")

    audio_files = files_from_dir(data_dir, audio_format)

    if annot_format is not None:
        if annot_file is None:
            annot_files = annotation.files_from_dir(
                annot_dir=data_dir, annot_format=annot_format
            )
            scribe = crowsetta.Transcriber(format=annot_format)
            annot_list = [scribe.from_file(annot_file).to_annot() for annot_file in annot_files]
        else:
            scribe = crowsetta.Transcriber(format=annot_format)
            annot_list = scribe.from_file(annot_file).to_annot()
        if isinstance(annot_list, crowsetta.Annotation):
            # if e.g. only one annotated audio file in directory, wrap in a list to make iterable
            # fixes https://github.com/NickleDave/vak/issues/467
            annot_list = [annot_list]
    else:  # if annot_format not specified
        annot_list = None

    if annot_list:
        audio_annot_map = map_annotated_to_annot(audio_files, annot_list, annot_format)
    else:
        # no annotation, so map spectrogram files to None
        audio_annot_map = dict((audio_path, None) for audio_path in audio_files)

    # use mapping (if generated/supplied) with labelset, if supplied, to filter
    if labelset:  # then remove annotations with labels not in labelset
        for audio_file, annot in list(audio_annot_map.items()):
            # loop in a verbose way (i.e. not a comprehension)
            # so we can give user warning when we skip files
            annot_labelset = set(annot.seq.labels)
            # below, set(labels_mapping) is a set of that dict's keys
            if not annot_labelset.issubset(set(labelset)):
                # because there's some label in labels that's not in labelset
                audio_annot_map.pop(audio_file)
                extra_labels = annot_labelset - labelset
                logger.info(
                    f"Found labels, {extra_labels}, in {pathlib.Path(audio_file).name}, "
                    "that are not in labels_mapping. Skipping file.",
                )

    # ---- actually make the dataframe ---------------------------------------------------------------------------------
    # this is defined here so all other arguments to 'to_dataframe' are in scope
    def _to_record(audio_annot_tuple):
        """helper function that enables parallelized creation of "records",
        i.e. rows for dataframe, from .
        Accepts a two-element tuple containing (1) a dictionary that represents a spectrogram
        and (2) annotation for that file"""
        audio_path, annot = audio_annot_tuple
        dat, samplerate = constants.AUDIO_FORMAT_FUNC_MAP[audio_format](audio_file)
        audio_dur = dat.shape[-1] * (1 / samplerate)

        if annot is not None:
            annot_path = annot.annot_path
        else:
            annot_path = np.nan

        def abspath(a_path):
            if isinstance(a_path, str) or isinstance(a_path, pathlib.Path):
                return str(pathlib.Path(a_path).absolute())
            elif np.isnan(a_path):
                return a_path

        record = tuple(
            [
                abspath(audio_path),
                abspath(annot_path),
                annot_format if annot_format else constants.NO_ANNOTATION_FORMAT,
                audio_dur,
                samplerate,
            ]
        )
        return record

    audio_path_annot_tuples = db.from_sequence(audio_annot_map.items())
    logger.info(
        "creating pandas.DataFrame representing dataset from audio files",
    )
    with ProgressBar():
        records = list(audio_path_annot_tuples.map(_to_record))

    return pd.DataFrame.from_records(data=records, columns=DF_COLUMNS)


def make_frame_classification_arrays_from_audio_paths_and_annots(

):
    X_T, Y_T, source_id_vec, inds_in_source_vec, annots = [], [], [], [], []
    for source_id, (audio_path, annot_path) in enumerate(
            zip(audio_paths, annot_paths)
    ):
        # next two lines convert cbin to wav
        audio, sampfreq = evfuncs.load_cbin(audio_path)
        audio = audio.astype(np.float64) / 32768.0
        # make vector of times for labeled timebins
        t = np.arange(audio.shape[-1]) / sampfreq
        X_T.append(audio)

        # add to Y_T
        annot = scribe.from_file(annot_path).to_annot()
        annots.append(annot)  # we use this to save a .csv below
        lbls_int = [labelmap[lbl] for lbl in annot.seq.labels]
        lbl_tb = vak.transforms.labeled_timebins.from_segments(
            lbls_int,
            annot.seq.onsets_s,
            annot.seq.offsets_s,
            t,
            unlabeled_label=labelmap["unlabeled"],
        )
        Y_T.append(lbl_tb)

        # add to source_id and source_inds
        n_frames = t.shape[-1]  # number of frames
        source_id_vec.append(
            np.ones((n_frames,)).astype(np.int32) * source_id
        )
        inds_in_source_vec.append(
            np.arange(n_frames)
        )

    X_T = np.concatenate(X_T)
    Y_T = np.concatenate(Y_T)
    source_id_vec = np.concatenate(source_id_vec)
    inds_in_source_vec = np.concatenate(inds_in_source_vec)
    generic_seq = crowsetta.formats.seq.GenericSeq(annots=annots)


def make_frame_classification_arrays_from_audio_dataset(
        dataset_df: pd.DataFrame,
        dataset_path: pathlib.Path,
        purpose: str,
        labelmap: dict | None = None,
):
    dataset_path = pathlib.Path(dataset_path)

    logger.info(f"Will use labelmap: {labelmap}")

    for split in dataset_df.split:
        if split == 'None':
            # these are files that didn't get assigned to a split
            continue

        logger.info(f"Processing split: {split}")
        split_dst = dataset_path / split
        logger.info(f"Will save in: {split}")
        split_dst.mkdir(exist_ok=True)

        split_df = dataset_df[dataset_df.split == split]
        audio_paths = split_df['audio_path'].values
        if purpose != 'predict':
            annots = common.annotation.from_df(split_df)
        else:
            annots = None

        print(f"Loading data from {len(audio_paths)} spectrogram files "
              f"and {len(annots)} annotations")

        (inputs,
         source_id_vec,
         inds_in_source_vec,
         frame_labels) = make_frame_classification_arrays_from_audio_paths_and_annots(
            audio_paths,
            labelmap,
            annots,
        )

        logger.info(
            f"Saving ``inputs`` for frame classification dataset with size {round(inputs.nbytes * 1e-10, 2)} GB"
        )
        np.save(split_dst / datasets.frame_classification.INPUT_ARRAY_FILENAME, inputs)
        logger.info(
            "Saving ``source_id`` vector for frame classification dataset with size "
            f"{round(source_id_vec.nbytes * 1e-6, 2)} MB"
        )
        np.save(split_dst / datasets.frame_classification.SOURCE_IDS_ARRAY_FILENAME, source_id_vec)
        logger.info(
            "Saving ``inds_in_source_vec`` vector for frame classification dataset "
            f"with size {round(inds_in_source_vec.nbytes * 1e-6, 2)} MB"
        )
        np.save(split_dst / datasets.frame_classification.INDS_IN_SOURCE_ARRAY_FILENAME, inds_in_source_vec)
        if purpose != 'predict':
            logger.info(
                "Saving frame labels vector (targets) for frame classification dataset "
                f"with size {round(frame_labels.nbytes * 1e-6, 2)} MB"
            )
            np.save(split_dst / datasets.frame_classification.FRAME_LABELS_ARRAY_FILENAME, frame_labels)
            logger.info(
                "Saving annotations as csv"
            )
            generic_seq = crowsetta.formats.seq.GenericSeq(annots=annots)
            generic_seq.to_file(split_dst / datasets.frame_classification.ANNOTATION_CSV_FILENAME)
