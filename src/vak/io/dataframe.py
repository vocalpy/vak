from datetime import datetime
import os

from crowsetta import Transcriber
import numpy as np

from . import audio, spect
from .. import annotation
from ..annotation import source_annot_map
from ..logging import log_or_print


def from_files(data_dir,
               annot_format=None,
               labelset=None,
               output_dir=None,
               annot_file=None,
               audio_format=None,
               spect_format=None,
               spect_params=None,
               spect_output_dir=None,
               logger=None):
    """prepare a dataset of vocalizations from a directory of audio or spectrogram files containing vocalizations,
    and (optionally) annotation for those files. The dataset is returned as a pandas DataFrame.

    Datasets are used to train neural networks, or for predicting annotations for the dataset itself using a
    trained neural network.

    Parameters
    ----------
    data_dir : str
        path to directory with audio or spectrogram files from which to make dataset
    annot_format : str
        format of annotations. Any format that can be used with the
        crowsetta library is valid. Default is None.
    labelset : set, list
        of str or int, set of labels for vocalizations. Default is None.
        If not None, then files will be skipped where the 'labels' array in the
        corresponding annotation contains labels that are not found in labelset
    output_dir : str
        path to location where data sets should be saved. Default is None,
        in which case data sets is saved in data_dir.
    load_spects : bool
        if True, load spectrograms. If False, return a VocalDataset without spectograms loaded.
        Default is True. Set to False when you want to create a VocalDataset for use
        later, but don't want to load all the spectrograms into memory yet.
    audio_format : str
        format of audio files. One of {'wav', 'cbin'}.
    spect_format : str
        format of array files containing spectrograms as 2-d matrices.
        One of {'mat', 'npz'}.
    annot_file : str
        Path to a single annotation file. Default is None.
        Used when a single file contains annotations for multiple audio files.
    spect_params : dict, vak.config.spect.SpectParamsConfig.
        Parameters for creating spectrograms.
        Default is None (implying that spectrograms are already made).
    spect_output_dir : str
        path to location where spectrogram files should be saved. Default is None,
        in which case it defaults to 'spectrograms_generated_{time stamp}'.

    Other Parameters
    ----------------
    logger : logging.Logger
        instance created by vak.logging.get_logger. Default is None.

    Returns
    -------
    vak_df : pandas.DataFrame
        the dataset prepared from the directory specified

    Notes
    -----
    If dataset is created from audio files, then .spect.npz files will be
    generated from the audio files and saved in output_dir.
    """
    # ---- pre-conditions ----------------------------------------------------------------------------------------------
    if labelset is not None:
        if type(labelset) not in (set, list):
            raise TypeError(
                f"type of labelset must be set or list, but type was: {type(labelset)}"
            )

        if type(labelset) == list:
            labelset_set = set(labelset)
            if len(labelset) != len(labelset_set):
                raise ValueError(
                    'labelset contains repeated elements, should be a set (i.e. all members unique.\n'
                    f'Labelset was: {labelset}'
                )
            else:
                labelset = labelset_set

    if output_dir:
        if not os.path.isdir(output_dir):
            raise NotADirectoryError(
                f'output_dir not found: {output_dir}'
            )
    elif output_dir is None:
        output_dir = data_dir

    if audio_format is None and spect_format is None:
        raise ValueError("Must specify either audio_format or spect_format")

    if audio_format and spect_format:
        raise ValueError("Cannot specify both audio_format and spect_format, "
                         "unclear whether to create spectrograms from audio files or "
                         "use already-generated spectrograms from array files")

    if spect_output_dir:
        if not os.path.isdir(spect_output_dir):
            raise NotADirectoryError(
                f'spect_output_dir not found: {spect_output_dir}'
            )

    if annot_format is not None:
        if annot_file is None:
            annot_files = annotation.files_from_dir(annot_dir=data_dir,
                                                    annot_format=annot_format)
            scribe = Transcriber(annot_format=annot_format)
            annot_list = scribe.from_file(annot_file=annot_files)
        else:
            scribe = Transcriber(annot_format=annot_format)
            annot_list = scribe.from_file(annot_file=annot_file)
    else:  # if annot_format not specified
        annot_list = None

    # ------ if making dataset from audio files, need to make into array files first! ----------------------------------
    if audio_format:
        log_or_print(f'making array files containing spectrograms from audio files in: {data_dir}',
                     logger=logger, level='info')
        audio_files = audio.files_from_dir(data_dir, audio_format)
        if annot_list:
            audio_annot_map = source_annot_map(audio_files, annot_list)
            if labelset:  # then remove annotations with labels not in labelset
                # do this here instead of inside function call so that items get removed
                # from annot_list here and won't cause an error because they're still
                # in this list when we call spect.from_files
                for audio_file, annot in list(audio_annot_map.items()):
                    # loop in a verbose way (i.e. not a comprehension)
                    # so we can give user warning when we skip files
                    annot_labelset = set(annot.seq.labels)
                    # below, set(labels_mapping) is a set of that dict's keys
                    if not annot_labelset.issubset(set(labelset)):
                        # because there's some label in labels that's not in labelset
                        audio_annot_map.pop(audio_file)
                        log_or_print(
                            f'found labels in {annot.annot_file} for {audio_file} not in labels_mapping, '
                             f'skipping audio file: {audio_file}',
                            logger=logger, level='info')
                audio_files = []
                annot_list = []
                for k,v in audio_annot_map.items():
                    audio_files.append(k)
                    annot_list.append(v)

        timenow = datetime.now().strftime('%y%m%d_%H%M%S')
        if spect_output_dir is None:
            spect_output_dir = os.path.join(output_dir,
                                            f'spectrograms_generated_{timenow}')
            os.makedirs(spect_output_dir)
        spect_files = audio.to_spect(audio_format=audio_format,
                                     spect_params=spect_params,
                                     output_dir=spect_output_dir,
                                     audio_files=audio_files,
                                     annot_list=annot_list,
                                     labelset=labelset)
        spect_format = 'npz'
    else:  # if audio format is None
        spect_files = None

    from_files_kwargs = {
        'spect_format': spect_format,
        'labelset': labelset,
        'annot_list': annot_list,
        'annot_format': annot_format,
    }

    if spect_files:
        from_files_kwargs['spect_files'] = spect_files
        log_or_print(f'creating datasetfrom spectrogram files in: {output_dir}', logger=logger, level='info')
    else:
        from_files_kwargs['spect_dir'] = data_dir
        log_or_print(f'creating dataset from spectrogram files in: {data_dir}', logger=logger, level='info')

    vak_df = spect.to_dataframe(**from_files_kwargs, logger=logger)
    return vak_df


def add_split_col(df, split):
    """add a 'split' column to a pandas DataFrame.
    Useful for assigning an entire dataset to the same "split",
    e.g. 'train' or 'predict'.
    All rows in the 'split' column will have the value specified.

    Parameters
    ----------
    df : pandas.DataFrame
        that represents a dataset of vocalizations
    split : str
        string that will be assigned to every row in the added "split" column.
        One of {'train', 'val', 'test', 'predict'}.
    """
    if split not in {'train', 'val', 'test', 'predict'}:
        raise ValueError(
            f"value for split should be one of {{'train', 'val', 'test', 'predict'}}, but was {split}"
        )
    split_col = np.asarray([split for _ in range(len(df))], dtype='object')
    df['split'] = split_col
    return df


def validate_and_get_timebin_dur(df, expected_timebin_dur=None):
    """validate timebin duration for a dataset represented by a pandas DataFrame

    checks that there is a single, unique value for the time bin duration of all
    spectrograms in the dataset, and if so, returns it

    Parameters
    ----------
    df : pandas.Dataframe
        created by dataframe.from_files or spect.to_dataframe
    expected_timebin_dur : float

    Returns
    -------
    timebin_dur : float
        duration of time bins for all spectrograms in the dataset
    """
    timebin_dur = df['timebin_dur'].unique()
    if len(timebin_dur) > 1:
        raise ValueError(
            f'found more than one time bin duration in dataset: {timebin_dur}'
        )
    elif len(timebin_dur) == 1:
        timebin_dur = timebin_dur.item()

    if expected_timebin_dur:
        if timebin_dur != expected_timebin_dur:
            raise ValueError(
                'timebin duration from dataset, {}, did not match expected timebin duration'
            )

    return timebin_dur


def split_dur(df, split):
    """get duration of a split in the dataset"""
    return df[df['split'] == split]['duration'].sum()
