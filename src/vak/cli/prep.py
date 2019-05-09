import os
import logging
from configparser import ConfigParser
from datetime import datetime

from crowsetta import Transcriber

from ..utils.data import make_data_dicts
from .. import dataset


def prep(labelset,
         data_dir,
         total_train_set_dur,
         val_dur,
         test_dur,
         config_file,
         annot_format,
         silent_gap_label=0,
         skip_files_with_labels_not_in_labelset=True,
         output_dir=None,
         audio_format=None,
         array_format=None,
         annot_file=None,
         spect_params=None):
    """prepare datasets for training, validating, and/or testing networks

    Parameters
    ----------
    labelset : list
        of str or int, set of labels for syllables
    all_labels_are_int : bool
        if True, labels are of type int, not str
    data_dir : str
        path to directory with audio files from which to make dataset
    total_train_set_dur : float
        total duration of training set, in seconds.
        Training subsets of shorter duration will be drawn from this set.
    val_dur : float
        total duration of validation set, in seconds.
    test_dur : float
        total duration of test set, in seconds.
    silent_gap_label : int
        label for time bins of silent gaps between syllables.
        Type is int because labels are converted to a set of
        n consecutive integers {0,1,2...n} where n is the number
        of syllable classes + the silent gap class.
        Default is 0 (in which case labels are {1,2,3,...,n}).
    skip_files_with_labels_not_in_labelset : bool
        if True, skip a file if the labels variable contains labels not
        found in 'labelset'. Default is True.
    output_dir : str
        Path to location where data sets should be saved. Default is None,
        in which case data sets are saved in the current working directory.
    audio_format : str
        format of audio files. One of {'wav', 'cbin'}.
    array_format : str
        format of array files containing spectrograms as 2-d matrices.
        One of {'mat', 'npz'}.
    annot_format : str
        format of annotations. Any format that can be used with the
        crowsetta library is valid.
    annot_file : str
        Path to a single annotation file. Default is None.
        Used when a single file contains annotations for multiple audio files.
    spect_params : dict
        Dictionary of parameters for creating spectrograms.
        Default is None (implying that spectrograms are already made).

    Returns
    -------
    None

    Saves .spect files and data_dicts in output_dir specified by user.
    """
    if audio_format is None and array_format is None:
        raise ValueError("Must specify either audio_format or array_format")

    if audio_format and array_format:
        raise ValueError("Cannot specify both audio_format and array_format, "
                         "unclear whether to create spectrograms from audio files or "
                         "use already-generated spectrograms from array files")

    logger = logging.getLogger(__name__)
    logger.setLevel('INFO')

    # # map labels to series of consecutive integers from 0 to n inclusive
    # # where 0 is the label for silent periods between syllables
    # # and n is the number of syllable labels
    # if all_labels_are_int:
    #     labels_mapping = dict(zip([int(label) for label in labelset],
    #                               range(1, len(labelset) + 1)))
    # else:
    #     labels_mapping = dict(zip(labelset,
    #                               range(1, len(labelset) + 1)))
    # labels_mapping['silent_gap_label'] = silent_gap_label
    # if sorted(labels_mapping.values()) != list(range(len(labels_mapping))):
    #     raise ValueError('Labels mapping does not map to a consecutive'
    #                      'series of integers from 0 to n (where 0 is the '
    #                      'silent gap label and n is the number of syllable'
    #                      'labels).')
    #
    if output_dir is None:
        output_dir = data_dir

    if annot_file is None:
        annot_files = dataset.annot.files_from_dir(annot_dir=data_dir,
                                                   annot_format=annot_format)
        scribe = Transcriber(voc_format=annot_format)
        annot_list = scribe.to_seq(file=annot_files)
    else:
        scribe = Transcriber(voc_format=annot_format)
        annot_list = scribe.to_seq(file=annot_file)

    # ------ if making dataset from audio files, need to make into array files first! ----------------------------
    if audio_format:
        logger.info(
            f'making array files containing spectrograms from audio files in: {data_dir}'
        )
        audio_files = dataset.audio.files_from_dir(data_dir, audio_format)
        array_files = dataset.audio.to_arr_files(audio_format=audio_format,
                                                 spect_params=spect_params,
                                                 output_dir=output_dir,
                                                 audio_files=audio_files,
                                                 annot_list=annot_list,
                                                 labelset=labelset,
                                                 skip_files_with_labels_not_in_labelset=skip_files_with_labels_not_in_labelset
                                                 )
        array_format = 'npz'
    else:
        array_files = None

    from_arr_kwargs = {
        'array_format': array_format,
        'labelset': labelset,
        'skip_files_with_labels_not_in_labelset': skip_files_with_labels_not_in_labelset,
        'load_arr': False,
        'annot_list': annot_list,
    }

    if array_files:
        from_arr_kwargs['array_files'] = array_files
        logger.info(
            f'creating VocalDataset from array files in: {output_dir}'
        )
    else:
        from_arr_kwargs['array_dir'] = data_dir
        logger.info(
            f'creating VocalDataset from array files in: {data_dir}'
        )

    vocal_dataset = dataset.array.from_arr_files(**from_arr_kwargs)

    timenow = datetime.now().strftime('%y%m%d_%H%M%S')

    # lastly rewrite config file,
    # so that paths where results were saved are automatically in config.
    config = ConfigParser()
    config.read(config_file)
    for key, saved_data_dict_path in saved_data_dict_paths.items():
        config.set(section='TRAIN',
                   option=key + '_data_path',
                   value=saved_data_dict_path)

    with open(config_file, 'w') as config_file_rewrite:
        config.write(config_file_rewrite)
