import logging
import os
import sys
from datetime import datetime
from glob import glob
from configparser import ConfigParser

from ..utils.data import make_data_dicts
from ..utils.spect import from_list
from ..utils.mat import convert_mat_to_spect

from ..dataset.audio import _get_audio_files


def prep(labelset,
         all_labels_are_int,
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
         spect_format=None,
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
    spect_format : str
        format of files containg spectrograms as 2-d matrices.
        One of {'mat', 'npy'}.
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
    if audio_format is None and spect_format is None:
        raise ValueError("Must specify either audio_format or spect_format")

    if audio_format and spect_format:
        raise ValueError("Cannot specify both audio_format and spect_format, "
                         "unclear whether to create spectrograms from audio files or "
                         "use already-generated spectrograms")

    logger = logging.getLogger(__name__)
    logger.setLevel('INFO')

    # map labels to series of consecutive integers from 0 to n inclusive
    # where 0 is the label for silent periods between syllables
    # and n is the number of syllable labels
    if all_labels_are_int:
        labels_mapping = dict(zip([int(label) for label in labelset],
                                  range(1, len(labelset) + 1)))
    else:
        labels_mapping = dict(zip(labelset,
                                  range(1, len(labelset) + 1)))
    labels_mapping['silent_gap_label'] = silent_gap_label
    if sorted(labels_mapping.values()) != list(range(len(labels_mapping))):
        raise ValueError('Labels mapping does not map to a consecutive'
                         'series of integers from 0 to n (where 0 is the '
                         'silent gap label and n is the number of syllable'
                         'labels).')

    timenow = datetime.now().strftime('%y%m%d_%H%M%S')

    if output_dir is None:
        output_dir = os.path.join(data_dir,
                                  'spectrograms_' + timenow)
    else:
        output_dir = os.path.join(output_dir,
                                  'spectrograms_' + timenow)
    os.mkdir(output_dir)

    if spect_format:
        logger.info(f'loading spectrograms from: {data_dir}')
        if spect_format == 'mat':
            spect_files = glob(os.path.join(data_dir, '*.mat'))
        elif spect_format == 'npy':
            spect_files = glob(os.path.join(data_dir, '*.npy'))
        spect_files_path = convert_mat_to_spect(spect_files,
                                                annot_file,
                                                output_dir,
                                                labels_mapping=labels_mapping)

    elif audio_format:
        logger.info(f'making spectrograms from audio files in: {data_dir}')
        audio_files = _get_audio_files(audio_format, data_dir)
        spect_files_path = \
            from_list(audio_files,
                      spect_params,
                      output_dir,
                      labels_mapping,
                      skip_files_with_labels_not_in_labelset,
                      annot_file)

    saved_data_dict_paths = make_data_dicts(output_dir,
                                            total_train_set_dur,
                                            val_dur,
                                            test_dur,
                                            labelset,
                                            spect_files_path)

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


if __name__ == "__main__":
    config_file = os.path.normpath(sys.argv[1])
    config = config.parse_config(config_file)
    prep(labelset=config.data.labelset,
         all_labels_are_int=config.data.all_labels_are_int,
         data_dir=config.data.data_dir,
         total_train_set_dur=config.data.total_train_set_dur,
         val_dur=config.data.val_dur,
         test_dur=config.data.test_dur,
         config_file=config_file,
         silent_gap_label=config.data.silent_gap_label,
         skip_files_with_labels_not_in_labelset=config.data.skip_files_with_labels_not_in_labelset,
         output_dir=config.data.output_dir,
         mat_spect_files_path=config.data.mat_spect_files_path,
         mat_spects_annotation_file=config.data.mat_spects_annotation_file,
         spect_params=config.spect_params)
