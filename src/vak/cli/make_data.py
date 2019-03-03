import logging
import os
import sys
from datetime import datetime
from glob import glob
from configparser import ConfigParser

import joblib

from vak.utils.data import make_spects_from_list_of_files, make_data_dicts
from vak.utils.mat import convert_mat_to_spect
import vak.config


def make_data(labelset,
              all_labels_are_int,
              data_dir,
              total_train_set_dur,
              val_dur,
              test_dur,
              config_file,
              silent_gap_label=0,
              skip_files_with_labels_not_in_labelset=True,
              output_dir=None,
              mat_spect_files_path=None,
              mat_spects_annotation_file=None,
              spect_params=None,
              ):
    """make datasets for training models to generate a learning curve

    Parameters
    ----------
    labelset : list
        of str or int, set of labels for syllables
    all_labels_are_int : bool
        if True, labels are of type int, not str
    data_dir : str
        path to directory with audio files from which to make dataset
    total_train_set_dur : int
        total duration of training set, in seconds.
        Training subsets of shorter duration will be drawn from this set.
    val_dur : int
        total duration of validation set, in seconds.
    test_dur : int
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
    mat_spect_files_path : str
        Path to a directory of .mat files that contain spectrograms.
        Default is None (and this implies user is supplying audio files
         instead of supplying spectrograms in .mat files).
    mat_spects_annotation_file : str
        Path to annotation file associated with .mat files.
        Default is None.
    spect_params : dict
        Dictionary of parameters for creating spectrograms.
        Default is None (implying that spectrograms are already made).

    Returns
    -------
    None

    Saves .spect files and data_dicts in output_dir specified by user.
    """
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

    if mat_spect_files_path:
        print('will use spectrograms from .mat files in {}'
              .format(mat_spect_files_path))
        mat_spect_files = glob(os.path.join(mat_spect_files_path, '*.mat'))
        if output_dir is None:
            output_dir = os.path.join(mat_spect_files_path,
                                  'spectrograms_' + timenow)
        os.mkdir(output_dir)

        spect_files_path = convert_mat_to_spect(mat_spect_files,
                                                mat_spects_annotation_file,
                                                output_dir,
                                                labels_mapping=labels_mapping)

    else:
        logger.info('will make training data from: {}'.format(data_dir))
        if output_dir is None:
            output_dir = os.path.join(data_dir,
                                  'spectrograms_' + timenow)
        else:
            output_dir = os.path.join(output_dir,
                                  'spectrograms_' + timenow)
        os.mkdir(output_dir)

        cbins = glob(os.path.join(data_dir, '*.cbin'))
        if cbins == []:
            # if we don't find .cbins in data_dir, look in sub-directories
            cbins = []
            subdirs = glob(os.path.join(data_dir,'*/'))
            for subdir in subdirs:
                cbins.extend(glob(os.path.join(data_dir,
                                               subdir,
                                               '*.cbin')))
        if cbins == []:
            # try looking for .wav files
            wavs = glob(os.path.join(data_dir, '*.wav'))

            if cbins == [] and wavs == []:
                raise FileNotFoundError('No .cbin or .wav files found in {} or'
                                        'immediate sub-directories'
                                        .format(data_dir))
            # look for canary annotation
            annotation_file = glob(os.path.join(data_dir, '*annotation*.mat'))
            if len(annotation_file) == 1:
                annotation_file = annotation_file[0]
            else:  # try Koumura song annotation
                annotation_file = glob(os.path.join(data_dir, '../Annotation.xml'))
                if len(annotation_file) == 1:
                    annotation_file = annotation_file[0]
                else:
                    raise ValueError('Found more than one annotation.mat file: {}. '
                                     'Please include only one such file in the directory.'
                                     .format(annotation_file))

        if cbins:
            spect_files_path = \
                make_spects_from_list_of_files(cbins,
                                               spect_params,
                                               output_dir,
                                               labels_mapping,
                                               skip_files_with_labels_not_in_labelset)
        elif wavs:
            spect_files_path = \
                make_spects_from_list_of_files(wavs,
                                               spect_params,
                                               output_dir,
                                               labels_mapping,
                                               skip_files_with_labels_not_in_labelset,
                                               annotation_file)

    saved_data_dict_paths = make_data_dicts(output_dir,
                                            total_train_set_dur,
                                            val_dur,
                                            test_dur,
                                            labelset,
                                            spect_files_path)

    # lastly rewrite config file,
    # so that paths where results were saved are automatically in config.
    # also need to add number of frequency bins in spectrogram, so
    # networks can access that programmatically instead of user needing to
    # declare it.
    config = ConfigParser()
    config.read(config_file)
    for key, saved_data_dict_path in saved_data_dict_paths.items():
        config.set(section='TRAIN',
                   option=key + '_data_path',
                   value=saved_data_dict_path)
    train_data = joblib.load(saved_data_dict_paths['train'])
    freq_bins = train_data['X_train'].shape[0]
    config.set(section='DATA',
               option='freq_bins',
               value=str(freq_bins))

    with open(config_file, 'w') as config_file_rewrite:
        config.write(config_file_rewrite)


if __name__ == "__main__":
    config_file = os.path.normpath(sys.argv[1])
    config = vak.config.parse_config(config_file)
    make_data(labelset=config.data.labelset,
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
