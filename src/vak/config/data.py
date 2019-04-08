"""parses [DATA] section of config"""
import os

import attr
from attr.validators import instance_of, optional

from ..utils.data import range_str
from .validators import is_a_directory, is_a_file


@attr.s
class DataConfig:
    """class to represent [DATA] section of config.ini file

    Attributes
    ----------
    labelset : list
        of str or int, set of labels for syllables
    all_labels_are_int : bool
        if True, labels are of type int, not str
    silent_gap_label  : int
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
    data_dir : str
        path to directory with audio files from which to make dataset
    total_train_set_dur : float
        total duration of training set, in seconds.
        Training subsets of shorter duration will be drawn from this set.
    val_dur : float
        total duration of validation set, in seconds.
    test_dur : float
        total duration of test set, in seconds.
    save_transformed_data : bool
        if True, save transformed data (i.e. scaled, reshaped). The data can then
        be used on a subsequent run (e.g. if you want to compare results
        from different hyperparameters across the exact same training set).
        Also useful if you need to check what the data looks like when fed to networks.
        Default is False.
    """
    labelset = attr.ib(validator=instance_of(list))

    total_train_set_dur = attr.ib(validator=optional(instance_of(float)), default=None)
    val_dur = attr.ib(validator=optional(instance_of(float)), default=None)
    test_dur = attr.ib(validator=optional(instance_of(float)), default=None)
    all_labels_are_int = attr.ib(validator=instance_of(bool), default=False)
    silent_gap_label = attr.ib(validator=instance_of(int), default=0)
    skip_files_with_labels_not_in_labelset = attr.ib(validator=instance_of(bool), default=True)
    output_dir = attr.ib(validator=optional(is_a_directory), default=None)
    mat_spect_files_path = attr.ib(validator=optional(is_a_directory), default=None)
    mat_spects_annotation_file = attr.ib(validator=optional(is_a_file), default=None)
    data_dir = attr.ib(validator=optional(is_a_directory), default=None)
    save_transformed_data = attr.ib(validator=instance_of(bool), default=False)


def parse_data_config(config, config_file):
    """parse [DATA] section of config.ini file

    Parameters
    ----------
    config : ConfigParser
        containing config.ini file already loaded by parse function
    config_file : str
        path to config file (used for error messages)

    Returns
    -------
    data_config : vak.config.data.DataConfig
        instance of class that represents [DATA] section of config.ini file
    """
    config_dict = {}

    labelset = config['DATA']['labelset']
    # make mapping from syllable labels to consecutive integers
    # start at 1, because 0 is assumed to be label for silent gaps
    if '-' in labelset or ',' in labelset:
        # if user specified range of ints using a str
        config_dict['labelset'] = range_str(labelset)
    else:  # assume labelset is characters
        config_dict['labelset'] = list(labelset)

    # to make type-checking consistent across .mat / .cbin / Koumura .wav files
    # set all_labels_are_int flag
    # currently only used with .mat files
    if config.has_option('DATA', 'all_labels_are_int'):
        config_dict['all_labels_are_int'] = config.getboolean('DATA', 'all_labels_are_int')

    if config.has_option('DATA', 'silent_gap_label'):
        config_dict['silent_gap_label'] = int(config['DATA']['silent_gap_label'])

    if config.has_option('DATA', 'skip_files_with_labels_not_in_labelset'):
        config_dict[
            'skip_files_with_labels_not_in_labelset'
        ] = config.getboolean('DATA',
                              'skip_files_with_labels_not_in_labelset')

    if config.has_option('DATA', 'output_dir'):
        output_dir = config['DATA']['output_dir']
        output_dir = os.path.expanduser(output_dir)
        config_dict['output_dir'] = os.path.abspath(output_dir)

    # if using spectrograms from .mat files
    if config.has_option('DATA', 'mat_spect_files_path'):
        # make spect_files file from .mat spect files and annotation file
        config_dict['mat_spect_files_path'] = config['DATA']['mat_spect_files_path']
        config_dict['mat_spects_annotation_file'] = config['DATA']['mat_spect_files_annotation_file']

    if config.has_option('DATA', 'data_dir'):
        data_dir = config['DATA']['data_dir']
        config_dict['data_dir'] = os.path.expanduser(data_dir)

    if config.has_option('DATA', 'total_train_set_duration'):
        config_dict['total_train_set_dur'] = float(config['DATA']['total_train_set_duration'])

    if config.has_option('DATA', 'validation_set_duration'):
        config_dict['val_dur'] = float(config['DATA']['validation_set_duration'])

    if config.has_option('DATA', 'test_set_duration'):
        config_dict['test_dur'] = float(config['DATA']['test_set_duration'])

    if config.has_option('DATA', 'save_transformed_data'):
        config_dict['save_transformed_data'] = config.getboolean('DATA', 'save_transformed_data')

    return DataConfig(**config_dict)
