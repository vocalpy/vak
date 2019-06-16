"""parses [DATA] section of config"""
import os

import attr
from attr.validators import instance_of, optional

from ..utils.data import range_str
from .validators import is_a_directory, is_a_file, is_audio_format, is_annot_format, is_spect_format


@attr.s
class DataConfig:
    """class to represent [DATA] section of config.ini file

    Attributes
    ----------
    labelset : list
        of str or int, set of labels for syllables
    total_train_set_dur : float
        total duration of training set, in seconds.
        Training subsets of shorter duration will be drawn from this set.
    val_dur : float
        total duration of validation set, in seconds.
    test_dur : float
        total duration of test set, in seconds.
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
    data_dir : str
        path to directory with audio files from which to make dataset
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

    output_dir = attr.ib(validator=optional(is_a_directory), default=None)

    audio_format = attr.ib(validator=optional(is_audio_format), default=None)
    spect_format = attr.ib(validator=optional(is_spect_format), default=None)
    annot_file = attr.ib(validator=optional(is_a_file), default=None)
    annot_format = attr.ib(validator=optional(is_annot_format), default=None)
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
    if config.has_option('DATA', 'spect_format') and config.has_option('DATA', 'audio_format'):
        raise ValueError("[DATA] section of config.ini file cannot specify both audio_format and "
                         "spect_format, unclear whether to create spectrograms from audio files or "
                         "use already-generated spectrograms")

    if not(config.has_option('DATA', 'spect_format')) and not(config.has_option('DATA', 'audio_format')):
        raise ValueError("[DATA] section of config.ini file must specify either audio_format or "
                         "spect_format")

    config_dict = {}

    labelset = config['DATA']['labelset']
    # make mapping from syllable labels to consecutive integers
    # start at 1, because 0 is assumed to be label for silent gaps
    if '-' in labelset or ',' in labelset:
        # if user specified range of ints using a str
        config_dict['labelset'] = range_str(labelset)
    else:  # assume labelset is characters
        config_dict['labelset'] = list(labelset)

    if config.has_option('DATA', 'total_train_set_duration'):
        config_dict['total_train_set_dur'] = float(config['DATA']['total_train_set_duration'])

    if config.has_option('DATA', 'validation_set_duration'):
        config_dict['val_dur'] = float(config['DATA']['validation_set_duration'])

    if config.has_option('DATA', 'test_set_duration'):
        config_dict['test_dur'] = float(config['DATA']['test_set_duration'])

    if config.has_option('DATA', 'output_dir'):
        output_dir = config['DATA']['output_dir']
        output_dir = os.path.expanduser(output_dir)
        config_dict['output_dir'] = os.path.abspath(output_dir)

    if config.has_option('DATA', 'audio_format'):
        config_dict['audio_format'] = config['DATA']['audio_format']

    if config.has_option('DATA', 'spect_format'):
        config_dict['spect_format'] = config['DATA']['spect_format']

    if config.has_option('DATA', 'annot_format'):
        config_dict['annot_format'] = config['DATA']['annot_format']

    if config.has_option('DATA', 'annot_file'):
        config_dict['annot_file'] = config['DATA']['annot_file']

    if config.has_option('DATA', 'data_dir'):
        data_dir = config['DATA']['data_dir']
        config_dict['data_dir'] = os.path.expanduser(data_dir)

    if config.has_option('DATA', 'save_transformed_data'):
        config_dict['save_transformed_data'] = config.getboolean('DATA', 'save_transformed_data')

    return DataConfig(**config_dict)
