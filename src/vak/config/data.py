"""parses [DATA] section of config"""
import os
from collections import namedtuple

from vak.utils.data import range_str

fields = ['labelset',
          'all_labels_are_int',
          'silent_gap_label',
          'skip_files_with_labels_not_in_labelset',
          'output_dir',
          'mat_spect_files_path',
          'mat_spects_annotation_file',
          'data_dir',
          'total_train_set_dur',
          'val_dur',
          'test_dur',
          'freq_bins']
DataConfig = namedtuple('DataConfig', fields)


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
    data_config : DataConfig
        namedtuple with fields:
            labelset
            all_labels_are_int
            silent_gap_label
            skip_files_with_labels_not_in_labelset
            output_dir
            mat_spect_files_path
            mat_spects_annotation_file
            data_dir
            total_train_set_dur
            val_dur
            test_dur
            freq_bins
    """
    labelset = config['DATA']['labelset']
    # make mapping from syllable labels to consecutive integers
    # start at 1, because 0 is assumed to be label for silent gaps
    if '-' in labelset or ',' in labelset:
        # if user specified range of ints using a str
        labelset = range_str(labelset)
    else:  # assume labelset is characters
        labelset = list(labelset)

    # to make type-checking consistent across .mat / .cbin / Koumura .wav files
    # set all_labels_are_int flag
    # currently only used with .mat files
    if config.has_option('DATA', 'all_labels_are_int'):
        all_labels_are_int = config.getboolean('DATA', 'all_labels_are_int')
    else:
        all_labels_are_int = False

    if config.has_option('DATA', 'silent_gap_label'):
        silent_gap_label = int(config['DATA']['silent_gap_label'])
    else:
        silent_gap_label = 0

    if config.has_option('DATA', 'skip_files_with_labels_not_in_labelset'):
        skip_files_with_labels_not_in_labelset = config.getboolean(
            'DATA',
            'skip_files_with_labels_not_in_labelset')
    else:
        skip_files_with_labels_not_in_labelset = True

    if config.has_option('DATA', 'output_dir'):
        output_dir = config['DATA']['output_dir']
        output_dir = os.path.expanduser(output_dir)
        output_dir = os.path.abspath(output_dir)
        if not os.path.isdir(output_dir):
            raise NotADirectoryError('{} specified as output_dir in {}, '
                                     'but not recognized as a directory'
                                     .format(output_dir, config_file))
    else:
        output_dir = None

    # if using spectrograms from .mat files
    if config.has_option('DATA', 'mat_spect_files_path'):
        # make spect_files file from .mat spect files and annotation file
        mat_spect_files_path = config['DATA']['mat_spect_files_path']
        mat_spects_annotation_file = config['DATA']['mat_spect_files_annotation_file']
    else:
        mat_spect_files_path = None
        mat_spects_annotation_file = None

    data_dir = config['DATA']['data_dir']
    data_dir = os.path.expanduser(data_dir)
    if not os.path.isdir(data_dir):
        raise NotADirectoryError('{} specified as data_dir in {}, '
                                 'but not recognized as a directory'
                                 .format(data_dir, config_file))

    if config.has_option('DATA', 'total_train_set_duration'):
        total_train_set_dur = float(config['DATA']['total_train_set_duration'])
    else:
        total_train_set_dur = None

    if config.has_option('DATA', 'validation_set_duration'):
        val_dur = float(config['DATA']['validation_set_duration'])
    else:
        val_dur = None

    if config.has_option('DATA', 'test_set_duration'):
        test_dur = float(config['DATA']['test_set_duration'])
    else:
        test_dur = None

    if config.has_option('DATA', 'freq_bins'):
        freq_bins = int(config['DATA']['freq_bins'])
    else:
        freq_bins = None

    return DataConfig(labelset,
                      all_labels_are_int,
                      silent_gap_label,
                      skip_files_with_labels_not_in_labelset,
                      output_dir,
                      mat_spect_files_path,
                      mat_spects_annotation_file,
                      data_dir,
                      total_train_set_dur,
                      val_dur,
                      test_dur,
                      freq_bins)
