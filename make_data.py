import os
import sys
from glob import glob
from datetime import datetime
import logging
from configparser import ConfigParser

from cnn_bilstm.utils import make_spects_from_list_of_cbins, make_data_dicts

if __name__ == "__main__":
    config_file = os.path.normpath(sys.argv[1])
    if not config_file.endswith('.ini'):
        raise ValueError('{} is not a valid config file, '
                         'must have .ini extension'.format(config_file))
    if not os.path.isfile(config_file):
        raise FileNotFoundError('config file {} is not found'
                                .format(config_file))
    config = ConfigParser()
    config.read(config_file)

    logger = logging.getLogger(__name__)
    logger.setLevel('INFO')

    # require user to specify parameters for spectrogram
    # instead of having defaults (as was here previously)
    # helps ensure we don't mix up different params
    spect_params = {}
    spect_params['fft_size'] = int(config['SPECTROGRAM']['fft_size'])
    spect_params['step_size'] = int(config['SPECTROGRAM']['step_size'])
    spect_params['freq_cutoffs'] = [float(element)
                                    for element in
                                    config['SPECTROGRAM']['freq_cutoffs']
                                        .split(',')]
    spect_params['thresh'] = float(config['SPECTROGRAM']['thresh'])
    spect_params['log_transform'] = config.getboolean('SPECTROGRAM',
                                                      'log_transform')

    data_dir = config['DATA']['data_dir']
    logger.info('will make training data from: {}'.format(data_dir))
    timenow = datetime.now().strftime('%y%m%d_%H%M%S')
    if config.has_option('DATA','output_dir'):
        output_dir = os.path.join(config['DATA']['output_dir'],
                                  'spectrograms_' + timenow)
    else:
        output_dir = os.path.join(data_dir,
                                  'spectrograms_' + timenow)
    os.mkdir(output_dir)

    labelset = list(config['DATA']['labelset'])
    # make mapping from syllable labels to consecutive integers
    # start at 1, because 0 is assumed to be label for silent gaps
    labels_mapping = dict(zip(labelset,
                              range(1, len(labelset) + 1)))
    skip_files_with_labels_not_in_labelset = config.getboolean(
        'DATA',
        'skip_files_with_labels_not_in_labelset')

    if not os.path.isdir(data_dir):
        raise NotADirectoryError('{} not recognized '
                                 'as a directory'.format(data_dir))

    cbins = glob(os.path.join(data_dir, '*.cbin'))
    if len(cbins) == 0:
        # if we don't find .cbins in data_dir, look in sub-directories
        cbins = []
        subdirs = glob(os.path.join(data_dir,'*/'))
        for subdir in subdirs:
            cbins.extend(glob(os.path.join(data_dir,
                                           subdir,
                                           '*.cbin')))
    if len(cbins) == 0:
        raise FileNotFoundError('No .cbin files found in {} or'
                                'immediate sub-directories'
                                .format(data_dir))

    spect_files_path = \
        make_spects_from_list_of_cbins(cbins,
                                       spect_params,
                                       output_dir,
                                       labels_mapping,
                                       skip_files_with_labels_not_in_labelset)

    make_data_dicts(output_dir,
                    float(config['DATA']['total_train_set_duration']),
                    float(config['DATA']['validation_set_duration']),
                    float(config['DATA']['test_set_duration']),
                    labelset,
                    spect_files_path)

    spects = []
    labels = []
    all_time_bins = []
    labeled_timebins = []
    all_time_bins.append(time_bins)
    spects.append(spect)
    labels.append(this_labels)
    labeled_timebins.append(this_labeled_timebins)
