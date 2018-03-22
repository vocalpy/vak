import os
import sys
from glob import glob
from datetime import datetime
import logging
from configparser import ConfigParser

from cnn_bilstm.utils import make_spects_from_list_of_files, make_data_dicts, range_str


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
    if config.has_option('SPECTROGRAM', 'thresh'):
        spect_params['thresh'] = float(config['SPECTROGRAM']['thresh'])
    if config.has_option('SPECTROGRAM', 'transform_type'):
        spect_params['transform_type'] = config['SPECTROGRAM']['transform_type']
        valid_transform_types = {'log_spect', 'log_spect_plus_one'}
        if spect_params['transform_type'] not in valid_transform_types:
            raise ValueError('Value for `transform_type`, {}, in [SPECTROGRAM] '
                             'section of .ini file is not recognized. Must be one '
                             'of the following: {}'
                             .format(spect_params['transform_type'],
                                     valid_transform_types))

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

    labelset = config['DATA']['labelset']
    labels_mapping = {}
    # make mapping from syllable labels to consecutive integers
    # start at 1, because 0 is assumed to be label for silent gaps
    if '-' in labelset or ',' in labelset:
        # if user specified range of ints using a str
        labelset = range_str(labelset)
    else:  # assume labelset is characters
        labelset = list(labelset)

    # map to series of consecutive integers from 0 to n inclusive
    # where 0 is the label for silent periods between syllables
    # and n is the number of syllable labels
    labels_mapping = dict(zip(labelset,
                              range(1, len(labelset) + 1)))
    labels_mapping['silent_gap_label'] = 0

    if sorted(labels_mapping.values()) != list(range(len(labels_mapping))):
        raise ValueError('Labels mapping does not map to a consecutive'
                         'series of integers from 0 to n (where 0 is the '
                         'silent gap label and n is the number of syllable'
                         'labels).')

    skip_files_with_labels_not_in_labelset = config.getboolean(
        'DATA',
        'skip_files_with_labels_not_in_labelset')

    if not os.path.isdir(data_dir):
        raise NotADirectoryError('{} not recognized '
                                 'as a directory'.format(data_dir))

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


    make_data_dicts(output_dir,
                    float(config['DATA']['total_train_set_duration']),
                    float(config['DATA']['validation_set_duration']),
                    float(config['DATA']['test_set_duration']),
                    labelset,
                    spect_files_path)
