import logging
import os
import sys
from configparser import ConfigParser
from datetime import datetime
from glob import glob

from ..utils import make_spects_from_list_of_files, make_data_dicts, \
    range_str

from ..utils.mat import convert_mat_to_spect


def dataset(config_file):
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
    if config.has_option('DATA','all_labels_are_int'):
        all_labels_are_int = config.getboolean('DATA','all_labels_are_int')
    else:
        all_labels_are_int = False

    # map labels to series of consecutive integers from 0 to n inclusive
    # where 0 is the label for silent periods between syllables
    # and n is the number of syllable labels
    if all_labels_are_int:
        labels_mapping = dict(zip([int(label) for label in labelset],
                                  range(1, len(labelset) + 1)))
    else:
        labels_mapping = dict(zip(labelset,
                                  range(1, len(labelset) + 1)))
    if not config.has_option('DATA', 'silent_gap_label'):
        labels_mapping['silent_gap_label'] = 0
    else:
        labels_mapping['silent_gap_label'] = int(
            config['DATA']['silent_gap_label'])
    if sorted(labels_mapping.values()) != list(range(len(labels_mapping))):
        raise ValueError('Labels mapping does not map to a consecutive'
                         'series of integers from 0 to n (where 0 is the '
                         'silent gap label and n is the number of syllable'
                         'labels).')

    skip_files_with_labels_not_in_labelset = config.getboolean(
        'DATA',
        'skip_files_with_labels_not_in_labelset')

    timenow = datetime.now().strftime('%y%m%d_%H%M%S')
    if config.has_option('DATA', 'output_dir'):
        output_dir = os.path.join(config['DATA']['output_dir'],
                                  'spectrograms_' + timenow)
    else:
        output_dir = None

    ### if using spectrograms from .mat files ###
    if config.has_option('DATA','mat_spect_files_path'):
        # make spect_files file from .mat spect files and annotation file
        mat_spect_files_path = config['DATA']['mat_spect_files_path']
        print('will use spectrograms from .mat files in {}'
              .format(mat_spect_files_path))
        mat_spect_files = glob(os.path.join(mat_spect_files_path,'*.mat'))
        mat_spects_annotation_file = config['DATA']['mat_spect_files_annotation_file']
        if output_dir is None:
            output_dir = os.path.join(mat_spect_files_path,
                                  'spectrograms_' + timenow)
        os.mkdir(output_dir)

        spect_files_path = convert_mat_to_spect(mat_spect_files,
                                                mat_spects_annotation_file,
                                                output_dir,
                                                labels_mapping=labels_mapping)
    else:
        mat_spect_files_path = None

    ### if **not** using spectrograms from .mat files ###
    if mat_spect_files_path is None:
        if not config.has_section('SPECTROGRAM'):
            raise ValueError('No annotation_path specified in config_file that '
                             'would point to annotated spectrograms, but no '
                             'parameters provided to generate spectrograms '
                             'either.')
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
        if not os.path.isdir(data_dir):
            raise NotADirectoryError('{} not recognized '
                                     'as a directory'.format(data_dir))
        logger.info('will make training data from: {}'.format(data_dir))
        if output_dir is None:
            output_dir = os.path.join(data_dir,
                                  'spectrograms_' + timenow)
        os.mkdir(output_dir)

        cbins = glob(os.path.join(data_dir, '*.cbin'))
        if cbins == []:
            # if we don't find .cbins in data_dir, look in sub-directories
            cbins = []
            subdirs = glob(os.path.join(data_dir, '*/'))
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
                float(config['DATA']['total_train_set_duration']),
                float(config['DATA']['validation_set_duration']),
                float(config['DATA']['test_set_duration']),
                labelset,
                spect_files_path)

    # lastly rewrite config file,
    # so that paths where results were saved are automatically in config
    for key, saved_data_dict_path in saved_data_dict_paths.items():
        config.set(section='TRAIN',
                   option=key + '_data_path',
                   value=saved_data_dict_path)
    with open(config_file, 'w') as config_file_rewrite:
        config.write(config_file_rewrite)

if __name__ == "__main__":
    config_file = os.path.normpath(sys.argv[1])
    dataset(config_file)
