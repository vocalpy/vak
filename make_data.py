import os
import sys
from glob import glob
from datetime import datetime
import logging
from configparser import ConfigParser

from cnn_bilstm.utils import make_spects_from_list_of_files, make_data_dicts


def range_str(range_str, sort=True):
    """Generate range of ints from string

    Example:
        >>> range_str('1-4,6,9-11')
        [1,2,3,4,6,9,10,11]

    Takes a range in form of "a-b" and returns
    a list of numbers between a and b inclusive.
    Also accepts comma separated ranges like "a-b,c-d,f"  which will
    return a list with numbers from a to b, c to d, and f.

    Parameters
    ----------
    range_str : str
        of form 'a-b,c'
        where a hyphen indicates a range
        and a comma separates ranges or single numbers
    sort : bool
        If True, sort output before returning. Default is True.

    Returns
    -------
    list_range : list
        of int, produced by parsing range_str
    """
    # adapted from
    # http://code.activestate.com/recipes/577279-generate-list-of-numbers-from-hyphenated-and-comma/
    s = "".join(range_str.split())  # removes white space
    list_range = []
    for substr in range_str.split(','):
        subrange = substr.split('-')
        if len(subrange) not in [1, 2]:
            raise SyntaxError("unable to parse range {} in labelset {}."
                              .format(subrange, substr))
        list_range.extend(
            [int(subrange[0])]
        ) if len(subrange) == 1 else list_range.extend(
            range(int(subrange[0]), int(subrange[1]) + 1))

    if sort:
        list_range.sort()

    return list_range


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

    labelset = config['DATA']['labelset']
    # make mapping from syllable labels to consecutive integers
    # start at 1, because 0 is assumed to be label for silent gaps
    if '-' in labelset or ',' in labelset:
        # if user specified range of ints using a str
        labelset = range_str(labelset)
        # since labels are ints, don't actually change them.
        labels_mapping = dict(zip(labelset, labelset))
        labels_mapping['labels_are_ints'] = 'Yes'
    else:  # assume labelset is characters
        labelset = list(labelset)
        labels_mapping = dict(zip(labelset,
                                  range(1, len(labelset) + 1)))
        labels_mapping['labels_are_ints'] = 'No'

    if config.has_option('DATA', 'silent_gap_label'):
        silent_gap_label = int(config['DATA']['silent_gap_label'])
        labels_mapping['silent_gap_label'] = silent_gap_label
    else:
        if 0 not in labels_mapping.values():
            # default to 0 as silent gap label
            labels_mapping['silent_gap_label'] = 0
        else:
            # failing that, just use max int value plus one
            labels_mapping['silent_gap_label'] = \
                max(labels_mapping.values()) + 1

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
        wavs = glob(os.path.join(data_dir, '*.wav'))
        annotation_file = glob(os.path.join(data_dir, '*annotation*.mat'))
        if len(annotation_file) == 1:
            annotation_file = annotation_file[0]
        else:
            raise ValueError('Found more than one annotation.mat file: {}. '
                             'Please include only one such file in the directory.'
                             .format(annotation_file))

    if cbins == [] and wavs == []:
        raise FileNotFoundError('No .cbin or .wav files found in {} or'
                                'immediate sub-directories'
                                .format(data_dir))

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
