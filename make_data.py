import sys
from datetime import datetime
import logging
from configparser import ConfigParser

if __name__ == "__main__":
    config_file = sys.argv[1]
    if not config_file.endswith('.ini'):
        raise ValueError('{} is not a valid config file, '
                         'must have .ini extension'.format(config_file))
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

    # given directory, makes a new directory of .spect files from each .cbin file
    # also go into sub-directories
    data_dir = config['DATA']['data_dir']
    logger.info('will make training data from: {}'.format(data_dir))
    timenow = datetime.now().strftime('%y%m%d_%H%M%S')
    if config.has_option('DATA','output_dir'):
        data_dirname = os.path.join(output_dir,
                                       'cnn_bilstm_data_' + timenow)
    else:
        data_dirname = os.path.join(data_dir,
                                       'cnn_bilstm_data_' + timenow)
    os.mkdir(data_dirname)

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



        spects = []
        labels = []
        all_time_bins = []
        labeled_timebins = []
        all_time_bins.append(time_bins)
        spects.append(spect)
        labels.append(this_labels)
        labeled_timebins.append(this_labeled_timebins)




    (train_data_dict,
     train_data_dict_path) = make_data_dict(labels_mapping,
                                            train_data_dir,
                                            number_song_files,
                                            spect_params,
                                            skip_files_with_labels_not_in_labelset)

    total_data_set_duration =
    assert total_dur_from_spects == total_data_set_duration
