import logging
import os
import sys
from datetime import datetime
from glob import glob

from songdeck.utils.data import make_spects_from_list_of_files, make_data_dicts

from songdeck.utils.mat import convert_mat_to_spect


def make_data(labelset,
              all_labels_are_int,
              data_dir,
              train_set_dur,
              val_dur,
              test_dur,
              silent_gap_label=0,
              skip_files_with_labels_not_in_labelset=True,
              output_dir=None,
              mat_spect_files_path=None,
              spect_params=None,
              ):
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

        logger.info('will make training data from: {}'.format(data_dir))
        if output_dir is None:
            output_dir = os.path.join(data_dir,
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
                                            train_set_dur,
                                            val_dur,
                                            test_dur,
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
    make_data(config_file)
