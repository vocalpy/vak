import os

import songdeck

HERE = os.path.dirname(__file__)
config_file = os.path.join(HERE, 'tmp_Makefile_config.ini')
config = songdeck.config.parse.parse_config(config_file)
songdeck.cli.make_data(labelset=config.data.labelset,
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
