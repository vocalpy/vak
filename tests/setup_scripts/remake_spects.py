import os
from configparser import ConfigParser

import vak

HERE = os.path.dirname(__file__)
config_file = os.path.join(HERE, 'tmp_Makefile_config.ini')
config = ConfigParser()
config.read(config_file)
data_config = vak.config.parse.parse_data_config(config, config_file)
spect_params = vak.config.parse.parse_spect_config(config)
vak.cli.make_data(labelset=data_config.labelset,
                  all_labels_are_int=data_config.all_labels_are_int,
                  data_dir=data_config.data_dir,
                  total_train_set_dur=data_config.total_train_set_dur,
                  val_dur=data_config.val_dur,
                  test_dur=data_config.test_dur,
                  config_file=config_file,
                  silent_gap_label=data_config.silent_gap_label,
                  skip_files_with_labels_not_in_labelset=data_config.skip_files_with_labels_not_in_labelset,
                  output_dir=data_config.output_dir,
                  mat_spect_files_path=data_config.mat_spect_files_path,
                  mat_spects_annotation_file=data_config.mat_spects_annotation_file,
                  spect_params=spect_params)
