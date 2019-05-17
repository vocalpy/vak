import os
from configparser import ConfigParser

import vak

HERE = os.path.dirname(__file__)
config_file = os.path.join(HERE, 'tmp_Makefile_config.ini')
config = ConfigParser()
config.read(config_file)
data_config = vak.config.parse.parse_data_config(config, config_file)
spect_params = vak.config.parse.parse_spect_config(config)
vak.cli.prep(labelset=data_config.labelset,
             data_dir=data_config.data_dir,
             total_train_set_dur=data_config.total_train_set_dur,
             test_dur=data_config.test_dur,
             config_file=config_file,
             annot_format=data_config.annot_format,
             val_dur=data_config.val_dur,
             skip_files_with_labels_not_in_labelset=data_config.skip_files_with_labels_not_in_labelset,
             output_dir=data_config.output_dir,
             audio_format=data_config.audio_format,
             spect_format=data_config.spect_format,
             annot_file=data_config.annot_file,
             spect_params=spect_params)
