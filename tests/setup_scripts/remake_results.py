import os
from configparser import ConfigParser

import vak

HERE = os.path.dirname(__file__)
config_file = os.path.join(HERE, 'tmp_Makefile_config.ini')
config_obj = ConfigParser()
config_obj.read(config_file)
train_config = vak.config.parse.parse_train_config(config_obj, config_file)
nets_config = vak.config.parse._get_nets_config(config_obj, train_config.networks)
spect_params = vak.config.parse.parse_spect_config(config_obj)
data_config = vak.config.parse.parse_data_config(config_obj, config_file)
output_config = vak.config.parse.parse_output_config(config_obj)

vak.cli.learncurve(train_data_dict_path=train_config.train_vds_path,
                   val_data_dict_path=train_config.val_vds_path,
                   spect_params=spect_params,
                   total_train_set_duration=data_config.total_train_set_dur,
                   train_set_durs=train_config.train_set_durs,
                   num_replicates=train_config.num_replicates,
                   num_epochs=train_config.num_epochs,
                   config_file=config_file,
                   networks=nets_config,
                   val_error_step=train_config.val_error_step,
                   checkpoint_step=train_config.checkpoint_step,
                   patience=train_config.patience,
                   save_only_single_checkpoint_file=train_config.save_only_single_checkpoint_file,
                   normalize_spectrograms=train_config.normalize_spectrograms,
                   use_train_subsets_from_previous_run=train_config.use_train_subsets_from_previous_run,
                   previous_run_path=train_config.previous_run_path,
                   root_results_dir=output_config.root_results_dir)
