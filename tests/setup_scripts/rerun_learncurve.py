import os
from configparser import ConfigParser

import vak

HERE = os.path.dirname(__file__)
config_file = os.path.join(HERE, 'tmp_Makefile_config.ini')
if not os.path.isfile(config_file):
    raise FileNotFoundError(
        f'{config_file} not found')

config_obj = ConfigParser()
config_obj.read(config_file)
learncurve_config = vak.config.parse.parse_learncurve_config(config_obj, config_file)
nets_config = vak.config.parse._get_nets_config(config_obj, learncurve_config.networks)
spect_params = vak.config.parse.parse_spect_config(config_obj)
prep_config = vak.config.parse.parse_prep_config(config_obj, config_file)

vak.cli.learning_curve(train_vds_path=learncurve_config.train_vds_path,
                       val_vds_path=learncurve_config.val_vds_path,
                       test_vds_path=learncurve_config.test_vds_path,
                       total_train_set_duration=prep_config.total_train_set_dur,
                       train_set_durs=learncurve_config.train_set_durs,
                       num_replicates=learncurve_config.num_replicates,
                       num_epochs=learncurve_config.num_epochs,
                       config_file=config_file,
                       networks=nets_config,
                       val_error_step=learncurve_config.val_error_step,
                       checkpoint_step=learncurve_config.checkpoint_step,
                       patience=learncurve_config.patience,
                       save_only_single_checkpoint_file=learncurve_config.save_only_single_checkpoint_file,
                       normalize_spectrograms=learncurve_config.normalize_spectrograms,
                       use_train_subsets_from_previous_run=learncurve_config.use_train_subsets_from_previous_run,
                       previous_run_path=learncurve_config.previous_run_path,
                       root_results_dir=learncurve_config.root_results_dir)
