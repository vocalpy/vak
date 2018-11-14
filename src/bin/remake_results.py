import os

import songdeck

HERE = os.path.dirname(__file__)
config_file = os.path.join(HERE, 'tmp_Makefile_config.ini')
config = songdeck.config.parse.parse_config(config_file)
songdeck.cli.learncurve(train_data_dict_path=config.train.train_data_dict_path,
                        val_data_dict_path=config.train.val_data_dict_path,
                        spect_params=config.spect_params,
                        total_train_set_duration=config.data.total_train_set_dur,
                        train_set_durs=config.train.train_set_durs,
                        num_replicates=config.train.num_replicates,
                        num_epochs=config.train.num_epochs,
                        config_file=config_file,
                        networks=config.networks,
                        val_error_step=config.train.val_error_step,
                        checkpoint_step=config.train.checkpoint_step,
                        patience=config.train.patience,
                        save_only_single_checkpoint_file=config.train.save_only_single_checkpoint_file,
                        normalize_spectrograms=config.train.normalize_spectrograms,
                        use_train_subsets_from_previous_run=config.train.use_train_subsets_from_previous_run,
                        previous_run_path=config.train.previous_run_path,
                        root_results_dir=config.output.root_results_dir)
