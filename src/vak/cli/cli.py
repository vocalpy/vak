from .train import train
from .learncurve import learning_curve
from .predict import predict
from .prep import prep


def cli(command, config_file):
    """command-line interface

    Parameters
    ----------
    command : string
        One of {'prep', 'train', 'predict', 'finetune', 'learncurve'}
    config_file : str, Path
        path to a config.toml file
    """
    if command == 'prep':
        prep(toml_path=config_file)

    elif command == 'train':
        train(toml_path=config_file)

    elif command == 'finetune':
        raise NotImplementedError

    elif command == 'predict':
        predict(toml_path=config_file)

    # elif command == 'learncurve':
    #     if config_obj.has_option('TRAIN', 'results_dir_made_by_main_script'):
    #         raise ValueError(
    #             f"config file {config_file} already has option 'results_dir_made_by_main_script' "
    #             "in [TRAIN] section. \nRunning learncurve will overwrite that option."
    #             "Please either remove the option from this file or make a copy of the config.ini "
    #             "file with a new name and remove it from that file.\n"
    #             f"Currently the option is set to: {config_obj['TRAIN']['results_dir_made_by_main_script']}"
    #         )
    #     learncurve_config = parse_learncurve_config(config_obj, config_file)
    #     models = config.models.from_config(config_obj, learncurve_config.models)
    #     prep_config = parse_prep_config(config_obj, config_file)
    #     if learncurve_config.train_vds_path is None:
    #         raise ValueError("must set 'train_vds_path' option in [LEARNCURVE] section of config.ini file "
    #                          "before running 'learncurve'")
    #     if learncurve_config.val_vds_path is None:
    #         raise ValueError("must set 'val_vds_path' option in [LEARNCURVE] section of config.ini file "
    #                          "before running 'learncurve'")
    #     learning_curve(train_vds_path=learncurve_config.train_vds_path,
    #                    val_vds_path=learncurve_config.val_vds_path,
    #                    test_vds_path=learncurve_config.test_vds_path,
    #                    train_dur=prep_config.train_dur,
    #                    train_set_durs=learncurve_config.train_set_durs,
    #                    num_replicates=learncurve_config.num_replicates,
    #                    num_epochs=learncurve_config.num_epochs,
    #                    config_file=config_file,
    #                    networks=models,
    #                    val_error_step=learncurve_config.val_error_step,
    #                    checkpoint_step=learncurve_config.checkpoint_step,
    #                    patience=learncurve_config.patience,
    #                    save_only_single_checkpoint_file=learncurve_config.save_only_single_checkpoint_file,
    #                    normalize_spectrograms=learncurve_config.normalize_spectrograms,
    #                    use_train_subsets_from_previous_run=learncurve_config.use_train_subsets_from_previous_run,
    #                    previous_run_path=learncurve_config.previous_run_path,
    #                    root_results_dir=learncurve_config.root_results_dir,
    #                    save_transformed_data=learncurve_config.save_transformed_data)
    else:
        raise ValueError(
            f'command not recognized: {command}'
        )
