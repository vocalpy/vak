import os
from configparser import ConfigParser
from configparser import MissingSectionHeaderError, ParsingError, DuplicateOptionError, DuplicateSectionError

from .train import train
from .predict import predict
from .learncurve import learning_curve
from .prep import prep
from ..config import parse_spect_config, parse_data_config, parse_train_config, \
    parse_predict_config, parse_output_config
from ..config.parse import _get_nets_config


def cli(command, config_file):
    """command-line interface

    Parameters
    ----------
    command : string
        One of {'prep', 'train', 'predict', 'finetune', 'learncurve'}
    config_file : string
        path to a config.ini file
    """
    # check config_file exists,
    # because if it doesn't ConfigParser will just return an "empty" instance w/out sections or options
    if not os.path.isfile(config_file):
        raise FileNotFoundError(f'config file not found: {config_file}')

    try:
        config_obj = ConfigParser()
        config_obj.read(config_file)
    except (MissingSectionHeaderError, ParsingError, DuplicateOptionError, DuplicateSectionError):
        # try to add some context for users that do not spend their lives thinking about ConfigParser objects
        print(f"Error when opening the following config_file: {config_file}")
        raise
    except:
        # say something different if we can't add very good context
        print(f"Unexpected error when opening the following config_file: {config_file}")
        raise

    if command == 'prep':
        data_config = parse_data_config(config_obj, config_file)
        spect_params = parse_spect_config(config_obj)

        prep(labelset=data_config.labelset,
             data_dir=data_config.data_dir,
             train_dur=data_config.total_train_set_dur,
             test_dur=data_config.test_dur,
             config_file=config_file,
             annot_format=data_config.annot_format,
             val_dur=data_config.val_dur,
             output_dir=data_config.output_dir,
             audio_format=data_config.audio_format,
             spect_format=data_config.spect_format,
             annot_file=data_config.annot_file,
             spect_params=spect_params)

    elif command == 'train':
        train_config = parse_train_config(config_obj, config_file)
        nets_config = _get_nets_config(config_obj, train_config.networks)
        output_config = parse_output_config(config_obj)
        train(train_vds_path=train_config.train_vds_path,
              val_vds_path=train_config.val_vds_path,
              networks=nets_config,
              num_epochs=train_config.num_epochs,
              config_file=config_file,
              val_error_step=train_config.val_error_step,
              checkpoint_step=train_config.checkpoint_step,
              patience=train_config.patience,
              save_only_single_checkpoint_file=train_config.save_only_single_checkpoint_file,
              normalize_spectrograms=train_config.normalize_spectrograms,
              root_results_dir=output_config.root_results_dir,
              save_transformed_data=train_config.save_transformed_data)

    elif command == 'finetune':
        raise NotImplementedError

    elif command == 'predict':
        predict_config = parse_predict_config(config_obj)
        nets_config = _get_nets_config(config_obj, predict_config.networks)
        spect_params = parse_spect_config(config_obj)
        predict(checkpoint_path=predict_config.checkpoint_path,
                networks=nets_config,
                spect_params=spect_params,
                dir_to_predict=predict_config.dir_to_predict,
                spect_scaler_path=predict_config.spect_scaler_path)

    elif command == 'learncurve':
        if config_obj.has_option('OUTPUT', 'results_dir_made_by_main_script'):
            raise ValueError(
                f"config file {config_file} already has option 'results_dir_made_by_main_script' "
                "in [OUTPUT] section. \nRunning learncurve will overwrite that option."
                "Please either remove the option from this file or make a copy of the config.ini "
                "file with a new name and remove it from that file.\n"
                f"Currently the option is set to: {config_obj['OUTPUT']['results_dir_made_by_main_script']}"
            )
        train_config = parse_train_config(config_obj, config_file)
        nets_config = _get_nets_config(config_obj, train_config.networks)
        data_config = parse_data_config(config_obj, config_file)
        output_config = parse_output_config(config_obj)
        if train_config.train_vds_path is None:
            raise ValueError("must set 'train_vds_path' option in [TRAIN] section of config.ini file "
                             "before running 'learncurve'")
        if train_config.val_vds_path is None:
            raise ValueError("must set 'val_vds_path' option in [TRAIN] section of config.ini file "
                             "before running 'learncurve'")
        learning_curve(train_vds_path=train_config.train_vds_path,
                       val_vds_path=train_config.val_vds_path,
                       test_vds_path=train_config.test_vds_path,
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
                       root_results_dir=output_config.root_results_dir,
                       save_transformed_data=data_config.save_transformed_data)
