import os
from configparser import ConfigParser
from configparser import MissingSectionHeaderError, ParsingError, DuplicateOptionError, DuplicateSectionError

from .train import train
from .predict import predict
from .learncurve import learning_curve
from .prep import prep
from .. import config
from ..config import parse_learncurve_config, parse_predict_config, parse_prep_config, parse_spect_config, \
    parse_train_config


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
        prep_config = parse_prep_config(config_obj, config_file)
        spect_params = parse_spect_config(config_obj)

        prep(labelset=prep_config.labelset,
             data_dir=prep_config.data_dir,
             train_dur=prep_config.train_dur,
             test_dur=prep_config.test_dur,
             config_file=config_file,
             annot_format=prep_config.annot_format,
             val_dur=prep_config.val_dur,
             output_dir=prep_config.output_dir,
             audio_format=prep_config.audio_format,
             spect_format=prep_config.spect_format,
             annot_file=prep_config.annot_file,
             spect_params=spect_params)

    elif command == 'train':
        train(config_path=config_file)

    elif command == 'finetune':
        raise NotImplementedError

    elif command == 'predict':
        predict_config = parse_predict_config(config_obj)
        models = config.models.from_config(config_obj, predict_config.models)
        predict(predict_vds_path=predict_config.predict_vds_path,
                train_vds_path=predict_config.train_vds_path,
                checkpoint_path=predict_config.checkpoint_path,
                networks=models,
                spect_scaler_path=predict_config.spect_scaler_path)

    elif command == 'learncurve':
        if config_obj.has_option('TRAIN', 'results_dir_made_by_main_script'):
            raise ValueError(
                f"config file {config_file} already has option 'results_dir_made_by_main_script' "
                "in [TRAIN] section. \nRunning learncurve will overwrite that option."
                "Please either remove the option from this file or make a copy of the config.ini "
                "file with a new name and remove it from that file.\n"
                f"Currently the option is set to: {config_obj['TRAIN']['results_dir_made_by_main_script']}"
            )
        learncurve_config = parse_learncurve_config(config_obj, config_file)
        models = config.models.from_config(config_obj, learncurve_config.models)
        prep_config = parse_prep_config(config_obj, config_file)
        if learncurve_config.train_vds_path is None:
            raise ValueError("must set 'train_vds_path' option in [LEARNCURVE] section of config.ini file "
                             "before running 'learncurve'")
        if learncurve_config.val_vds_path is None:
            raise ValueError("must set 'val_vds_path' option in [LEARNCURVE] section of config.ini file "
                             "before running 'learncurve'")
        learning_curve(train_vds_path=learncurve_config.train_vds_path,
                       val_vds_path=learncurve_config.val_vds_path,
                       test_vds_path=learncurve_config.test_vds_path,
                       train_dur=prep_config.train_dur,
                       train_set_durs=learncurve_config.train_set_durs,
                       num_replicates=learncurve_config.num_replicates,
                       num_epochs=learncurve_config.num_epochs,
                       config_file=config_file,
                       networks=models,
                       val_error_step=learncurve_config.val_error_step,
                       checkpoint_step=learncurve_config.checkpoint_step,
                       patience=learncurve_config.patience,
                       save_only_single_checkpoint_file=learncurve_config.save_only_single_checkpoint_file,
                       normalize_spectrograms=learncurve_config.normalize_spectrograms,
                       use_train_subsets_from_previous_run=learncurve_config.use_train_subsets_from_previous_run,
                       previous_run_path=learncurve_config.previous_run_path,
                       root_results_dir=learncurve_config.root_results_dir,
                       save_transformed_data=learncurve_config.save_transformed_data)
