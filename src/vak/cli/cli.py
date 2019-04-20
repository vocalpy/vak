import os
from configparser import ConfigParser
from configparser import MissingSectionHeaderError, ParsingError, DuplicateOptionError, DuplicateSectionError

from .train import train
from .predict import predict
from .learncurve import learncurve
from .summary import summary
from .prep import make_data
from ..config import parse_spect_config, parse_data_config, parse_train_config, \
    parse_predict_config, parse_output_config
from ..config.parse import _get_nets_config


def cli(command, config_files):
    """command-line interface

    Parameters
    ----------
    command : string
        One of {'prep', 'train', 'predict', 'finetune', 'learncurve', 'summary'}
    config_files : string or list
        config.ini files
    """
    for config_file in config_files:
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
            if spect_params is None and data_config.mat_spect_files_path is None:
                # then user needs to specify spectrogram parameters
                raise ValueError('No annotation_path specified in config_file that '
                                 'would point to annotated spectrograms, but no '
                                 'parameters provided to generate spectrograms '
                                 'either.')

            make_data(labelset=data_config.labelset,
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

        elif command == 'train':
            train_config = parse_train_config(config_obj, config_file)
            nets_config = _get_nets_config(config_obj, train_config.networks)
            spect_params = parse_spect_config(config_obj)
            output_config = parse_output_config(config_obj)
            train(train_data_dict_path=train_config.train_data_dict_path,
                  val_data_dict_path=train_config.val_data_dict_path,
                  spect_params=spect_params,
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
                    labels_mapping_path=predict_config.labels_mapping_path,
                    spect_params=spect_params,
                    dir_to_predict=predict_config.dir_to_predict,
                    spect_scaler_path=predict_config.spect_scaler_path)

        elif command == 'learncurve':
            train_config = parse_train_config(config_obj, config_file)
            nets_config = _get_nets_config(config_obj, train_config.networks)
            spect_params = parse_spect_config(config_obj)
            data_config = parse_data_config(config_obj, config_file)
            output_config = parse_output_config(config_obj)
            if train_config.train_data_dict_path is None:
                raise ValueError("must set 'train_data_path' option in [TRAIN] section of config.ini file "
                                 "before running 'learncurve'")
            if train_config.val_data_dict_path is None:
                raise ValueError("must set 'val_data_path' option in [TRAIN] section of config.ini file "
                                 "before running 'learncurve'")
            learncurve(train_data_dict_path=train_config.train_data_dict_path,
                       val_data_dict_path=train_config.val_data_dict_path,
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
                       root_results_dir=output_config.root_results_dir,
                       save_transformed_data=data_config.save_transformed_data)

        elif command == 'summary':
            train_config = parse_train_config(config_obj, config_file)
            nets_config = _get_nets_config(config_obj, train_config.networks)
            data_config = parse_data_config(config_obj, config_file)
            output_config = parse_output_config(config_obj)
            summary(results_dirname=output_config.results_dirname,
                    train_data_dict_path=train_config.train_data_dict_path,
                    networks=nets_config,
                    train_set_durs=train_config.train_set_durs,
                    num_replicates=train_config.num_replicates,
                    labelset=data_config.labelset,
                    test_data_dict_path=train_config.test_data_dict_path,
                    normalize_spectrograms=train_config.normalize_spectrograms,
                    save_transformed_data=data_config.save_transformed_data)
