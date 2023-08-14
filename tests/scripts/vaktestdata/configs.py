"""Helper functions for moving and modifying configs"""
import logging
import pathlib
import shutil

# TODO: use tomli
import toml

from . import constants


logger = logging.getLogger(__name__)


def copy_config_files():
    """copy config files from setup to data_for_tests/configs

    the copied files are the ones that get modified when this setup script runs,
    while the originals in this directory remain unchanged.
    """
    logger.info(
        f"Making directory to copy config files:\n{constants.GENERATED_TEST_CONFIGS_ROOT}"
    )
    constants.GENERATED_TEST_CONFIGS_ROOT.mkdir(parents=True)

    logger.info(
        "Copying config files run to generate test data from ./tests/data_for_tests/configs to "
        f"{constants.GENERATED_TEST_CONFIGS_ROOT}"
    )


    copied_configs = []

    for config_metadata in constants.CONFIG_METADATA:
        toml_path = constants.TEST_CONFIGS_ROOT / config_metadata.filename
        if not toml_path.exists():
            raise FileNotFoundError(f"{toml_path} not found")

        dst = constants.GENERATED_TEST_CONFIGS_ROOT.joinpath(toml_path.name)
        logger.info(f"\tCopying '{toml_path.name}'")
        shutil.copy(src=toml_path, dst=dst)
        copied_configs.append(dst)

    return copied_configs


def add_dataset_path_from_prepped_configs():
    """This helper function goes through all configs in
    :data:`vaktestdata.constants.CONFIG_METADATA`
    and for any that have a filename for the attribute
    "use_dataset_from_config", it sets the option 'dataset_path'
    in the config file that the metadata corresponds to
    to the same option from the file specified
    by the attribute.
    """
    configs_to_change = [
        config_metadata
        for config_metadata in constants.CONFIG_METADATA
        if config_metadata.use_dataset_from_config is not None
    ]

    for config_metadata in configs_to_change:
        config_to_change_path = constants.GENERATED_TEST_CONFIGS_ROOT / config_metadata.filename
        if config_metadata.config_type == 'train_continue':
            section = 'TRAIN'
        else:
            section = config_metadata.config_type.upper()

        config_dataset_path = constants.GENERATED_TEST_CONFIGS_ROOT / config_metadata.use_dataset_from_config

        with config_dataset_path.open("r") as fp:
            dataset_config_toml = toml.load(fp)
        dataset_path = dataset_config_toml[section]['dataset_path']
        with config_to_change_path.open("r") as fp:
            config_to_change_toml = toml.load(fp)
        config_to_change_toml[section]['dataset_path'] = dataset_path
        with config_to_change_path.open("w") as fp:
            toml.dump(config_to_change_toml, fp)


def fix_options_in_configs(config_metadata_list, command, single_train_result=True):
    """Fix values assigned to options in predict and eval configs.

    Need to do this because both predict and eval configs have options
    that can only be assigned *after* running the corresponding `train` config.
    """
    if command not in ('eval', 'predict', 'train_continue'):
        raise ValueError(
            f'invalid command to fix config options: {command}'
        )

    for config_metadata in config_metadata_list:
        config_to_fix = constants.GENERATED_TEST_CONFIGS_ROOT / config_metadata.filename
        config_to_use_result_from = constants.GENERATED_TEST_CONFIGS_ROOT / config_metadata.use_result_from_config

        # now use the config to find the results dir and get the values for the options we need to set
        # which are checkpoint_path, spect_scaler_path, and labelmap_path
        with config_to_use_result_from.open("r") as fp:
            config_toml = toml.load(fp)
        root_results_dir = pathlib.Path(config_toml["TRAIN"]["root_results_dir"])
        results_dir = sorted(root_results_dir.glob("results_*"))
        if len(results_dir) > 1:
            if single_train_result:
                raise ValueError(
                    f"Did not find just a single results directory in root_results_dir from train_config:\n"
                    f"{config_toml}"
                    f"root_results_dir was: {root_results_dir}"
                    f'Matches for "results_*" were: {results_dir}'
                )
            else:
                results_dir = results_dir[-1]
        elif len(results_dir) == 1:
            results_dir = results_dir[0]
        else:
            raise ValueError(
                f"Did not find a results directory in root_results_dir from train_config:\n"
                f"{config_toml}"
                f"root_results_dir was:\n{root_results_dir}"
                f'Matches for "results_*" were:\n{results_dir}'
            )

        # these are the only options whose values we need to change
        # and they are the same for both predict and eval
        checkpoint_path = sorted(results_dir.glob("**/checkpoints/checkpoint.pt"))[0]
        if config_toml['TRAIN']['normalize_spectrograms']:
            spect_scaler_path = sorted(results_dir.glob("StandardizeSpect"))[0]
        else:
            spect_scaler_path = None
        labelmap_path = sorted(results_dir.glob("labelmap.json"))[0]

        # now add these values to corresponding options in predict / eval config
        with config_to_fix.open("r") as fp:
            config_toml = toml.load(fp)

        if command == 'train_continue':
            section = 'TRAIN'
        else:
            section = command.upper()

        config_toml[section]["checkpoint_path"] = str(checkpoint_path)
        if spect_scaler_path:
            config_toml[section]["spect_scaler_path"] = str(spect_scaler_path)
        else:
            if 'spect_scaler_path' in config_toml[section]:
                # remove any existing 'spect_scaler_path' option
                del config_toml[section]["spect_scaler_path"]
        if command != 'train_continue':  # train always gets labelmap from dataset dir, not from a config option
            config_toml[section]["labelmap_path"] = str(labelmap_path)

        with config_to_fix.open("w") as fp:
            toml.dump(config_toml, fp)
