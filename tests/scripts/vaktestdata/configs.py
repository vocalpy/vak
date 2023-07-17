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

    for toml_path in constants.CONFIGS_TO_RUN:
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


def fix_options_in_configs(config_paths, model, command, single_train_result=True):
    """Fix values assigned to options in predict and eval configs.

    Need to do this because both predict and eval configs have options
    that can only be assigned *after* running the corresponding `train` config.
    """
    if command not in ('eval', 'predict', 'train_continue'):
        raise ValueError(
            f'invalid command to fix config options: {command}'
        )
    configs_to_fix, train_configs = [], []
    # split configs into predict/eval/"train_continue" configs and other configs
    for config_path in config_paths:
        if command in config_path.name:
            configs_to_fix.append(config_path)
        elif 'train' in config_path.name and 'continue' not in config_path.name:
            train_configs.append(config_path)

    for config_to_fix in configs_to_fix:
        # figure out which 'train' config corresponds to this 'predict' or 'eval' config
        # by using 'suffix' of config file names. `train` suffix will match `predict`/'eval' suffix
        prefix, suffix = config_to_fix.name.split(command)
        train_config_to_use = []
        for train_config in train_configs:
            train_prefix, train_suffix = train_config.name.split("train")
            if train_prefix.startswith(model) and train_suffix == suffix:
                train_config_to_use.append(train_config)
        if len(train_config_to_use) > 1:
            raise ValueError(
                f"Did not find just a single train config that matches with '{command}' config:\n"
                f"{config_to_fix}\n"
                f"Matches were: {train_config_to_use}"
            )
        train_config_to_use = train_config_to_use[0]

        # now use the config to find the results dir and get the values for the options we need to set
        # which are checkpoint_path, spect_scaler_path, and labelmap_path
        with train_config_to_use.open("r") as fp:
            train_config_toml = toml.load(fp)
        root_results_dir = pathlib.Path(train_config_toml["TRAIN"]["root_results_dir"])
        results_dir = sorted(root_results_dir.glob("results_*"))
        if len(results_dir) > 1:
            if single_train_result:
                raise ValueError(
                    f"Did not find just a single results directory in root_results_dir from train_config:\n"
                    f"{train_config_to_use}"
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
                f"{train_config_to_use}"
                f"root_results_dir was:\n{root_results_dir}"
                f'Matches for "results_*" were:\n{results_dir}'
            )

        # these are the only options whose values we need to change
        # and they are the same for both predict and eval
        checkpoint_path = sorted(results_dir.glob("**/checkpoints/checkpoint.pt"))[0]
        if train_config_toml['TRAIN']['normalize_spectrograms']:
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
