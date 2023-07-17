"""Helper functions for moving and modifying configs"""
import shutil

# TODO: use tomli
import toml

from . import constants



def copy_config_files():
    """copy config files from setup to data_for_tests/configs

    the copied files are the ones that get modified when this setup script runs,
    while the originals in this directory remain unchanged.
    """
    print(
        "Copying config files run to generate test data from ./tests/data_for_tests/configs to "
        "./tests/data_for_tests/generated/configs"
    )

    constants.GENERATED_TEST_CONFIGS_ROOT.mkdir(parents=True)

    copied_configs = []

    for toml_path in constants.CONFIGS_TO_RUN:
        if not toml_path.exists():
            raise FileNotFoundError(f"{toml_path} not found")

        dst = constants.GENERATED_TEST_CONFIGS_ROOT.joinpath(toml_path.name)
        print(f"\tcopying to {dst}")
        shutil.copy(src=toml_path, dst=dst)
        copied_configs.append(dst)

    return copied_configs


def add_dataset_path_from_prepped_configs(target_configs, target_model, source_configs, source_model):
    for target_config_path in target_configs:
        suffix_to_match = target_config_path.name.replace(target_model, '')  # remove model name at start of config name
        source_config_path = [
            source_config_path
            for source_config_path in source_configs
            if source_config_path.name.replace(source_model, '') == suffix_to_match
        ]
        source_config_path = source_config_path[0]
        command = [
            command
            for command in COMMANDS
            if command in source_config_path.name
        ][0]
        if command == 'train_continue':
            section = 'TRAIN'
        else:
            section = command.upper()
        print(
            f"Re-using prepped dataset from model '{source_model}' config:\n{source_config_path}\n"
            f"Will use for model '{target_model}' config:\n{target_config_path}"
        )

        with source_config_path.open("r") as fp:
            source_config_toml = toml.load(fp)
        dataset_path = source_config_toml[section]['dataset_path']
        with target_config_path.open("r") as fp:
            target_config_toml = toml.load(fp)
        target_config_toml[section]['dataset_path'] = dataset_path
        with target_config_path.open("w") as fp:
            toml.dump(target_config_toml, fp)


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
        root_results_dir = Path(train_config_toml["TRAIN"]["root_results_dir"])
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
