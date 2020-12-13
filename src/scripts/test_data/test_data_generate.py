from pathlib import Path
import shutil

import toml
import vak

HERE = Path(__file__).parent
# convention is that all the config.ini files in setup_scripts/ that should be
# run when setting up for development have filenames of the form `setup_*_config.ini'
# e.g., 'setup_learncurve_config.ini'
PREP_CONFIGS_TO_RUN = HERE.glob('setup*.toml')
TEST_DATA_ROOT = HERE.joinpath('..', 'test_data')
TEST_CONFIGS_ROOT = TEST_DATA_ROOT.joinpath('configs')


def copy_config_files():
    """copy config files from setup to test_data/configs

    the copied files are the ones that get modified when this setup script runs,
    while the originals in this directory remain unchanged.
    """
    for toml_path in PREP_CONFIGS_TO_RUN:
        if not toml_path.exists():
            raise FileNotFoundError(
                f'{toml_path} not found')

        test_toml_path = TEST_CONFIGS_ROOT.joinpath(toml_path.name.replace('setup', 'test'))
        print(f"\tcopying to {test_toml_path}")
        shutil.copy(src=toml_path, dst=test_toml_path)


def run_prep(test_config_paths):
    """run ``vak prep`` for all test configs"""
    for test_config_path in test_config_paths:
        if not test_config_path.exists():
            raise FileNotFoundError(
                f'{test_config_path} not found')
        print(f"re-running vak prep to set up for test, using config: {test_config_path.name}")
        vak.cli.prep(toml_path=test_config_path)


def run_results(test_config_paths):
    """run ``vak {command}`` for all test configs,
    where {command} is determined from the config file name
    """
    for test_config_path in test_config_paths:
        if 'train' in test_config_path.name:
            vak.cli.train(toml_path=test_config_path)
        elif 'eval' in test_config_path.name:
            vak.cli.eval(toml_path=test_config_path)
        elif 'predict' in test_config_path.name:
            vak.cli.predict(toml_path=test_config_path)
        elif 'learncurve' in test_config_path.name:
            vak.cli.learncurve.learning_curve(toml_path=test_config_path)
        else:
            raise ValueError(
                f'unable to determine command to run from config name:\n{test_config_path}'
            )


def fix_options_in_configs(test_config_paths, command):
    """fix values assigned to options in predict and eval configs

    Need to do this because both predict and eval configs have options
    that can only be assigned *after* running the corresponding `train` config
    """
    # split configs into train and predict or eval configs
    configs_to_fix = [test_config for test_config in test_config_paths if command in test_config.name]
    train_configs = [test_config for test_config in test_config_paths if 'train' in test_config.name]

    for config_to_fix in configs_to_fix:
        # figure out which 'train' config corresponds to this 'predict' or 'eval' config
        # by using 'suffix' of config file names. `train` suffix will match `predict`/'eval' suffix
        prefix, suffix = config_to_fix.name.split(command)
        train_config_to_use = []
        for train_config in train_configs:
            train_prefix, train_suffix = train_config.name.split('train')
            if train_suffix == suffix:
                train_config_to_use.append(train_config)
        if len(train_config_to_use) != 1:
            raise ValueError(
                f'did not find just a single train config that matches with predict config:\n'
                f'{config_to_fix}'
                f'Matches were: {train_config_to_use}'
            )
        train_config_to_use = train_config_to_use[0]

        # now use the config to find the results dir and get the values for the options we need to set
        # which are checkpoint_path, spect_scaler_path, and labelmap_path
        with train_config_to_use.open('r') as fp:
            train_config_toml = toml.load(fp)
        root_results_dir = Path(train_config_toml['TRAIN']['root_results_dir'])
        results_dir = sorted(root_results_dir.glob('results_*'))
        if len(results_dir) != 1:
            raise ValueError(
                f'did not find just a single results directory in root_results_dir from train_config:\n'
                f'{train_config_to_use}'
                f'root_results_dir was: {root_results_dir}'
                f'Matches for "results_*" were: {results_dir}'
            )
        results_dir = results_dir[0]
        # these are the only options whose values we need to change
        # and they are the same for both predict and eval
        checkpoint_path = sorted(results_dir.glob('**/checkpoints/checkpoint.pt'))[0]
        spect_scaler_path = sorted(results_dir.glob('StandardizeSpect'))[0]
        labelmap_path = sorted(results_dir.glob('labelmap.json'))[0]

        # now add these values to corresponding options in predict / eval config
        with config_to_fix.open('r') as fp:
            config_toml = toml.load(fp)
        config_toml[command.upper()]['checkpoint_path'] = str(checkpoint_path)
        config_toml[command.upper()]['spect_scaler_path'] = str(spect_scaler_path)
        config_toml[command.upper()]['labelmap_path'] = str(labelmap_path)
        with config_to_fix.open('w') as fp:
            toml.dump(config_toml, fp)


# need to run 'train' config before we run 'predict'
# so we can add checkpoints, etc., from training to predict
COMMANDS = (
    'train',
    'learncurve',
    'eval',
    'predict',
)


def main():
    copy_config_files()

    test_config_paths = sorted(
        TEST_CONFIGS_ROOT.glob('test*toml')
    )
    print(
        f'will generate test data from these test config files: {test_config_paths}'
    )
    for command in COMMANDS:
        print(
            f'running configs for command: {command}'
        )
        command_config_paths = [test_config_path
                                for test_config_path in test_config_paths
                                if command in test_config_path.name]
        print(
            f'using the following configs:\n{command_config_paths}'
        )
        if command == 'predict' or command == 'eval':
            # fix values for required options in predict / eval configs
            # using results from running the corresponding train configs.
            # this only works if we ran the train configs already,
            # which we should have because of ordering of COMMANDS constant above
            fix_options_in_configs(test_config_paths, command)

        run_prep(test_config_paths=command_config_paths)
        run_results(test_config_paths=command_config_paths)


if __name__ == '__main__':
    main()
