from pathlib import Path
import shutil

import vak

HERE = Path(__file__).parent
# convention is that all the config.ini files in setup_scripts/ that should be
# run when setting up for development have filenames of the form `setup_*_config.ini'
# e.g., 'setup_learncurve_config.ini'
PREP_CONFIGS_TO_RUN = HERE.glob('setup*.toml')
TEST_DATA_ROOT = HERE.joinpath('..', 'test_data')
TEST_CONFIGS_ROOT = TEST_DATA_ROOT.joinpath('configs')


def main():
    for toml_path in PREP_CONFIGS_TO_RUN:
        if not toml_path.exists():
            raise FileNotFoundError(
                f'{toml_path} not found')

        print(f"re-running vak prep to set up for test, using config: {toml_path.name}")
        test_toml_path = TEST_CONFIGS_ROOT.joinpath(toml_path.name.replace('setup', 'test'))
        print(f"\tcopying to {test_toml_path}")
        shutil.copy(src=toml_path, dst=test_toml_path)
        print(f"\trunning vak prep with copied config: {test_toml_path.name}")
        vak.cli.prep(toml_path=test_toml_path)


if __name__ == '__main__':
    main()
