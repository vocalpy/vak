from pathlib import Path
import shutil

import vak

HERE = Path(__file__).parent
# convention is that all the config.ini files in setup_scripts/ that should be
# run when setting up for development have filenames of the form `setup_*_config.ini'
# e.g., 'setup_learncurve_config.ini'
PREP_CONFIGS_TO_RUN = HERE.glob('setup_*_config.toml')


def main():
    for toml_path in PREP_CONFIGS_TO_RUN:
        if not toml_path.exists():
            raise FileNotFoundError(
                f'{toml_path} not found')

        print(f"preparing datasets for tests using config: {toml_path}")
        tmp_toml_path = Path(toml_path.parent).joinpath(f'tmp_{toml_path.name}')
        print(f"\tcopying to {tmp_toml_path}")
        shutil.copy(src=toml_path, dst=tmp_toml_path)

        vak.cli.prep(toml_path=tmp_toml_path)


if __name__ == '__main__':
    main()
