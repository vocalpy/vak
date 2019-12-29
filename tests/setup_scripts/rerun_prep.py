from pathlib import Path
import shutil

import vak

HERE = Path(__file__).parent
# convention is that all the config.ini files in setup_scripts/ that should be
# run when setting up for development have filenames of the form `prep_*_config.ini'
# e.g., 'prep_learncurve_config.ini'
PREP_CONFIGS_TO_RUN = HERE.glob('prep_*_config.ini')


def main():
    for config_path in PREP_CONFIGS_TO_RUN:
        if not config_path.exists():
            raise FileNotFoundError(
                f'{config_path} not found')

        print(f"preparing datasets for tests using config: {config_path}")
        tmp_config_path = Path(config_path.parent).joinpath(f'tmp_{config_path.name}')
        print(f"\tcopying to {tmp_config_path}")
        shutil.copy(src=config_path, dst=tmp_config_path)

        vak.cli.prep(config_path=tmp_config_path)


if __name__ == '__main__':
    main()
