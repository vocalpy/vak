from configparser import ConfigParser
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

        config = ConfigParser()
        config.read(tmp_config_path)
        prep_config = vak.config.parse.parse_prep_config(config, tmp_config_path)
        spect_params = vak.config.parse.parse_spect_config(config)
        vak.cli.prep(data_dir=prep_config.data_dir,
                     labelset=prep_config.labelset,
                     config_file=tmp_config_path,
                     annot_format=prep_config.annot_format,
                     train_dur=prep_config.total_train_set_dur,
                     test_dur=prep_config.test_dur,
                     val_dur=prep_config.val_dur,
                     output_dir=prep_config.output_dir,
                     audio_format=prep_config.audio_format,
                     spect_format=prep_config.spect_format,
                     annot_file=prep_config.annot_file,
                     spect_params=spect_params)


if __name__ == '__main__':
    main()
