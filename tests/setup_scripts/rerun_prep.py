import os
from pathlib import Path
from configparser import ConfigParser

import vak

HERE = Path(__file__).parent
TEST_DATA_DIR = HERE.joinpath('..', 'test_data')


def main():
    config_file = str(HERE.joinpath('tmp_Makefile_config.ini'))
    if not os.path.isfile(config_file):
        raise FileNotFoundError(
            f'{config_file} not found')

    config = ConfigParser()
    config.read(config_file)
    data_config = vak.config.parse.parse_data_config(config, config_file)
    spect_params = vak.config.parse.parse_spect_config(config)
    vak.cli.prep(data_dir=data_config.data_dir,
                 labelset=data_config.labelset,
                 config_file=config_file,
                 annot_format=data_config.annot_format,
                 train_dur=data_config.total_train_set_dur,
                 test_dur=data_config.test_dur,
                 val_dur=data_config.val_dur,
                 output_dir=data_config.output_dir,
                 audio_format=data_config.audio_format,
                 spect_format=data_config.spect_format,
                 annot_file=data_config.annot_file,
                 spect_params=spect_params)

    # make a "predict.vds.json" to use when testing predict functions
    dir_to_predict = TEST_DATA_DIR.joinpath('cbins', 'gy6or6', '032412')
    for_vds_fname = dir_to_predict.name
    vds_fname = f'{for_vds_fname}.predict.vds.json'
    vak.dataset.prep(data_dir=str(dir_to_predict),
                     vds_fname=vds_fname,
                     output_dir=TEST_DATA_DIR.joinpath('vds'),
                     save_vds=True,
                     audio_format='cbin',
                     spect_params=spect_params)


if __name__ == '__main__':
    main()
