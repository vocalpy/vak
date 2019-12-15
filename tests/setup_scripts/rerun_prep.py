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
    prep_config = vak.config.parse.parse_prep_config(config, config_file)
    spect_params = vak.config.parse.parse_spect_config(config)
    vak.cli.prep(data_dir=prep_config.data_dir,
                 labelset=prep_config.labelset,
                 config_file=config_file,
                 annot_format=prep_config.annot_format,
                 train_dur=prep_config.total_train_set_dur,
                 test_dur=prep_config.test_dur,
                 val_dur=prep_config.val_dur,
                 output_dir=prep_config.output_dir,
                 audio_format=prep_config.audio_format,
                 spect_format=prep_config.spect_format,
                 annot_file=prep_config.annot_file,
                 spect_params=spect_params)

    # make a "predict.vds.json" to use when testing predict functions
    dir_to_predict = TEST_DATA_DIR.joinpath('cbins', 'gy6or6', '032412')
    for_vds_fname = dir_to_predict.name
    vak_df = vak.io.dataframe.from_files(data_dir=str(dir_to_predict),
                                         audio_format='cbin',
                                         spect_params=spect_params)
    vak_df = vak.io.dataframe.add_split_col(vak_df, split='predict')
    csv_path = TEST_DATA_DIR.joinpath(f'{for_vds_fname}.predict.csv')
    vak_df.to_csv(csv_path)


if __name__ == '__main__':
    main()
