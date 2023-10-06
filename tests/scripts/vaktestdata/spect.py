import vak

from . import constants

CONFIG_TO_USE = constants.GENERATED_TEST_CONFIGS_ROOT / "TweetyNet_train_audio_cbin_annot_notmat.toml"


def prep_spects():
    """This function prepares a directory of .spect.npz files
    using :func:`vak.prep.spectrogram_dataset`.

    It is used to generate fixture for the data
    `fixtures.spect.spect_dir_npz`,
    that is in turn used by the fixture
    `fixtures.spect.specific_spect_dir`,
    that is used by the test
    `tests.test_prep.test_spectrogram_dataset.test_spect_helper.test_make_dataframe_of_spect_files`.
    """
    spect_dir_npz = constants.GENERATED_TEST_DATA_ROOT / "spect-dir-npz"
    spect_dir_npz.mkdir()

    cfg = vak.config.parse.from_toml_path(CONFIG_TO_USE)

    _ = vak.prep.spectrogram_dataset.prep_spectrogram_dataset(
        labelset=cfg.prep.labelset,
        data_dir=cfg.prep.data_dir,
        annot_format=cfg.prep.annot_format,
        audio_format=cfg.prep.audio_format,
        spect_params=cfg.spect_params,
        spect_output_dir=spect_dir_npz,
        audio_dask_bag_kwargs=cfg.prep.audio_dask_bag_kwargs,
    )
