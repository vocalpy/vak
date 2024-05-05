# Do this here to suppress warnings before we import vak
import logging
import shutil
import warnings

from numba.core.errors import NumbaDeprecationWarning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)

import pandas as pd
import tomlkit

import vak

from . import constants


logger = logging.getLogger(__name__)


def set_up_source_files_and_csv_files_for_frame_classification_models():
    """Set up source files and csv files
    used when testing functionality for frame classification models.

    This function does the following
    - First, get only config files that have the model family set to "frame_classification"
    - Then for all those config files:
      - Generate spectrograms for all the ones that have "spect_output_dir"
      - Then for all the other ones that have "data_dir", set that option in the config file
    - Then for *all* the config files, run `get_or_make_source_files` (again)
      - to get a source files dataframe
      - and save this to csv
      - and then save it again with a ``'split'`` column added
    """
    # first just get configs we're going to prep later
    configs_to_make_spectrograms = [
        config_metadata
        for config_metadata in constants.CONFIG_METADATA
        if config_metadata.model_family == "frame_classification" and config_metadata.spect_output_dir is not None
    ]

    for config_metadata in configs_to_make_spectrograms:
        spect_output_dir = constants.GENERATED_SPECT_OUTPUT_DIR / config_metadata.spect_output_dir
        spect_output_dir.mkdir(parents=True)

        config_path = constants.GENERATED_TEST_CONFIGS_ROOT / config_metadata.filename
        logger.info(
            f"\nRunning :func:`vak.prep.frame_classification.get_or_make_source_files` to generate data for tests, "
            f"using config:\n{config_path.name}"
        )
        cfg = vak.config.Config.from_toml_path(config_path, tables_to_parse='prep')

        source_files_df: pd.DataFrame = vak.prep.frame_classification.get_or_make_source_files(
            data_dir=cfg.prep.data_dir,
            input_type=cfg.prep.input_type,
            audio_format=cfg.prep.audio_format,
            spect_format=cfg.prep.spect_format,
            spect_params=cfg.prep.spect_params,
            spect_output_dir=spect_output_dir,
            annot_format=cfg.prep.annot_format,
            annot_file=cfg.prep.annot_file,
            labelset=cfg.prep.labelset,
            audio_dask_bag_kwargs=cfg.prep.audio_dask_bag_kwargs,
        )

        # We copy annotation files to spect_output_dir
        # so we can "prep" from that directory later.
        # This means we have repeats of some files still, which is annoying;
        # .not.mat files are about ~1.2K though
        for annot_path in source_files_df['annot_path'].values:
            shutil.copy(annot_path, spect_output_dir)

        csv_path = constants.GENERATED_SOURCE_FILES_CSV_DIR / f'{config_metadata.filename}-source-files.csv'
        source_files_df.to_csv(csv_path, index=False)

        config_toml: dict = vak.config.load._load_toml_from_path(config_path)
        purpose = vak.cli.prep.purpose_from_toml(config_toml, config_path)
        dataset_df: pd.DataFrame = vak.prep.frame_classification.assign_samples_to_splits(
            purpose,
            source_files_df,
            dataset_path=spect_output_dir,
            train_dur=cfg.prep.train_dur,
            val_dur=cfg.prep.val_dur,
            test_dur=cfg.prep.test_dur,
            labelset=cfg.prep.labelset,
        )
        source_files_with_split_csv_path = (
                constants.GENERATED_SOURCE_FILES_WITH_SPLITS_CSV_DIR /
                f'{config_metadata.filename}-source-files-with-split.csv'
        )
        dataset_df.to_csv(source_files_with_split_csv_path)

    configs_to_add_data_dir = [
        config_metadata
        for config_metadata in constants.CONFIG_METADATA
        if config_metadata.model_family == "frame_classification" and config_metadata.data_dir is not None
    ]

    for config_metadata in configs_to_add_data_dir:
        config_path = constants.GENERATED_TEST_CONFIGS_ROOT / config_metadata.filename
        logger.info(
            f"\nRunning :func:`vak.prep.frame_classification.get_or_make_source_files` to generate data for tests, "
            f"using config:\n{config_path.name}"
        )

        with config_path.open("r") as fp:
            tomldoc = tomlkit.load(fp)
        data_dir = constants.GENERATED_TEST_DATA_ROOT / config_metadata.data_dir
        tomldoc['vak']['prep']['data_dir'] = str(data_dir)
        with config_path.open("w") as fp:
            tomlkit.dump(tomldoc, fp)

        cfg = vak.config.Config.from_toml_path(config_path, tables_to_parse='prep')

        source_files_df: pd.DataFrame = vak.prep.frame_classification.get_or_make_source_files(
            data_dir=cfg.prep.data_dir,
            input_type=cfg.prep.input_type,
            audio_format=cfg.prep.audio_format,
            spect_format=cfg.prep.spect_format,
            spect_params=cfg.prep.spect_params,
            spect_output_dir=None,
            annot_format=cfg.prep.annot_format,
            annot_file=cfg.prep.annot_file,
            labelset=cfg.prep.labelset,
            audio_dask_bag_kwargs=cfg.prep.audio_dask_bag_kwargs,
        )

        csv_path = constants.GENERATED_SOURCE_FILES_CSV_DIR / f'{config_metadata.filename}-source-files.csv'
        source_files_df.to_csv(csv_path, index=False)

        config_toml: dict = vak.config.load._load_toml_from_path(config_path)
        purpose = vak.cli.prep.purpose_from_toml(config_toml, config_path)
        dataset_df: pd.DataFrame = vak.prep.frame_classification.assign_samples_to_splits(
            purpose,
            source_files_df,
            dataset_path=data_dir,
            train_dur=cfg.prep.train_dur,
            val_dur=cfg.prep.val_dur,
            test_dur=cfg.prep.test_dur,
            labelset=cfg.prep.labelset,
        )
        source_files_with_split_csv_path = (
                constants.GENERATED_SOURCE_FILES_WITH_SPLITS_CSV_DIR /
                f'{config_metadata.filename}-source-files-with-split.csv'
        )
        dataset_df.to_csv(source_files_with_split_csv_path)

    configs_without_spect_output_or_data_dir_to_change = [
        config_metadata
        for config_metadata in constants.CONFIG_METADATA
        if config_metadata.model_family == "frame_classification" and (
                config_metadata.spect_output_dir is None and config_metadata.data_dir is None
        )
    ]
    for config_metadata in configs_without_spect_output_or_data_dir_to_change:
        config_path = constants.GENERATED_TEST_CONFIGS_ROOT / config_metadata.filename
        if not config_path.exists():
            raise FileNotFoundError(f"{config_path} not found")
        logger.info(
            f"\nRunning :func:`vak.prep.frame_classification.get_or_make_source_files` to generate data for tests, "
            f"using config:\n{config_path.name}"
        )
        cfg = vak.config.Config.from_toml_path(config_path, tables_to_parse='prep')
        source_files_df: pd.DataFrame = vak.prep.frame_classification.get_or_make_source_files(
            data_dir=cfg.prep.data_dir,
            input_type=cfg.prep.input_type,
            audio_format=cfg.prep.audio_format,
            spect_format=cfg.prep.spect_format,
            spect_params=cfg.prep.spect_params,
            spect_output_dir=None,
            annot_format=cfg.prep.annot_format,
            annot_file=cfg.prep.annot_file,
            labelset=cfg.prep.labelset,
            audio_dask_bag_kwargs=cfg.prep.audio_dask_bag_kwargs,
        )

        csv_path = constants.GENERATED_SOURCE_FILES_CSV_DIR / f'{config_metadata.filename}-source-files.csv'
        source_files_df.to_csv(csv_path, index=False)

        config_toml: dict = vak.config.load._load_toml_from_path(config_path)
        purpose = vak.cli.prep.purpose_from_toml(config_toml, config_path)
        dataset_df: pd.DataFrame = vak.prep.frame_classification.assign_samples_to_splits(
            purpose,
            source_files_df,
            dataset_path=cfg.prep.data_dir,
            train_dur=cfg.prep.train_dur,
            val_dur=cfg.prep.val_dur,
            test_dur=cfg.prep.test_dur,
            labelset=cfg.prep.labelset,
        )
        source_files_with_split_csv_path = (
                constants.GENERATED_SOURCE_FILES_WITH_SPLITS_CSV_DIR /
                f'{config_metadata.filename}-source-files-with-split.csv'
        )
        dataset_df.to_csv(source_files_with_split_csv_path)
