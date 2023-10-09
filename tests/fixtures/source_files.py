"""Fixtures having to do with source files, i.e.,
the "raw" files that go into a data set
used with neural networks
"""
import pandas as pd
import pytest

from .test_data import GENERATED_TEST_DATA_ROOT

# copied from vaktestdata.constants; could we add this to that with sys.path? or vice versa
GENERATED_SOURCE_FILES_CSV_DIR = GENERATED_TEST_DATA_ROOT / "source-files-csv"
GENERATED_SOURCE_FILES_WITH_SPLITS_CSV_DIR = GENERATED_TEST_DATA_ROOT / "source-files-with-splits-csv"


@pytest.fixture
def specific_source_files_csv_path():
    """Factory fixture that returns a specific source file csv"""
    def _specific_source_files_csv_path(
            config_type,
            model_name,
            annot_format,
            audio_format=None,
            spect_format=None,
    ):
        if audio_format and spect_format:
            raise ValueError(
                "Specify audio_format or spect_format, not both"
            )
        if audio_format:
            csv_filename = f'{model_name}_{config_type}_audio_{audio_format}_annot_{annot_format}.toml-source-files.csv'
        elif spect_format:
            csv_filename = f'{model_name}_{config_type}_spect_{spect_format}_annot_{annot_format}.toml-source-files.csv'
        csv_path = GENERATED_SOURCE_FILES_CSV_DIR / csv_filename
        return csv_path

    return _specific_source_files_csv_path


@pytest.fixture
def specific_source_files_df(
        specific_source_files_csv_path
):
    """Factory fixture that returns a specific source file csv"""
    def _specific_source_files_df(
            config_type,
            model_name,
            annot_format,
            audio_format=None,
            spect_format=None,
    ):
        csv_path = specific_source_files_csv_path(
            config_type,
            model_name,
            annot_format,
            audio_format,
            spect_format,
        )
        df = pd.read_csv(csv_path)
        return df
    return _specific_source_files_df


@pytest.fixture
def specific_source_files_with_split_csv_path():
    """Factory fixture that returns a specific source file csv"""
    def _specific_source_files_with_split_csv_path(
            config_type,
            model_name,
            annot_format,
            audio_format=None,
            spect_format=None,
    ):
        if audio_format and spect_format:
            raise ValueError(
                "Specify audio_format or spect_format, not both"
            )
        if audio_format:
            csv_filename = f'{model_name}_{config_type}_audio_{audio_format}_annot_{annot_format}.toml-source-files-with-split.csv'
        elif spect_format:
            csv_filename = f'{model_name}_{config_type}_spect_{spect_format}_annot_{annot_format}.toml-source-files-with-split.csv'
        csv_path = GENERATED_SOURCE_FILES_WITH_SPLITS_CSV_DIR / csv_filename
        return csv_path

    return _specific_source_files_with_split_csv_path


@pytest.fixture
def specific_source_files_with_split_df(
        specific_source_files_with_split_csv_path
):
    """Factory fixture that returns a specific source file csv"""
    def _specific_source_files_with_split_df(
            config_type,
            model_name,
            annot_format,
            audio_format=None,
            spect_format=None,
    ):
        csv_path = specific_source_files_with_split_csv_path(
            config_type,
            model_name,
            annot_format,
            audio_format,
            spect_format,
        )
        df = pd.read_csv(csv_path)
        return df
    return _specific_source_files_with_split_df
