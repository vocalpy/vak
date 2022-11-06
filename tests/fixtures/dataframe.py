import pandas as pd
import pytest


@pytest.fixture
def specific_dataframe(specific_prep_csv_path):
    """returns a function that will return a
    dataframe corresponding to the ``prep``ared dataset
    from a specific configuration file,
    determined by characteristics specified by the caller:
    `config_type`, `audio_format`, `spect_format`, `annot_format`
    """

    def _specific_dataframe(
        config_type,
        model,
        annot_format,
        audio_format=None,
        spect_format=None,
    ):
        csv_path = specific_prep_csv_path(
            config_type, model, annot_format, audio_format, spect_format
        )
        return pd.read_csv(csv_path)

    return _specific_dataframe


@pytest.fixture
def train_cbin_notmat_df(specific_dataframe):
    """Returns a specific dataframe
    for tests that don't need to use a factory,
    they just need some dataframe
    representing a dataset to test on,
    e.g., the ``SpectStandardize.fit_df`` method
    """
    return specific_dataframe(
        config_type="train",
        model="teenytweetynet",
        audio_format="cbin",
        annot_format="notmat"
    )
