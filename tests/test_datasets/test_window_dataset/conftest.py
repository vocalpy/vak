import json

import numpy as np
import pandas as pd
import pytest

import vak
import vak.datasets

from ...fixtures.test_data import GENERATED_TEST_DATA_ROOT
from ...fixtures.config import GENERATED_TEST_CONFIGS_ROOT


# get the corresponding .toml config file that generated the dataset
A_LEARNCURVE_TOML_PATH = GENERATED_TEST_CONFIGS_ROOT / 'teenytweetynet_learncurve_audio_cbin_annot_notmat.toml'


def window_dataset_from_csv_kwargs_list():
    """Returns a list of dicts, each dict kwargs for ``WindowDataset.from_csv``,
    that we use to parametrize a fixture below, ``window_dataset_from_csv_kwargs``.

    We do it this way to have one test case for each unique set of the vectors
    that represent windows in the dataset.
    There will be a unique set for each training replicate in a learncurve run.
    """
    window_dataset_from_csv_kwargs_list = []

    cfg = vak.config.parse.from_toml_path(A_LEARNCURVE_TOML_PATH)
    dataset_path = cfg.learncurve.dataset_path
    metadata = vak.datasets.metadata.Metadata.from_dataset_path(dataset_path)
    dataset_csv_path = dataset_path / metadata.dataset_csv_filename
    dataset_df = pd.read_csv(dataset_csv_path)

    dataset_learncurve_dir = dataset_path / 'learncurve'
    splits_path = dataset_learncurve_dir / 'learncurve-splits-metadata.csv'
    splits_df = pd.read_csv(splits_path)

    # stuff we need just to be able to instantiate window dataset
    with (dataset_path / 'labelmap.json').open('r') as fp:
        labelmap = json.load(fp)

    for splits_df_row in splits_df.itertuples():
        window_dataset_kwargs = dict(
            csv_path=cfg.learncurve.dataset_path,
            labelmap=labelmap,
            window_size=cfg.dataloader.window_size,
        )
        for window_dataset_kwarg in [
            "source_ids",
            "source_inds",
            "window_inds",
        ]:
            vec_filename = getattr(splits_df_row, f'{window_dataset_kwarg}_npy_filename')
            window_dataset_kwargs[window_dataset_kwarg] = np.load(
                dataset_learncurve_dir / vec_filename
                )

        window_dataset_from_csv_kwargs_list.append(
            window_dataset_kwargs
        )

    return window_dataset_from_csv_kwargs_list


@pytest.fixture(params=window_dataset_from_csv_kwargs_list())
def window_dataset_from_csv_kwargs(request):
    return request.param
