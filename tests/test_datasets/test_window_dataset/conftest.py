import numpy as np
import pandas as pd
import pytest

import vak
import vak.datasets

from ...fixtures.results import GENERATED_LEARNCURVE_RESULTS_BY_MODEL


def window_dataset_from_csv_kwargs_list():
    """Returns a list of dicts, each dict kwargs for ``WindowDataset.from_csv``,
    that we use to parametrize a fixture below, ``window_dataset_from_csv_kwargs``.

    We do it this way to have one test case for each unique set of the vectors
    that represent windows in the dataset.
    There will be a unique set for each training replicate in a learncurve run.
    """
    window_dataset_from_csv_kwargs_list = []

    # hard-coded for now that we use the first (currently, only)
    # results_dir in generated/results/learncurve/teenytweetynet
    previous_run_path = GENERATED_LEARNCURVE_RESULTS_BY_MODEL['teenytweetynet'][0]
    toml_path = sorted(previous_run_path.glob('*toml'))[0]

    cfg = vak.config.parse.from_toml_path(toml_path)

    dataset_path = cfg.learncurve.dataset_path
    dataset_df = pd.read_csv(dataset_path)

    # stuff we need just to be able to instantiate window dataset
    labelmap = vak.labels.to_map(cfg.prep.labelset, map_unlabeled=True)

    train_dur_dataset_paths = vak.core.learncurve.splits.from_previous_run_path(
        previous_run_path,
    )

    for train_dur, dataset_paths in train_dur_dataset_paths.items():
        for replicate_num, this_train_dur_this_replicate_dataset_path in enumerate(
                dataset_paths
        ):
            replicate_num += 1  # so log statements below match replicate nums returned by train_dur_dataset_paths
            this_train_dur_this_replicate_results_path = (
                this_train_dur_this_replicate_dataset_path.parent
            )

            window_dataset_kwargs = dict(
                csv_path=cfg.learncurve.dataset_path,
                labelmap=labelmap,
                window_size=cfg.dataloader.window_size,
            )
            for vector_kwarg in [
                "source_ids",
                "source_inds",
                "window_inds",
            ]:
                window_dataset_kwargs[vector_kwarg] = np.load(
                    this_train_dur_this_replicate_results_path.joinpath(
                        f"{vector_kwarg}.npy"
                    )
                )
            window_dataset_from_csv_kwargs_list.append(
                window_dataset_kwargs
            )

    return window_dataset_from_csv_kwargs_list

@pytest.fixture(params=window_dataset_from_csv_kwargs_list())
def window_dataset_from_csv_kwargs(request):
    return request.param
