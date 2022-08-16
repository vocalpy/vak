import tempfile

import pandas as pd
import pytest

import vak.converters
import vak.core.learncurve.train_dur_csv_paths
import vak.datasets.seq
import vak.io.dataframe
import vak.labels
import vak.paths

SPECT_KEY = "s"
TIMEBINS_KEY = "t"


@pytest.fixture
def tmp_previous_run_path_factory(tmp_path):
    def _tmp_previous_run_path(
        train_set_durs, num_replicates, dataset_csv_path_filename="fake_data_prep.csv"
    ):
        results_path = vak.paths.generate_results_dir_name_as_path(tmp_path)
        results_path.mkdir()
        csv_path = tmp_path.joinpath("data_dir").joinpath(dataset_csv_path_filename)

        for train_set_dur in train_set_durs:
            path_this_train_dur = results_path.joinpath(
                vak.core.learncurve.train_dur_csv_paths.train_dur_dirname(train_set_dur)
            )
            path_this_train_dur.mkdir()
            for replicate_num in range(1, num_replicates + 1):
                path_this_replicate = path_this_train_dur.joinpath(
                    vak.core.learncurve.train_dur_csv_paths.replicate_dirname(
                        replicate_num
                    )
                )
                path_this_replicate.mkdir()
                fake_subset_csv_path = path_this_replicate.joinpath(
                    vak.core.learncurve.train_dur_csv_paths.subset_csv_filename(
                        csv_path, train_set_dur, replicate_num
                    )
                )
                # make fake subset csv path exist
                with fake_subset_csv_path.open("w") as fp:
                    fp.write("")
                assert fake_subset_csv_path.exists()

        return results_path

    return _tmp_previous_run_path


@pytest.mark.parametrize(
    "train_set_durs, num_replicates",
    [
        ([4, 6], 2),
        ([30, 45, 60, 120, 180], 10),
    ],
)
def test_dict_from_dir(tmp_previous_run_path_factory, train_set_durs, num_replicates):
    """note: this test specifically tests that
    csv paths are sorted numerically by replicate number,
    not alphabetically.
    See https://github.com/NickleDave/vak/issues/340

    this is the main reason for factoring out the helper function ``_dict_from_dir``
    """
    previous_run_path = tmp_previous_run_path_factory(train_set_durs, num_replicates)

    train_dur_csv_paths = vak.core.learncurve.train_dur_csv_paths._dict_from_dir(
        previous_run_path
    )

    assert isinstance(train_dur_csv_paths, dict)
    assert sorted(train_dur_csv_paths.keys()) == train_set_durs
    for train_set_dur, csv_list in train_dur_csv_paths.items():
        replicate_nums = [int(csv.parent.name.split("_")[-1]) for csv in csv_list]
        assert replicate_nums == list(range(1, num_replicates + 1))


@pytest.mark.parametrize("window_size", [44, 88, 176])
def test_from_dir(
    specific_config,
    labelset_notmat,
    tmp_path,
    default_model,
    device,
    previous_run_path_factory,
    window_size,
):
    root_results_dir = tmp_path.joinpath("tmp_root_results_dir")
    root_results_dir.mkdir()

    options_to_change = [
        {
            "section": "LEARNCURVE",
            "option": "root_results_dir",
            "value": str(root_results_dir),
        },
        {"section": "LEARNCURVE", "option": "device", "value": device},
        {
            "section": "LEARNCURVE",
            "option": "previous_run_path",
            "value": str(previous_run_path_factory(default_model)),
        },
        {"section": "DATALOADER", "option": "window_size", "value": window_size},
    ]
    toml_path = specific_config(
        config_type="learncurve",
        model=default_model,
        audio_format="cbin",
        annot_format="notmat",
        options_to_change=options_to_change,
    )
    cfg = vak.config.parse.from_toml_path(toml_path)

    csv_path = cfg.learncurve.csv_path
    dataset_df = pd.read_csv(csv_path)
    timebin_dur = vak.io.dataframe.validate_and_get_timebin_dur(dataset_df)

    labelset_notmat = vak.converters.labelset_to_set(labelset_notmat)
    has_unlabeled = vak.datasets.seq.validators.has_unlabeled(csv_path, TIMEBINS_KEY)
    if has_unlabeled:
        map_unlabeled = True
    else:
        map_unlabeled = False
    labelmap = vak.labels.to_map(labelset_notmat, map_unlabeled=map_unlabeled)

    results_path = tmp_path.joinpath("test_from_dir")
    results_path.mkdir()

    previous_run_path = previous_run_path_factory(default_model)
    previous_run_path = vak.converters.expanded_user_path(previous_run_path)

    train_dur_csv_paths = vak.core.learncurve.train_dur_csv_paths.from_dir(
        previous_run_path,
        cfg.learncurve.train_set_durs,
        timebin_dur,
        cfg.learncurve.num_replicates,
        results_path,
        window_size,
        SPECT_KEY,
        TIMEBINS_KEY,
        labelmap,
    )
    assert isinstance(train_dur_csv_paths, dict)
    assert sorted(train_dur_csv_paths.keys()) == sorted(cfg.learncurve.train_set_durs)
    for train_set_dur, csv_list in train_dur_csv_paths.items():
        replicate_nums = [int(csv.parent.name.split("_")[-1]) for csv in csv_list]
        assert replicate_nums == list(range(1, cfg.learncurve.num_replicates + 1))


@pytest.mark.parametrize("window_size", [44, 88, 176])
def test_from_df(
    specific_config, labelset_notmat, default_model, device, tmp_path, window_size
):
    root_results_dir = tmp_path.joinpath("tmp_root_results_dir")
    root_results_dir.mkdir()

    options_to_change = [
        {
            "section": "LEARNCURVE",
            "option": "root_results_dir",
            "value": str(root_results_dir),
        },
        {"section": "LEARNCURVE", "option": "device", "value": device},
        {"section": "DATALOADER", "option": "window_size", "value": window_size},
    ]
    toml_path = specific_config(
        config_type="learncurve",
        model=default_model,
        audio_format="cbin",
        annot_format="notmat",
        options_to_change=options_to_change,
    )
    cfg = vak.config.parse.from_toml_path(toml_path)

    csv_path = cfg.learncurve.csv_path
    dataset_df = pd.read_csv(csv_path)
    timebin_dur = vak.io.dataframe.validate_and_get_timebin_dur(dataset_df)

    labelset_notmat = vak.converters.labelset_to_set(labelset_notmat)
    has_unlabeled = vak.datasets.seq.validators.has_unlabeled(csv_path, TIMEBINS_KEY)
    if has_unlabeled:
        map_unlabeled = True
    else:
        map_unlabeled = False
    labelmap = vak.labels.to_map(labelset_notmat, map_unlabeled=map_unlabeled)

    results_path = tmp_path.joinpath("test_from_df")
    results_path.mkdir()

    train_dur_csv_paths = vak.core.learncurve.train_dur_csv_paths.from_df(
        dataset_df,
        csv_path,
        cfg.learncurve.train_set_durs,
        timebin_dur,
        cfg.learncurve.num_replicates,
        results_path,
        labelset_notmat,
        window_size,
        SPECT_KEY,
        TIMEBINS_KEY,
        labelmap,
    )
