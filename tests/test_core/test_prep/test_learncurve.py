import json

import pandas as pd
import pytest

import vak.converters
import vak.core.prep.learncurve
import vak.datasets.seq
import vak.io.dataframe
import vak.labels
import vak.paths


@pytest.mark.parametrize("window_size", [44, 88, 176])
def test_make_learncurve_splits_from_dataset_df(
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

    csv_path = cfg.learncurve.dataset_path
    dataset_df = pd.read_csv(csv_path)
    timebin_dur = vak.io.dataframe.validate_and_get_timebin_dur(dataset_df)

    labelset_notmat = vak.converters.labelset_to_set(labelset_notmat)
    has_unlabeled = vak.datasets.seq.validators.has_unlabeled(csv_path)
    if has_unlabeled:
        map_unlabeled = True
    else:
        map_unlabeled = False
    labelmap = vak.labels.to_map(labelset_notmat, map_unlabeled=map_unlabeled)

    dataset_path = tmp_path / f"test_make_learncurve_splits_from_dataset_df-window-size-{window_size}"
    dataset_path.mkdir()

    vak.core.prep.learncurve.make_learncurve_splits_from_dataset_df(
        dataset_df,
        csv_path,
        cfg.learncurve.train_set_durs,
        timebin_dur,
        cfg.learncurve.num_replicates,
        dataset_path,
        labelset_notmat,
        window_size,
        labelmap,
    )

    learncurve_splits_root = dataset_path / 'learncurve'
    assert learncurve_splits_root.exists()

    learncurve_splits_path = learncurve_splits_root / 'learncurve-splits-metadata.json'
    assert learncurve_splits_path.exists()

    with learncurve_splits_path.open('r') as fp:
        learncurve_metadata = json.read(fp)

    metadata_keys = sorted(learncurve_metadata.keys())
    assert metadata_keys == cfg.learncurve.train_set_durs

    for train_dur, replicate_dict in sorted(learncurve_metadata.items()):
        replicate_keys = sorted(replicate_dict.keys())
        assert replicate_keys == list(range(1, cfg.learncurve.num_replicates + 1))

        for replicate_num, splits_vectors_dict in sorted(
            replicate_dict.items()
        ):
            split_csv_path = learncurve_splits_root / splits_vectors_dict["split_csv_filename"]
            assert split_csv_path.exists()

            split_df = pd.read_csv(split_csv_path)
            assert split_df[split_df.split == 'train'].duration.sum() >= train_dur

            for vec_name in ("source_ids", "source_inds", "window_inds"):
                vector_path = learncurve_splits_root / f"{vec_name}-train-dur-{train_dur}-replicate-{replicate_num}.npy"
                assert vector_path.exists()

