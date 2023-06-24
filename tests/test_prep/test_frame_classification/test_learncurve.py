import shutil

import numpy as np
import pandas as pd
import pytest

import vak.common.converters
import vak.prep.learncurve
import vak.datasets.seq
import vak.common.labels
import vak.common.paths


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

    dataset_path = cfg.learncurve.dataset_path
    metadata = vak.datasets.metadata.Metadata.from_dataset_path(dataset_path)
    dataset_csv_path = dataset_path / metadata.dataset_csv_filename
    dataset_df = pd.read_csv(dataset_csv_path)

    labelset_notmat = vak.common.converters.labelset_to_set(labelset_notmat)
    has_unlabeled = vak.datasets.seq.validators.has_unlabeled(dataset_csv_path)
    if has_unlabeled:
        map_unlabeled = True
    else:
        map_unlabeled = False
    labelmap = vak.common.labels.to_map(labelset_notmat, map_unlabeled=map_unlabeled)

    tmp_dataset_path = tmp_path / f"test_make_learncurve_splits_from_dataset_df-window-size-{window_size}"
    shutil.copytree(dataset_path, tmp_dataset_path)
    shutil.rmtree(tmp_dataset_path / 'learncurve')  # since we're about to make this and test it works
    tmp_dataset_csv_path = tmp_dataset_path / dataset_csv_path.name

    vak.prep.learncurve.make_learncurve_splits_from_dataset_df(
        dataset_df,
        tmp_dataset_csv_path,
        cfg.prep.train_set_durs,
        cfg.prep.num_replicates,
        tmp_dataset_path,
        window_size,
        labelmap,
    )

    learncurve_splits_root = dataset_path / 'learncurve'
    assert learncurve_splits_root.exists()

    learncurve_splits_path = learncurve_splits_root / 'learncurve-splits-metadata.csv'
    assert learncurve_splits_path.exists()

    splits_df = pd.read_csv(learncurve_splits_path)

    assert sorted(splits_df['train_dur'].unique()) == cfg.prep.train_set_durs
    assert sorted(
        splits_df['replicate_num'].unique()
    ) == list(range(1, cfg.prep.num_replicates + 1))

    for train_dur in sorted(splits_df['train_dur'].unique()):
        train_dur_df = splits_df[np.isclose(splits_df['train_dur'], train_dur)].copy()
        assert sorted(
            train_dur_df['replicate_num']
        ) == list(range(1, cfg.prep.num_replicates + 1))

        for replicate_num in sorted(train_dur_df['replicate_num']):
            train_dur_replicate_df = splits_df[
                (np.isclose(splits_df['train_dur'], train_dur)) &
                (splits_df['replicate_num'] == replicate_num)
            ]
            assert len(train_dur_replicate_df) == 1

            split_csv_path = tmp_dataset_path / train_dur_replicate_df["split_csv_filename"].item()
            assert split_csv_path.exists()

            split_df = pd.read_csv(split_csv_path)
            assert split_df[split_df.split == 'train'].duration.sum() >= train_dur

            for vec_name in ("source_ids", "source_inds", "window_inds"):
                vec_filename = train_dur_replicate_df[f'{vec_name}_npy_filename'].item()
                vector_path = learncurve_splits_root / vec_filename
                assert vector_path.exists()
