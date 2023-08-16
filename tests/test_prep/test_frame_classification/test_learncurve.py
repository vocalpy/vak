import json
import shutil

import numpy as np
import pandas as pd
import pytest

import vak.common.converters
import vak.common.labels
import vak.common.paths
import vak.prep.frame_classification


@pytest.mark.parametrize(
    'model_name, audio_format, annot_format, input_type',
    [
        ('TweetyNet', 'cbin', 'notmat', 'spect')
    ]
)
def test_make_learncurve_splits_from_dataset_df(
    model_name, audio_format, annot_format, input_type, specific_config, device, tmp_path,
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
    ]
    toml_path = specific_config(
        config_type="learncurve",
        model=model_name,
        audio_format=audio_format,
        annot_format=annot_format,
        options_to_change=options_to_change,
    )
    cfg = vak.config.parse.from_toml_path(toml_path)

    dataset_path = cfg.learncurve.dataset_path
    metadata = vak.datasets.frame_classification.Metadata.from_dataset_path(dataset_path)
    dataset_csv_path = dataset_path / metadata.dataset_csv_filename
    dataset_df = pd.read_csv(dataset_csv_path)

    labelmap_path = dataset_path / "labelmap.json"
    with labelmap_path.open("r") as f:
        labelmap = json.load(f)

    tmp_dataset_path = tmp_path / f"test_make_learncurve_splits_from_dataset_df-window-size-{window_size}"
    shutil.copytree(dataset_path, tmp_dataset_path)
    shutil.rmtree(tmp_dataset_path / 'learncurve')  # since we're about to make this and test it works

    out = vak.prep.frame_classification.learncurve.make_learncurve_splits_from_dataset_df(
        dataset_df,
        "spect",
        cfg.prep.train_set_durs,
        cfg.prep.num_replicates,
        tmp_dataset_path,
        labelmap,
        audio_format=audio_format,
    )

    assert isinstance(out, pd.DataFrame)

    # learncurve_splits_root = dataset_path / 'learncurve'
    # assert learncurve_splits_root.exists()
    #
    # learncurve_splits_path = learncurve_splits_root / 'learncurve-splits-metadata.csv'
    # assert learncurve_splits_path.exists()
    #
    # splits_df = pd.read_csv(learncurve_splits_path)

    # TODO: test each split dir exists
    # TODO: (correctly) test dataset_df has all splits
    # TODO: don't test files get made since other functions do that

    splits_df = dataset_df[
        not dataset_df.split.isin(('train', 'val', 'test'))
    ]
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
            # assert len(train_dur_replicate_df) == 1
            #
            # split_csv_path = tmp_dataset_path / train_dur_replicate_df["split_csv_filename"].item()
            # assert split_csv_path.exists()
            #
            # split_df = pd.read_csv(split_csv_path)
            assert train_dur_replicate_df.duration.sum() >= train_dur
            #
            # for vec_name in ("frames_npy_path", "source_inds", "window_inds"):
            #     vec_filename = train_dur_replicate_df[f'{vec_name}_npy_filename'].item()
            #     vector_path = learncurve_splits_root / vec_filename
            #     assert vector_path.exists()
