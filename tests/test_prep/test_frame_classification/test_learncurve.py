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

    tmp_dataset_path = tmp_path / f"test_make_learncurve_splits_from_dataset_df"
    shutil.copytree(dataset_path, tmp_dataset_path)
    # delete all the split directories since we're about to test that we make them
    for train_dur in cfg.prep.train_set_durs:
        for replicate_num in range(1, cfg.prep.num_replicates + 1):
            train_dur_replicate_split_name = vak.common.learncurve.get_train_dur_replicate_split_name(
                    train_dur, replicate_num
                )
            split_dir = tmp_dataset_path / train_dur_replicate_split_name
            shutil.rmtree(split_dir)

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

    splits_df = dataset_df[
        ~dataset_df.split.isin(('train', 'val', 'test'))
    ]
    assert sorted(splits_df['train_dur'].unique()) == cfg.prep.train_set_durs
    assert sorted(
        splits_df['replicate_num'].unique()
    ) == list(range(1, cfg.prep.num_replicates + 1))

    # assert that each expected split name is in data frame
    all_split_names = []
    for train_dur in cfg.prep.train_set_durs:
        train_dur_df = splits_df[np.isclose(splits_df['train_dur'], train_dur)].copy()
        # assert correct number of replicates for this train duration
        assert sorted(
            train_dur_df['replicate_num']
        ) == list(range(1, cfg.prep.num_replicates + 1))

        for replicate_num in range(1, cfg.prep.num_replicates + 1):
            train_dur_replicate_split_name = vak.common.learncurve.get_train_dur_replicate_split_name(
                    train_dur, replicate_num
                )
            all_split_names.append(train_dur_replicate_split_name)

            # assert directory holding split files exists
            split_dir = tmp_dataset_path / train_dur_replicate_split_name
            assert split_dir.exists() and split_dir.is_dir()

            # assert this train_dur + replicate split exists in dataframe
            assert np.isin(train_dur_replicate_split_name, splits_df['split'].values)
            this_split_df = splits_df[splits_df['split'] == train_dur_replicate_split_name]

            # assert that it has the correct duration
            assert this_split_df['duration'].sum() >= train_dur

