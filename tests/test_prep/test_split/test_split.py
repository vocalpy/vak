import json
import math

import numpy as np
import pandas as pd
import pytest

import vak.common.annotation
import vak.prep.spectrogram_dataset.spect_helper
import vak.prep.split.split


NUM_SAMPLES = 10  # number of times to sample behavior of random-number generator


def train_test_dur_split_inds_output_matches_expected(
    train_dur,
    val_dur,
    test_dur,
    labelset,
    durs,
    labels,
    train_inds,
    val_inds,
    test_inds,
):
    for split, dur_in, inds in zip(
        ("train", "val", "test"),
        (train_dur, val_dur, test_dur),
        (train_inds, val_inds, test_inds),
    ):
        if dur_in is not None:
            dur_out = sum([durs[ind] for ind in inds])
            if dur_in >= 0:
                assert dur_out >= dur_in
            elif dur_in == -1:
                if split == "train":
                    assert math.isclose(
                        dur_out, sum(durs) - sum([durs[ind] for ind in test_inds])
                    )
                elif split == "test":
                    assert math.isclose(
                        dur_out, sum(durs) - sum([durs[ind] for ind in train_inds])
                    )

            all_lbls_this_set = [lbl for ind in inds for lbl in labels[ind]]
            assert labelset == set(all_lbls_this_set)
        else:
            assert inds is None

    assert set(train_inds).isdisjoint(set(test_inds))
    if val_dur is not None:
        assert set(train_inds).isdisjoint(set(val_inds))
        assert set(test_inds).isdisjoint(set(val_inds))

    return True


@pytest.mark.parametrize(
    "durs, labels, labelset, train_dur, val_dur, test_dur",
    [
        (
            (5, 5, 5, 5, 5),
            ([np.asarray(list("abcde")) for _ in range(5)]),
            set(list("abcde")),
            20,
            None,
            5,
        ),
        (
            (3, 2, 1, 3, 2, 3, 2, 1, 3, 2),
            ["abc", "ab", "c", "cde", "de", "abc", "ab", "c", "cde", "de"],
            set(list("abcde")),
            14,
            None,
            8,
        ),
        (
            (3, 2, 1, 3, 2, 3, 2, 1, 3, 2),
            ["abc", "ab", "c", "cde", "de", "abc", "ab", "c", "cde", "de"],
            set(list("abcde")),
            8,
            7,
            7,
        ),
    ],
)
def test_train_test_dur_split_inds_fake_data(
    durs, labels, labelset, train_dur, val_dur, test_dur
):
    for _ in range(NUM_SAMPLES):
        train_inds, val_inds, test_inds = vak.prep.split.split.train_test_dur_split_inds(
            durs, labels, labelset, train_dur, test_dur, val_dur
        )

        assert train_test_dur_split_inds_output_matches_expected(
            train_dur,
            val_dur,
            test_dur,
            labelset,
            durs,
            labels,
            train_inds,
            val_inds,
            test_inds,
        )


def test_train_test_dur_split_inds_fake_data_impossible():
    durs = (3, 2, 1, 3, 2, 3, 2, 1, 3, 2)
    labelset = set(list("abcde"))
    labels = ["abc", "ab", "c", "cde", "de", "abc", "ab", "c", "cde", "de"]
    labels = [np.asarray(list(lbl)) for lbl in labels]
    train_dur = 16
    val_dur = 2
    test_dur = 4
    with pytest.raises(ValueError):
        vak.prep.split.split.train_test_dur_split_inds(
            durs, labels, labelset, train_dur, test_dur, val_dur
        )


@pytest.mark.parametrize(
    "train_dur, val_dur, test_dur",
    [
        (35, 20, 35),
        (35, None, -1),
        (-1, None, 35),
    ],
)
def test_train_test_dur_split_inds_audio_cbin_annot_notmat(
    train_dur, val_dur, test_dur, audio_cbin_annot_notmat_durs_labels, labelset_notmat
):
    labelset_notmat = set(labelset_notmat)
    durs, labels = audio_cbin_annot_notmat_durs_labels

    for _ in range(NUM_SAMPLES):
        train_inds, val_inds, test_inds = vak.prep.split.split.train_test_dur_split_inds(
            durs, labels, labelset_notmat, train_dur, test_dur, val_dur
        )

        assert train_test_dur_split_inds_output_matches_expected(
            train_dur,
            val_dur,
            test_dur,
            labelset_notmat,
            durs,
            labels,
            train_inds,
            val_inds,
            test_inds,
        )


@pytest.mark.parametrize(
    "train_dur, val_dur, test_dur",
    [(200, 100, 200), (200, None, -1), (-1, None, 200), (200, 100, 200)],
)
def test_train_test_dur_split_inds_spect_mat(
    train_dur, val_dur, test_dur, spect_mat_annot_yarden_durs_labels, labelset_yarden
):
    labelset_yarden = set(labelset_yarden)
    durs, labels = spect_mat_annot_yarden_durs_labels
    train_dur = 200
    val_dur = 100
    test_dur = 200

    for _ in range(NUM_SAMPLES):
        train_inds, val_inds, test_inds = vak.prep.split.split.train_test_dur_split_inds(
            durs, labels, labelset_yarden, train_dur, test_dur, val_dur
        )

        assert train_test_dur_split_inds_output_matches_expected(
            train_dur,
            val_dur,
            test_dur,
            labelset_yarden,
            durs,
            labels,
            train_inds,
            val_inds,
            test_inds,
        )


def test_dataframe_None_raises():
    durs = (5, 5, 5, 5, 5)
    labelset = set(list("abcde"))
    labels = [np.asarray(list(labelset)) for _ in range(5)]

    train_dur = None
    val_dur = None
    test_dur = None

    with pytest.raises(ValueError):
        vak.prep.split.split.train_test_dur_split_inds(
            durs, labels, labelset, train_dur, test_dur, val_dur
        )


def test_dataframe_only_val_raises():
    durs = (5, 5, 5, 5, 5)
    labelset = set(list("abcde"))
    labels = [np.asarray(list(labelset)) for _ in range(5)]

    train_dur = None
    val_dur = 100
    test_dur = None

    # because we only specified duration for validation set
    with pytest.raises(ValueError):
        vak.prep.split.split.train_test_dur_split_inds(
            durs, labels, labelset, train_dur, test_dur, val_dur
        )


def test_dataframe_negative_dur_raises():
    durs = (5, 5, 5, 5, 5)
    labelset = set(list("abcde"))
    labels = [np.asarray(list(labelset)) for _ in range(5)]

    train_dur = -2
    test_dur = None
    val_dur = 100

    # because negative duration is invalid
    with pytest.raises(ValueError):
        vak.prep.split.split.train_test_dur_split_inds(
            durs, labels, labelset, train_dur, test_dur, val_dur
        )


def test_dataframe_specd_dur_gt_raises():
    durs = (5, 5, 5, 5, 5)
    labelset = set(list("abcde"))
    labels = [np.asarray(list(labelset)) for _ in range(5)]

    train_dur = 100
    test_dur = 100
    val_dur = 100
    # because total splits duration is greater than dataset duration
    with pytest.raises(ValueError):
        vak.prep.split.split.train_test_dur_split_inds(
            durs, labels, labelset, train_dur, test_dur, val_dur
        )


@pytest.mark.parametrize(
    "config_type, model_name, spect_format, audio_format, annot_format, train_dur, val_dur, test_dur",
    [
        ('train', 'TweetyNet', None, 'cbin', 'notmat', 45, None, None,),
        ('train', 'TweetyNet', None, 'cbin', 'notmat', 45, None, 30,),
        ('train', 'TweetyNet', None, 'cbin', 'notmat', 45, 15, 30,),
        ('train', 'TweetyNet', None, 'cbin', 'notmat', None, None, 30,),
        ('train', 'TweetyNet', 'mat', None, 'yarden', 200, None, None,),
        ('train', 'TweetyNet', 'mat', None, 'yarden', None, None, 200,),
        ('train', 'TweetyNet', 'mat', None, 'yarden', 200, None, 200,),
        ('train', 'TweetyNet', 'mat', None, 'yarden', 200, 80, 120,),
    ]
)
def test_split_frame_classification_dataframe(
    config_type, model_name, spect_format, audio_format, annot_format,  specific_dataset_path,
    train_dur, val_dur, test_dur,
):
    dataset_path = specific_dataset_path(
        spect_format=spect_format,
        audio_format=audio_format,
        annot_format=annot_format,
        config_type=config_type,
        model=model_name,
    )
    metadata = vak.datasets.frame_classification.Metadata.from_dataset_path(dataset_path)
    dataset_csv_path = dataset_path / metadata.dataset_csv_filename
    dataset_df = pd.read_csv(dataset_csv_path)
    dataset_df = dataset_df.drop(columns=('split'))
    labelmap_path = dataset_path / "labelmap.json"
    with labelmap_path.open("r") as f:
        labelmap = json.load(f)
    labelset = set(key for key in labelmap.keys() if key != 'unlabeled')

    dataset_df_split = vak.prep.split.split.frame_classification_dataframe(
        dataset_df, dataset_path, labelset=labelset, train_dur=train_dur, val_dur=val_dur, test_dur=test_dur
    )

    assert isinstance(dataset_df_split, pd.DataFrame)

    for split, duration in zip(
            ('train', 'val', 'test'),
            (train_dur, val_dur, test_dur),
    ):
        if duration is not None:
            duration_out = dataset_df_split[dataset_df_split["split"] == split].duration.sum()
            assert duration_out >= duration
        else:
            assert split not in dataset_df_split["split"].unique().tolist()


@pytest.mark.parametrize(
    "config_type, model_name, spect_format, audio_format, annot_format, train_dur, val_dur, test_dur",
    [
        ('train', 'ConvEncoderUMAP', None, 'cbin', 'notmat', 0.2, None, None,),
        ('train', 'ConvEncoderUMAP', None, 'cbin', 'notmat', 0.2, None, 0.15,),
        ('train', 'ConvEncoderUMAP', None, 'cbin', 'notmat', 0.2, 0.1, 0.15,),
    ]
)
def test_split_unit_dataframe(
    config_type, model_name, spect_format, audio_format, annot_format,  specific_dataset_path,
    train_dur, val_dur, test_dur,
):
    dataset_path = specific_dataset_path(
        spect_format=spect_format,
        audio_format=audio_format,
        annot_format=annot_format,
        config_type=config_type,
        model=model_name,
    )
    metadata = vak.datasets.parametric_umap.Metadata.from_dataset_path(dataset_path)
    dataset_csv_path = dataset_path / metadata.dataset_csv_filename
    dataset_df = pd.read_csv(dataset_csv_path)
    dataset_df = dataset_df.drop(columns=('split'))
    labelmap_path = dataset_path / "labelmap.json"
    with labelmap_path.open("r") as f:
        labelmap = json.load(f)
    labelset = set(key for key in labelmap.keys() if key != 'unlabeled')

    dataset_df_split = vak.prep.split.split.unit_dataframe(
        dataset_df, dataset_path, labelset=labelset, train_dur=train_dur, val_dur=val_dur, test_dur=test_dur
    )

    assert isinstance(dataset_df_split, pd.DataFrame)

    for split, duration in zip(
            ('train', 'val', 'test'),
            (train_dur, val_dur, test_dur),
    ):
        if duration is not None:
            duration_out = dataset_df_split[dataset_df_split["split"] == split].duration.sum()
            assert duration_out >= duration
        else:
            assert split not in dataset_df_split["split"].unique().tolist()