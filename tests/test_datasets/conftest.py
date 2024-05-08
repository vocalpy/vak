"""Fixtures used just by test_datasets"""
import json

import numpy as np
import pandas as pd
import pytest


SPLITS_JSON = {
    "splits_csv_path": "splits/inputs-targets-paths-csvs/Mouse-Pup-Call.id-SW.timebin-1.5-ms.call.id-data-only.train-dur-1500.0.replicate-1.splits.csv",
    "sample_id_vec_path": {
        "test": "splits/sample-id-vectors/Mouse-Pup-Call.id-SW.timebin-1.5-ms.call.id-data-only.train-dur-1500.0.replicate-1.splits.test.sample_ids.npy",
        "train": "splits/sample-id-vectors/Mouse-Pup-Call.id-SW.timebin-1.5-ms.call.id-data-only.train-dur-1500.0.replicate-1.splits.train.sample_ids.npy",
        "val": "splits/sample-id-vectors/Mouse-Pup-Call.id-SW.timebin-1.5-ms.call.id-data-only.train-dur-1500.0.replicate-1.splits.val.sample_ids.npy"
    },
    "inds_in_sample_vec_path": {
        "test": "splits/inds-in-sample-vectors/Mouse-Pup-Call.id-SW.timebin-1.5-ms.call.id-data-only.train-dur-1500.0.replicate-1.splits.test.inds_in_sample.npy",
        "train": "splits/inds-in-sample-vectors/Mouse-Pup-Call.id-SW.timebin-1.5-ms.call.id-data-only.train-dur-1500.0.replicate-1.splits.train.inds_in_sample.npy",
        "val": "splits/inds-in-sample-vectors/Mouse-Pup-Call.id-SW.timebin-1.5-ms.call.id-data-only.train-dur-1500.0.replicate-1.splits.val.inds_in_sample.npy"
    }
}

INPUTS_TARGETS_CSV_RECORDS = [
    {
        'frames_path': '0.wav.spect.npz',
        'multi_frame_labels_path': '0.wav.multi-frame-labels.npy',
        'binary_frame_labels_path': '0.wav.binary-frame-labels.npy',
        'boundary_frame_labels_path': '0.wav.boundary-frame-labels.npy',
        'split': 'train'
    },
    {
        'frames_path': '1.wav.spect.npz',
        'multi_frame_labels_path': '1.wav.multi-frame-labels.npy',
        'binary_frame_labels_path': '1.wav.binary-frame-labels.npy',
        'boundary_frame_labels_path': '1.wav.boundary-frame-labels.npy',
        'split': 'val'
    },
    {
        'frames_path': '2.wav.spect.npz',
        'multi_frame_labels_path': '2.wav.multi-frame-labels.npy',
        'binary_frame_labels_path': '2.wav.binary-frame-labels.npy',
        'boundary_frame_labels_path': '2.wav.boundary-frame-labels.npy',
        'split': 'test'
    },
    {
        'frames_path': '3.wav.spect.npz',
        'multi_frame_labels_path': '3.wav.multi-frame-labels.npy',
        'binary_frame_labels_path': '3.wav.binary-frame-labels.npy',
        'boundary_frame_labels_path': '3.wav.boundary-frame-labels.npy',
        'split': 'predict'
    },
]


@pytest.fixture
def mock_biosoundsegbench_dataset(tmp_path):
    dataset_path = tmp_path / "BioSoundSegBench"
    dataset_path.mkdir()
    splits_dir = dataset_path / "splits"
    splits_dir.mkdir()

    inputs_targets_csv_dir = splits_dir / "inputs-targets-paths-csvs"
    inputs_targets_csv_dir.mkdir()
    df = pd.DataFrame.from_records(INPUTS_TARGETS_CSV_RECORDS)
    splits_csv = df.to_csv(inputs_targets_csv_dir / "Mouse-Pup-Call.id-SW.timebin-1.5-ms.call.id-data-only.train-dur-1500.0.replicate-1.splits.csv")
    df.to_csv(splits_csv)

    sample_id_vecs_dir = splits_dir / "sample-id-vectors"
    sample_id_vecs_dir.mkdir()
    inds_in_sample_vecs_dir = splits_dir / "inds-in-sample-vectors"
    inds_in_sample_vecs_dir.mkdir()

    for split in "train", "val", "test":
        sample_id_vec = np.zeros(10)
        np.save(sample_id_vecs_dir / f"Mouse-Pup-Call.id-SW.timebin-1.5-ms.call.id-data-only.train-dur-1500.0.replicate-1.splits.{split}.sample_ids.npy", sample_id_vec)
        inds_in_sample_vec = np.arange(10)
        np.save(inds_in_sample_vecs_dir / f"Mouse-Pup-Call.id-SW.timebin-1.5-ms.call.id-data-only.train-dur-1500.0.replicate-1.splits.{split}.inds_in_sample.npy", inds_in_sample_vec)

    splits_path = dataset_path / "Mouse-Pup-Call.id-SW.timebin-1.5-ms.call.id-data-only.train-dur-1500.0.replicate-1.splits.json"
    with splits_path.open('w') as fp:
        json.dump(SPLITS_JSON, fp)

    return dataset_path, splits_path