"""Fixtures used just by test_datasets"""
import json

import numpy as np
import pandas as pd
import pytest


SPLITS_JSON = {
    "splits_csv_path": "splits/inputs-targets-paths-csvs/Mouse-Pup-Call.call.id-SW.frame-dur-1.5-ms.id-data-only.train-dur-1500.0.replicate-1.splits.csv",
    "sample_id_vec_path": {
        "test": "splits/sample-id-vectors/Mouse-Pup-Call.call.id-SW.frame-dur-1.5-ms.id-data-only.train-dur-1500.0.replicate-1.splits.test.sample_ids.npy",
        "train": "splits/sample-id-vectors/Mouse-Pup-Call.call.id-SW.frame-dur-1.5-ms.id-data-only.train-dur-1500.0.replicate-1.splits.train.sample_ids.npy",
        "val": "splits/sample-id-vectors/Mouse-Pup-Call.call.id-SW.frame-dur-1.5-ms.id-data-only.train-dur-1500.0.replicate-1.splits.val.sample_ids.npy"
    },
    "inds_in_sample_vec_path": {
        "test": "splits/inds-in-sample-vectors/Mouse-Pup-Call.call.id-SW.frame-dur-1.5-ms.id-data-only.train-dur-1500.0.replicate-1.splits.test.inds_in_sample.npy",
        "train": "splits/inds-in-sample-vectors/Mouse-Pup-Call.call.id-SW.frame-dur-1.5-ms.id-data-only.train-dur-1500.0.replicate-1.splits.train.inds_in_sample.npy",
        "val": "splits/inds-in-sample-vectors/Mouse-Pup-Call.call.id-SW.frame-dur-1.5-ms.id-data-only.train-dur-1500.0.replicate-1.splits.val.inds_in_sample.npy"
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

LABELMAPS_JSON = {
    "Bengalese-Finch-Song": {
        "syllable": {
            "bl26lb16": {
                "background": 0,
                "a": 1,
                "b": 2,
                "c": 3,
                "d": 4,
                "e": 5,
                "f": 6,
                "i": 7
            },
            "gr41rd51": {
                "background": 0,
                "a": 1,
                "b": 2,
                "c": 3,
                "d": 4,
                "e": 5,
                "f": 6,
                "g": 7,
                "i": 8,
                "j": 9,
                "k": 10,
                "m": 11
            },
            "gy6or6": {
                "background": 0,
                "a": 1,
                "b": 2,
                "c": 3,
                "d": 4,
                "e": 5,
                "f": 6,
                "g": 7,
                "h": 8,
                "i": 9,
                "j": 10,
                "k": 11
            },
            "or60yw70": {
                "background": 0,
                "a": 1,
                "b": 2,
                "c": 3,
                "d": 4,
                "e": 5,
                "f": 6,
                "g": 7,
                "i": 8
            },
            "Bird0": {
                "background": 0,
                "0": 1,
                "1": 2,
                "2": 3,
                "3": 4,
                "4": 5,
                "5": 6,
                "6": 7,
                "7": 8,
                "8": 9,
                "9": 10
            },
            "Bird4": {
                "background": 0,
                "0": 1,
                "1": 2,
                "2": 3,
                "3": 4,
                "4": 5,
                "5": 6,
                "6": 7,
                "7": 8
            },
            "Bird7": {
                "background": 0,
                "0": 1,
                "1": 2,
                "2": 3,
                "3": 4,
                "4": 5,
                "5": 6,
                "6": 7
            },
            "Bird9": {
                "background": 0,
                "0": 1,
                "1": 2,
                "2": 3,
                "3": 4,
                "4": 5,
                "5": 6
            }
        }
    },
    "Canary-Song": {
        "syllable": {
            "llb3": {
                "background": 0,
                "1": 1,
                "10": 2,
                "11": 3,
                "12": 4,
                "13": 5,
                "14": 6,
                "15": 7,
                "16": 8,
                "17": 9,
                "18": 10,
                "19": 11,
                "2": 12,
                "20": 13,
                "3": 14,
                "4": 15,
                "5": 16,
                "6": 17,
                "7": 18,
                "8": 19,
                "9": 20
            },
            "llb11": {
                "background": 0,
                "1": 1,
                "10": 2,
                "11": 3,
                "12": 4,
                "13": 5,
                "14": 6,
                "15": 7,
                "16": 8,
                "17": 9,
                "18": 10,
                "19": 11,
                "2": 12,
                "20": 13,
                "21": 14,
                "22": 15,
                "23": 16,
                "24": 17,
                "25": 18,
                "26": 19,
                "27": 20,
                "3": 21,
                "4": 22,
                "5": 23,
                "6": 24,
                "7": 25,
                "8": 26,
                "9": 27
            },
            "llb16": {
                "background": 0,
                "1": 1,
                "10": 2,
                "11": 3,
                "12": 4,
                "13": 5,
                "14": 6,
                "15": 7,
                "16": 8,
                "17": 9,
                "18": 10,
                "19": 11,
                "2": 12,
                "20": 13,
                "21": 14,
                "22": 15,
                "23": 16,
                "24": 17,
                "25": 18,
                "26": 19,
                "27": 20,
                "28": 21,
                "29": 22,
                "3": 23,
                "30": 24,
                "4": 25,
                "5": 26,
                "6": 27,
                "7": 28,
                "8": 29,
                "9": 30
            }
        }
    },
    "Human-Speech": {
        "phoneme": {
            "all": {
                "background": 0,
                "aa": 1,
                "ae": 2,
                "ah": 3,
                "ao": 4,
                "aw": 5,
                "ax": 6,
                "ax-h": 7,
                "axr": 8,
                "ay": 9,
                "b": 10,
                "bcl": 11,
                "ch": 12,
                "d": 13,
                "dcl": 14,
                "dh": 15,
                "dx": 16,
                "eh": 17,
                "el": 18,
                "em": 19,
                "en": 20,
                "eng": 21,
                "epi": 22,
                "er": 23,
                "ey": 24,
                "f": 25,
                "g": 26,
                "gcl": 27,
                "h#": 28,
                "hh": 29,
                "hv": 30,
                "ih": 31,
                "ix": 32,
                "iy": 33,
                "jh": 34,
                "k": 35,
                "kcl": 36,
                "l": 37,
                "m": 38,
                "n": 39,
                "ng": 40,
                "nx": 41,
                "ow": 42,
                "oy": 43,
                "p": 44,
                "pau": 45,
                "pcl": 46,
                "q": 47,
                "r": 48,
                "s": 49,
                "sh": 50,
                "t": 51,
                "tcl": 52,
                "th": 53,
                "uh": 54,
                "uw": 55,
                "ux": 56,
                "v": 57,
                "w": 58,
                "y": 59,
                "z": 60,
                "zh": 61
            }
        }
    },
    "Mouse-Pup-Call": {
        "call": {
            "all": {
                "background": 0,
                "BK": 1,
                "BW": 2,
                "GO": 3,
                "LL": 4,
                "LO": 5,
                "MU": 6,
                "MZ": 7,
                "NB": 8,
                "PO": 9,
                "SW": 10
            }
        }
    },
    "Zebra-Finch-Song": {
        "syllable": {
            "blu285": {
                "background": 0,
                "syll_0": 1,
                "syll_1": 2,
                "syll_2": 3,
                "syll_3": 4,
                "syll_4": 5,
                "syll_5": 6
            }
        }
    }
}


@pytest.fixture
def mock_biosoundsegbench_dataset(tmp_path):
    dataset_path = tmp_path / "BioSoundSegBench"
    dataset_path.mkdir()
    splits_dir = dataset_path / "splits"
    splits_dir.mkdir()

    inputs_targets_csv_dir = splits_dir / "inputs-targets-paths-csvs"
    inputs_targets_csv_dir.mkdir()
    df = pd.DataFrame.from_records(INPUTS_TARGETS_CSV_RECORDS)
    splits_csv = df.to_csv(inputs_targets_csv_dir / "Mouse-Pup-Call.call.id-SW.frame-dur-1.5-ms.id-data-only.train-dur-1500.0.replicate-1.splits.csv")
    df.to_csv(splits_csv)

    sample_id_vecs_dir = splits_dir / "sample-id-vectors"
    sample_id_vecs_dir.mkdir()
    inds_in_sample_vecs_dir = splits_dir / "inds-in-sample-vectors"
    inds_in_sample_vecs_dir.mkdir()

    for split in "train", "val", "test":
        sample_id_vec = np.zeros(10)
        np.save(sample_id_vecs_dir / f"Mouse-Pup-Call.call.id-SW.frame-dur-1.5-ms.id-data-only.train-dur-1500.0.replicate-1.splits.{split}.sample_ids.npy", sample_id_vec)
        inds_in_sample_vec = np.arange(10)
        np.save(inds_in_sample_vecs_dir / f"Mouse-Pup-Call.call.id-SW.frame-dur-1.5-ms.id-data-only.train-dur-1500.0.replicate-1.splits.{split}.inds_in_sample.npy", inds_in_sample_vec)

    splits_path = dataset_path / "Mouse-Pup-Call.call.id-SW.frame-dur-1.5-ms.id-data-only.train-dur-1500.0.replicate-1.splits.json"
    with splits_path.open('w') as fp:
        json.dump(SPLITS_JSON, fp)

    with (dataset_path / 'labelmaps.json').open('w') as fp:
        json.dump(LABELMAPS_JSON, fp)

    return dataset_path, splits_path
