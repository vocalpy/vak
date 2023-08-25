"""fixtures relating to array files containing spectrograms"""
import pytest

import vak.common.files.spect

from .annot import (
    ANNOT_LIST_NOTMAT,
    ANNOT_LIST_YARDEN,
    LABELSET_NOTMAT,
    LABELSET_YARDEN,
)
from .test_data import GENERATED_TEST_DATA_ROOT, SOURCE_TEST_DATA_ROOT


SPECT_DIR_MAT = SOURCE_TEST_DATA_ROOT.joinpath(
    "spect_mat_annot_yarden", "llb3", "spect"
)


@pytest.fixture
def spect_dir_mat():
    return SPECT_DIR_MAT


SPECT_DIR_NPZ = sorted(
        GENERATED_TEST_DATA_ROOT.joinpath(
            "prep", "train", "audio_cbin_annot_notmat", "TweetyNet"
        ).glob("*vak-frame-classification-dataset-generated*/spectrograms_generated_*")
    )[0]


@pytest.fixture
def spect_dir_npz():
    return SPECT_DIR_NPZ


@pytest.fixture
def specific_spect_dir(spect_dir_mat, spect_dir_npz):
    def _specific_spect_dir(spect_format):
        if spect_format == "mat":
            return spect_dir_mat
        elif spect_format == "npz":
            return spect_dir_npz
        else:
            raise ValueError(f"invalid spect_format: {spect_format}")

    return _specific_spect_dir


SPECT_LIST_MAT = sorted(SPECT_DIR_MAT.glob("*.mat"))


@pytest.fixture
def spect_list_mat():
    return SPECT_LIST_MAT


SPECT_LIST_NPZ = sorted(SPECT_DIR_NPZ.glob("*.spect.npz"))


@pytest.fixture
def spect_list_npz():
    return SPECT_LIST_NPZ


LABELSET_YARDEN_SET = set(LABELSET_YARDEN)
SPECT_LIST_MAT_ALL_LABELS_IN_LABELSET = []
SPECT_LIST_MAT_LABELS_NOT_IN_LABELSET = []
for spect_path in SPECT_LIST_MAT:
    audio_fname = vak.common.files.spect.find_audio_fname(spect_path)
    annot = [
        annot for annot in ANNOT_LIST_YARDEN if annot.notated_path.name == audio_fname
    ]
    assert len(annot) == 1
    annot = annot[0]
    if set(annot.seq.labels).issubset(LABELSET_YARDEN_SET):
        SPECT_LIST_MAT_ALL_LABELS_IN_LABELSET.append(spect_path)
    else:
        SPECT_LIST_MAT_LABELS_NOT_IN_LABELSET.append(spect_path)


@pytest.fixture
def spect_list_mat_all_labels_in_labelset():
    """list of .mat spectrogram files where all labels in associated annotation **are** in labelset"""
    return SPECT_LIST_MAT_ALL_LABELS_IN_LABELSET


@pytest.fixture
def spect_list_mat_labels_not_in_labelset():
    """list of .mat spectrogram files where some labels in associated annotation are **not** in labelset"""
    return SPECT_LIST_MAT_LABELS_NOT_IN_LABELSET


LABELSET_NOTMAT_SET = set(LABELSET_NOTMAT)
SPECT_LIST_NPZ_ALL_LABELS_IN_LABELSET = []
SPECT_LIST_NPZ_LABELS_NOT_IN_LABELSET = []
for spect_path in SPECT_LIST_NPZ:
    audio_fname = vak.common.files.spect.find_audio_fname(spect_path)
    annot = [
        annot for annot in ANNOT_LIST_NOTMAT if annot.notated_path.name == audio_fname
    ]
    assert len(annot) == 1
    annot = annot[0]
    if set(annot.seq.labels).issubset(LABELSET_NOTMAT_SET):
        SPECT_LIST_NPZ_ALL_LABELS_IN_LABELSET.append(spect_path)
    else:
        SPECT_LIST_NPZ_LABELS_NOT_IN_LABELSET.append(spect_path)


@pytest.fixture
def spect_list_npz_all_labels_in_labelset():
    """list of .npz spectrogram files where all labels in associated annotation **are** in labelset"""
    return SPECT_LIST_NPZ_ALL_LABELS_IN_LABELSET


@pytest.fixture
def spect_list_npz_labels_not_in_labelset():
    """list of .npz spectrogram files where some labels in associated annotation are  **not** in labelset"""
    return SPECT_LIST_NPZ_LABELS_NOT_IN_LABELSET


@pytest.fixture
def specific_spect_list(
    spect_list_mat,
    spect_list_mat_all_labels_in_labelset,
    spect_list_mat_labels_not_in_labelset,
    spect_list_npz,
    spect_list_npz_all_labels_in_labelset,
    spect_list_npz_labels_not_in_labelset,
):
    def _specific_spect_list(spect_format, qualifier=None):
        MAP = {
            "mat": {
                None: spect_list_mat,
                "all_labels_in_labelset": spect_list_mat_all_labels_in_labelset,
                "labels_not_in_labelset": spect_list_mat_labels_not_in_labelset,
            },
            "npz": {
                None: spect_list_npz,
                "all_labels_in_labelset": spect_list_npz_all_labels_in_labelset,
                "labels_not_in_labelset": spect_list_npz_labels_not_in_labelset,
            },
        }
        return MAP[spect_format][qualifier]

    return _specific_spect_list
