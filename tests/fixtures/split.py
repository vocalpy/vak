"""fixtures used to test the vak.split sub-package"""
from evfuncs import load_cbin
from scipy.io import loadmat
import pytest

from vak.timebins import timebin_dur_from_vec


@pytest.fixture
def audio_cbin_annot_notmat_durs_labels(
    audio_list_cbin, annot_list_notmat, labelset_notmat
):
    """returns list of durations of .cbin audio files, and
    another list with labels from annotations in each .not.mat file
    for the associated .cbin audio file"""
    durs = []
    labels = []
    for audio_file, annot in zip(audio_list_cbin, annot_list_notmat):
        if set(annot.seq.labels).issubset(labelset_notmat):
            labels.append(annot.seq.labels)
            data, fs = load_cbin(audio_file)
            durs.append(data.shape[0] / fs)
    return durs, labels


@pytest.fixture
def spect_mat_annot_yarden_durs_labels(
    spect_list_mat, annot_list_yarden, labelset_yarden
):
    """returns list of durations of .mat spectrogram files, and
    another list with labels from annotations in the .mat file
    for each associated .mat spectrogram file"""

    durs = []
    labels = []
    for spect_file_mat, annot in zip(spect_list_mat, annot_list_yarden):
        if set(annot.seq.labels).issubset(labelset_yarden):
            labels.append(annot.seq.labels)
            mat_dict = loadmat(spect_file_mat)
            timebin_dur = timebin_dur_from_vec(mat_dict["t"])
            dur = mat_dict["s"].shape[-1] * timebin_dur
            durs.append(dur)
    return durs, labels
