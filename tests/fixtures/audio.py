"""fixtures relating to audio files"""
import pytest

from .annot import LABELSET_NOTMAT, ANNOT_LIST_NOTMAT
from .test_data import SOURCE_TEST_DATA_ROOT


@pytest.fixture
def default_spect_params():
    return dict(
        fft_size=512,
        step_size=64,
        freq_cutoffs=(500, 10000),
        thresh=6.25,
        transform_type="log_spect",
        freqbins_key="f",
        timebins_key="t",
        spect_key="s",
        audio_path_key="audio_path",
    )


AUDIO_DIR_CBIN = SOURCE_TEST_DATA_ROOT.joinpath("audio_cbin_annot_notmat", "gy6or6", "032312")


@pytest.fixture
def audio_dir_cbin():
    return AUDIO_DIR_CBIN


AUDIO_LIST_CBIN = sorted(AUDIO_DIR_CBIN.glob("*.cbin"))


@pytest.fixture
def audio_list_cbin():
    return AUDIO_LIST_CBIN

LABELSET_NOTMAT_AS_SET = set(LABELSET_NOTMAT)


AUDIO_LIST_CBIN_ALL_LABELS_IN_LABELSET = []
AUDIO_LIST_CBIN_LABELS_NOT_IN_LABELSET = []
for audio_path in AUDIO_LIST_CBIN:
    audio_fname = audio_path.name
    annot = [
        annot for annot in ANNOT_LIST_NOTMAT if annot.notated_path.name == audio_fname
    ]
    assert len(annot) == 1
    annot = annot[0]
    if set(annot.seq.labels).issubset(LABELSET_NOTMAT_AS_SET):
        AUDIO_LIST_CBIN_ALL_LABELS_IN_LABELSET.append(audio_path)
    else:
        AUDIO_LIST_CBIN_LABELS_NOT_IN_LABELSET.append(audio_path)

@pytest.fixture
def audio_list_cbin_all_labels_in_labelset():
    """list of .cbin audio files where all labels in associated annotation **are** in labelset"""
    return AUDIO_LIST_CBIN_ALL_LABELS_IN_LABELSET


@pytest.fixture
def audio_list_cbin_labels_not_in_labelset():
    """list of .cbin audio files where some labels in associated annotation are **not** in labelset"""
    return AUDIO_LIST_CBIN_LABELS_NOT_IN_LABELSET


@pytest.fixture
def audio_list_factory(audio_list_cbin):
    """factory fixture, returns a function that
    returns a fixture containing a list of Annotation objects,
    given a specified annotation format

    so that unit tests can be parameterized with annotation format names
    """
    FORMAT_AUDIO_LIST_FIXTURE_MAP = {
        "audio_cbin_annot_notmat": audio_list_cbin,
    }

    def _audio_list_factory(audio_format, annot_format):
        key = f'audio_{audio_format}_annot_{annot_format}'
        return FORMAT_AUDIO_LIST_FIXTURE_MAP[key]

    return _audio_list_factory


@pytest.fixture
def specific_audio_list(
    audio_list_cbin,
    audio_list_cbin_all_labels_in_labelset,
    audio_list_cbin_labels_not_in_labelset,
):
    def _specific_audio_list(spect_format, qualifier=None):
        MAP = {
            "cbin": {
                None: audio_list_cbin,
                "all_labels_in_labelset": audio_list_cbin_all_labels_in_labelset,
                "labels_not_in_labelset": audio_list_cbin_labels_not_in_labelset,
            },
        }
        return MAP[spect_format][qualifier]

    return _specific_audio_list
