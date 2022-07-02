"""fixtures relating to audio files"""
import pytest


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


@pytest.fixture
def audio_dir_cbin(source_test_data_root):
    return source_test_data_root.joinpath("audio_cbin_annot_notmat", "gy6or6", "032312")


@pytest.fixture
def audio_list_cbin(audio_dir_cbin):
    return sorted(audio_dir_cbin.glob("*.cbin"))


@pytest.fixture
def audio_list_cbin_all_labels_in_labelset(
    audio_list_cbin, annot_list_notmat, labelset_notmat
):
    """list of .cbin audio files where all labels in associated annotation **are** in labelset"""
    labelset_notmat = set(labelset_notmat)
    audio_list_labels_in_labelset = []
    for audio_path in audio_list_cbin:
        audio_fname = audio_path.name
        annot = [
            annot for annot in annot_list_notmat if annot.audio_path.name == audio_fname
        ]
        assert len(annot) == 1
        annot = annot[0]
        if set(annot.seq.labels).issubset(labelset_notmat):
            audio_list_labels_in_labelset.append(audio_path)

    return audio_list_labels_in_labelset


@pytest.fixture
def audio_list_cbin_labels_not_in_labelset(
    audio_list_cbin, annot_list_notmat, labelset_notmat
):
    """list of .cbin audio files where some labels in associated annotation are **not** in labelset"""
    labelset_notmat = set(labelset_notmat)
    audio_list_labels_in_labelset = []
    for audio_path in audio_list_cbin:
        audio_fname = audio_path.name
        annot = [
            annot for annot in annot_list_notmat if annot.audio_path.name == audio_fname
        ]
        assert len(annot) == 1
        annot = annot[0]
        if not set(annot.seq.labels).issubset(labelset_notmat):
            audio_list_labels_in_labelset.append(audio_path)

    return audio_list_labels_in_labelset


# TODO: add .WAV from TIMIT
@pytest.fixture
def audio_dir_wav(source_test_data_root):
    return source_test_data_root.joinpath("audio_wav_annot_birdsongrec", "Bird0", "Wave")


@pytest.fixture
def audio_list_wav(audio_dir_wav):
    return sorted(audio_dir_wav.glob("*.wav"))


@pytest.fixture
def audio_list_factory(audio_list_cbin, audio_list_wav):
    """factory fixture, returns a function that
    returns a fixture containing a list of Annotation objects,
    given a specified annotation format

    so that unit tests can be parameterized with annotation format names
    """
    FORMAT_AUDIO_LIST_FIXTURE_MAP = {
        "cbin": audio_list_cbin,
        "wav": audio_list_wav,
    }

    def _audio_list_factory(format):
        return FORMAT_AUDIO_LIST_FIXTURE_MAP[format]

    return _audio_list_factory
