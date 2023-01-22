"""fixtures relating to annotation files"""
import crowsetta
import pytest
import toml


from .config import GENERATED_TEST_CONFIGS_ROOT
from .test_data import SOURCE_TEST_DATA_ROOT



ANNOT_FILE_YARDEN = SOURCE_TEST_DATA_ROOT.joinpath(
        "spect_mat_annot_yarden", "llb3", "llb3_annot_subset.mat"
    )


@pytest.fixture
def annot_file_yarden():
    return ANNOT_FILE_YARDEN


scribe_yarden = crowsetta.Transcriber(format="yarden")
ANNOT_LIST_YARDEN = scribe_yarden.from_file(ANNOT_FILE_YARDEN)


@pytest.fixture
def annot_list_yarden():
    return ANNOT_LIST_YARDEN


LABELSET_YARDEN = [
    str(an_int)
    for an_int in [1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19]
]


@pytest.fixture
def labelset_yarden():
    """labelset as it would be loaded from a toml file

    don't return a set because we need to use this to test functions that convert it to a set.
    We also don't use a config for this since it's entered there as a "label string"
    """
    return LABELSET_YARDEN


ANNOT_DIR_NOTMAT = SOURCE_TEST_DATA_ROOT.joinpath("audio_cbin_annot_notmat", "gy6or6", "032312")


@pytest.fixture
def annot_dir_notmat():
    return ANNOT_DIR_NOTMAT


ANNOT_FILES_NOTMAT = sorted(ANNOT_DIR_NOTMAT.glob("*.not.mat"))


@pytest.fixture
def annot_files_notmat():
    return ANNOT_FILES_NOTMAT


scribe_notmat = crowsetta.Transcriber(format="notmat")
ANNOT_LIST_NOTMAT = scribe_notmat.from_file(ANNOT_FILES_NOTMAT)


@pytest.fixture
def annot_list_notmat():
    return ANNOT_LIST_NOTMAT



a_train_notmat_config = sorted(
    GENERATED_TEST_CONFIGS_ROOT.glob("*train*notmat*toml")
)[0]  # get first config.toml from glob list
# doesn't really matter which config, they all have labelset
with a_train_notmat_config.open("r") as fp:
    a_train_notmat_toml = toml.load(fp)
LABELSET_NOTMAT = a_train_notmat_toml["PREP"]["labelset"]


@pytest.fixture
def labelset_notmat(generated_test_configs_root):
    """labelset as it would be loaded from a toml file

    don't return a set because we need to use this to test functions that convert it to a set"""
    return LABELSET_NOTMAT


ANNOT_FILE_BIRDSONGREC = SOURCE_TEST_DATA_ROOT.joinpath(
    "audio_wav_annot_birdsongrec", "Bird0", "Annotation.xml"
)


@pytest.fixture
def annot_file_birdsongrec():
    return ANNOT_FILE_BIRDSONGREC


scribe_birdsongrec = crowsetta.Transcriber(format="birdsong-recognition-dataset")
ANNOT_LIST_BIRDSONGREC = scribe_birdsongrec.from_file(ANNOT_FILE_BIRDSONGREC)


@pytest.fixture
def annot_list_birdsongrec():
    return ANNOT_LIST_BIRDSONGREC


ANNOT_DIR_TEXTGRID = SOURCE_TEST_DATA_ROOT.joinpath("audio_wav_annot_textgrid", "AGBk")


@pytest.fixture
def annot_dir_textgrid():
    return ANNOT_DIR_TEXTGRID


ANNOT_FILES_TEXTGRID = sorted(ANNOT_DIR_TEXTGRID.glob("*.TextGrid"))


@pytest.fixture
def annot_files_textgrid():
    return ANNOT_FILES_TEXTGRID


scribe_textgrid = crowsetta.Transcriber(format="textgrid")
ANNOT_LIST_TEXTGRID = scribe_textgrid.from_file(ANNOT_FILES_TEXTGRID)


@pytest.fixture
def annot_list_textgrid():
    return ANNOT_LIST_TEXTGRID


ANNOT_DIR_SIMPLE_SEQ = SOURCE_TEST_DATA_ROOT.joinpath(
    "audio_cbin_annot_simple_seq", "gy6or6", "032312"
)


@pytest.fixture
def annot_dir_simple_seq():
    return ANNOT_DIR_SIMPLE_SEQ


ANNOT_FILES_SIMPLE_SEQ = sorted(ANNOT_DIR_SIMPLE_SEQ.glob("*.cbin.csv"))


@pytest.fixture
def annot_files_simple_seq():
    return ANNOT_FILES_SIMPLE_SEQ


scribe_simple_seq = crowsetta.Transcriber(format="simple-seq")
ANNOT_LIST_SIMPLE_SEQ = scribe_simple_seq.from_file(ANNOT_FILES_SIMPLE_SEQ)


@pytest.fixture
def annot_list_simple_seq():
    return ANNOT_LIST_SIMPLE_SEQ


@pytest.fixture
def specific_annot_list(annot_list_birdsongrec,
                        annot_list_notmat,
                        annot_list_simple_seq,
                        annot_list_textgrid,
                        annot_list_yarden):
    """factory fixture, returns a function that
    returns a fixture containing a list of Annotation objects,
    given a specified annotation format

    so that unit tests can be parameterized with annotation format names
    """
    FORMAT_ANNOT_LIST_FIXTURE_MAP = {
        "birdsong-recognition-dataset": annot_list_birdsongrec,
        "notmat": annot_list_notmat,
        "simple-seq": annot_list_simple_seq,
        "textgrid": annot_list_textgrid,
        "yarden": annot_list_yarden,
    }

    def _annot_list_factory(format):
        return FORMAT_ANNOT_LIST_FIXTURE_MAP[format]

    return _annot_list_factory


@pytest.fixture
def specific_labelset(labelset_yarden, labelset_notmat):
    def _specific_labelset(annot_format):
        if annot_format == "yarden":
            return labelset_yarden
        elif annot_format == "notmat":
            return labelset_notmat
        else:
            raise ValueError(f"invalid annot_format: {annot_format}")

    return _specific_labelset


ANNOT_FILES_WITH_NO_SEGMENTS_DIR = SOURCE_TEST_DATA_ROOT / 'audio_cbin_annot_notmat' / 'gy6or6-annotated-with-no-labels'
ANNOTATED_FILES_NO_SEGMENTS = sorted(ANNOT_FILES_WITH_NO_SEGMENTS_DIR.glob('*.cbin'))
ANNOT_FILES_WITH_NO_SEGMENTS = sorted(ANNOT_FILES_WITH_NO_SEGMENTS_DIR.glob('*.not.mat'))
ANNOTATED_ANNOT_NO_SEGMENTS_TUPLES = list(
    zip(ANNOTATED_FILES_NO_SEGMENTS, ANNOT_FILES_WITH_NO_SEGMENTS)
)


@pytest.fixture(params=ANNOTATED_ANNOT_NO_SEGMENTS_TUPLES)
def annotated_annot_no_segments(request):
    """Tuple of (annotated (audio) path, annotation path),
    where the annotation file has no annotated segments in it.
    Used to test edge case for `has_unlabeled`,
    see https://github.com/vocalpy/vak/issues/378
    """
    return request.param
