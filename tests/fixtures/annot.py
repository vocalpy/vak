"""fixtures relating to annotation files"""
import crowsetta
import pytest
import toml


from .test_data import SOURCE_TEST_DATA_ROOT


@pytest.fixture
def annot_file_yarden(source_test_data_root):
    return source_test_data_root.joinpath(
        "spect_mat_annot_yarden", "llb3", "llb3_annot_subset.mat"
    )


@pytest.fixture
def annot_list_yarden(annot_file_yarden):
    scribe = crowsetta.Transcriber(format="yarden")
    annot_list = scribe.from_file(annot_file_yarden)
    return annot_list


@pytest.fixture
def labelset_yarden():
    """labelset as it would be loaded from a toml file

    don't return a set because we need to use this to test functions that convert it to a set
    """
    return [
        str(an_int)
        for an_int in [1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19]
    ]


@pytest.fixture
def annot_dir_notmat(source_test_data_root):
    return source_test_data_root.joinpath("audio_cbin_annot_notmat", "gy6or6", "032312")


@pytest.fixture
def annot_files_notmat(annot_dir_notmat):
    return sorted(annot_dir_notmat.glob("*.not.mat"))


@pytest.fixture
def annot_list_notmat(annot_files_notmat):
    scribe = crowsetta.Transcriber(format="notmat")
    annot_list = scribe.from_file(annot_files_notmat)
    return annot_list


@pytest.fixture
def labelset_notmat(generated_test_configs_root):
    """labelset as it would be loaded from a toml file

    don't return a set because we need to use this to test functions that convert it to a set"""
    a_train_notmat_config = sorted(
        generated_test_configs_root.glob("*train*notmat*toml")
    )[
        0
    ]  # get first config.toml from glob list
    # doesn't really matter which config, they all have labelset
    with a_train_notmat_config.open("r") as fp:
        a_train_notmat_toml = toml.load(fp)
    labelset = a_train_notmat_toml["PREP"]["labelset"]
    return labelset


@pytest.fixture
def annot_file_birdsongrec(source_test_data_root):
    return source_test_data_root.joinpath(
        "audio_wav_annot_birdsongrec", "Bird0", "Annotation.xml"
    )


@pytest.fixture
def annot_list_birdsongrec(annot_file_birdsongrec):
    scribe = crowsetta.Transcriber(format="birdsong-recognition-dataset")
    annot_list = scribe.from_file(annot_file_birdsongrec)
    return annot_list


@pytest.fixture
def annot_dir_textgrid(source_test_data_root):
    return source_test_data_root.joinpath("audio_wav_annot_textgrid", "AGBk")


@pytest.fixture
def annot_files_textgrid(annot_dir_textgrid):
    return sorted(annot_dir_textgrid.glob("*.TextGrid"))


@pytest.fixture
def annot_list_textgrid(annot_files_textgrid):
    scribe = crowsetta.Transcriber(format="textgrid")
    annot_list = scribe.from_file(annot_files_textgrid)
    return annot_list


@pytest.fixture
def annot_dir_simple_seq(source_test_data_root):
    return source_test_data_root.joinpath("audio_cbin_annot_simple_seq", "gy6or6", "032312")


@pytest.fixture
def annot_files_simple_seq(annot_dir_simple_seq):
    return sorted(annot_dir_simple_seq.glob("*.cbin.csv"))


@pytest.fixture
def annot_list_simple_seq(annot_files_simple_seq):
    scribe = crowsetta.Transcriber(format="simple-seq")
    annot_list = scribe.from_file(annot_files_simple_seq)
    return annot_list


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
