import crowsetta
import numpy as np
import pytest

import vak.annotation
import vak.io.audio


def test_files_from_dir(annot_dir_notmat, annot_files_notmat):
    annot_files_from_dir = vak.annotation.files_from_dir(
        annot_dir_notmat, annot_format="notmat"
    )

    annot_files_notmat = [str(annot_file) for annot_file in annot_files_notmat]
    assert sorted(annot_files_from_dir) == sorted(annot_files_notmat)


@pytest.mark.parametrize(
    "source_type, source_format, annot_format",
    [
        ("audio", "cbin", "notmat"),
        ("audio", "wav", "birdsong-recognition-dataset"),
        ("spect", "mat", "yarden"),
    ],
)
def test_source_annot_map(
    source_type,
    source_format,
    annot_format,
    audio_list_factory,
    spect_list_mat,
    specific_annot_list,
):
    if source_type == "audio":
        source_files = audio_list_factory(source_format)
    else:
        source_files = spect_list_mat
    annot_list = specific_annot_list(annot_format)
    source_annot_map = vak.annotation.map_annotated_to_annot(
        source_files=source_files, annot_list=annot_list
    )

    # test all the audio paths made it into the map
    source_files_from_map = list(source_annot_map.keys())
    for source_file in source_files:
        assert source_file in source_files_from_map

    # test all the annots made it into the map
    annot_list_from_map = list(source_annot_map.values())
    for annot in annot_list:
        assert annot in annot_list_from_map

    # test all mappings are correct
    for source_path, annot in list(source_annot_map.items()):
        assert vak.annotation.audio_stem_from_path(
            annot.audio_path
        ) == vak.annotation.audio_stem_from_path(source_path)


@pytest.mark.parametrize(
    'annot, duration, expected_result',
    [
        # common case: unlabeled periods between annotated segments, and before and after.
        # Function should return true.
        (
                crowsetta.Annotation(seq=crowsetta.Sequence.from_keyword(
                    onsets_s=np.array([1.0, 2.0, 3.0, 4.0]),
                    offsets_s=np.array([1.5, 2.5, 3.5, 4.5]),
                    labels=np.array(['a', 'b', 'c', 'd'])),
                    annot_path='/dummy/annot/path/annot.csv'),
            5.0, True
        ),
        # other common case we expect: all periods are labeled. Function should return False.
        (
                crowsetta.Annotation(seq=crowsetta.Sequence.from_keyword(
                    onsets_s=np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
                    offsets_s=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
                    labels=np.array(['a', 'b', 'c', 'd'])),
                    annot_path='/dummy/annot/path/annot.csv'),
                5.0, False
        ),
        # edge case 1: only unlabeled period is before annotated segments
        (
                crowsetta.Annotation(seq=crowsetta.Sequence.from_keyword(
                    onsets_s=np.array([1.0, 2.0, 3.0, 4.0]),
                    offsets_s=np.array([2.0, 3.0, 4.0, 5.0]),
                    labels=np.array(['a', 'b', 'c', 'd'])),
                annot_path='/dummy/annot/path/annot.csv'),
                5.0, True
        ),
        # edge case 2: only unlabeled period is after annotated segments
        (
                crowsetta.Annotation(seq=crowsetta.Sequence.from_keyword(
                    onsets_s=np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
                    offsets_s=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
                    labels=np.array(['a', 'b', 'c', 'd'])),
                annot_path='/dummy/annot/path/annot.csv'),
                6.0, True
        ),
    ]
)
def test_has_unlabeled(annot, duration, expected_result):
    """Test ``vak.annotation.has_unlabeled``,
    including edge cases as discussed in https://github.com/vocalpy/vak/issues/243
    """
    has_unlabeled = vak.annotation.has_unlabeled(annot, duration)
    assert has_unlabeled == expected_result
