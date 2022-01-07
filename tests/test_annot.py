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
    source_annot_map = vak.annotation.source_annot_map(
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
        assert vak.annotation.recursive_stem(
            annot.audio_path
        ) == vak.annotation.recursive_stem(source_path)
