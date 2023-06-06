import pathlib

import crowsetta
import evfuncs
import numpy as np
import pytest

import vak.common.annotation


def test_files_from_dir(annot_dir_notmat, annot_files_notmat):
    annot_files_from_dir = vak.common.annotation.files_from_dir(
        annot_dir_notmat, annot_format="notmat"
    )

    annot_files_notmat = [str(annot_file) for annot_file in annot_files_notmat]
    assert sorted(annot_files_from_dir) == sorted(annot_files_notmat)


@pytest.mark.parametrize(
    'path, audio_ext, expected_stem',
    [
        ('~/gy6or6/032212/gy6or6_baseline_230312_0808.138.cbin.not.mat', None, 'gy6or6_baseline_230312_0808.138'),
        ('`/gy6or6/032212/gy6or6_baseline_230312_0808.138.cbin.not.mat', 'cbin', 'gy6or6_baseline_230312_0808.138'),
        ('~/gy6or6/032212/gy6or6_baseline_230312_0808.138.cbin.not.mat', '.cbin', 'gy6or6_baseline_230312_0808.138'),
        ('~/gy6or6/032212/gy6or6_baseline_230312_0808.138.cbin', None, 'gy6or6_baseline_230312_0808.138'),
        ('`/gy6or6/032212/gy6or6_baseline_230312_0808.138.cbin', 'cbin', 'gy6or6_baseline_230312_0808.138'),
        ('~/gy6or6/032212/gy6or6_baseline_230312_0808.138.cbin', '.cbin', 'gy6or6_baseline_230312_0808.138'),
        ('gy6or6_baseline_230312_0808.138.cbin.not.mat', None, 'gy6or6_baseline_230312_0808.138'),
        ('gy6or6_baseline_230312_0808.138.cbin.not.mat', 'cbin', 'gy6or6_baseline_230312_0808.138'),
        ('gy6or6_baseline_230312_0808.138.cbin.not.mat', '.cbin', 'gy6or6_baseline_230312_0808.138'),
        ('Bird0/spectrograms/0.wav.npz', None, '0'),
        ('Bird0/spectrograms/0.wav.npz', 'wav', '0'),
        ('Bird0/spectrograms/0.wav.npz', '.wav', '0'),
        ('Bird0/spectrograms/0.wav', None, '0'),
        ('0.wav', None, '0'),
        ('0.wav', 'wav', '0'),
        ('0.wav', '.wav', '0'),
    ]
)
def test_audio_stem_from_path(path, audio_ext, expected_stem):
    stem = vak.common.annotation.audio_filename_from_path(path, audio_ext)
    assert stem == expected_stem


@pytest.mark.parametrize(
    'path, audio_ext',
    [
        ('~/gy6or6/032212/gy6or6_baseline_230312_0808.138.', None),
        ('`/gy6or6/032212/gy6or6_baseline_230312_0808.138.', 'cbin'),
        ('~/gy6or6/032212/gy6or6_baseline_230312_0808.138.', '.cbin'),
        ('~/gy6or6/032212/gy6or6_baseline_230312_0808.138.', None),
        ('`/gy6or6/032212/gy6or6_baseline_230312_0808.138.', 'cbin'),
        ('~/gy6or6/032212/gy6or6_baseline_230312_0808.138.', '.cbin'),
        ('gy6or6_baseline_230312_0808.138.', None),
        ('gy6or6_baseline_230312_0808.138.', 'cbin'),
        ('gy6or6_baseline_230312_0808.138.', '.cbin'),
        ('Bird0/spectrograms/0', None,),
        ('Bird0/spectrograms/0.', 'wav'),
        ('Bird0/spectrograms/0.', '.wav'),
        ('Bird0/spectrograms/0.', None),
        ('0.', None),
        ('0.', 'wav'),
        ('0.', '.wav'),
    ]
)
def test_audio_stem_from_path_raises(path, audio_ext):
    with pytest.raises(vak.common.annotation.AudioFilenameNotFoundError):
        vak.common.annotation.audio_filename_from_path(path, audio_ext)


@pytest.mark.parametrize(
    "source_type, source_format, annot_format, audio_ext",
    [
        ("audio", "wav", "birdsong-recognition-dataset", None),
        ("spect", "mat", "yarden", None),
        ("audio", "wav", "birdsong-recognition-dataset", "wav"),
        ("spect", "mat", "yarden", "wav"),
    ],
)
def test__map_using_notated_path(
    source_type,
    source_format,
    annot_format,
    audio_ext,
    audio_list_factory,
    spect_list_mat,
    specific_annot_list,
):
    if source_type == "audio":
        annotated_files = audio_list_factory(source_format, annot_format)
    else:
        annotated_files = spect_list_mat
    annot_list = specific_annot_list(annot_format)

    annotated_annot_map = vak.common.annotation._map_using_notated_path(
        annotated_files=annotated_files, annot_list=annot_list, audio_ext=audio_ext
    )

    # test all the notated paths made it into the map
    annotated_files_from_map = list(annotated_annot_map.keys())
    for source_file in annotated_files:
        assert str(source_file) in annotated_files_from_map

    # test all the annots made it into the map
    annot_list_from_map = list(annotated_annot_map.values())
    for annot in annot_list:
        assert annot in annot_list_from_map

    # test all mappings are correct
    for source_path, annot in list(annotated_annot_map.items()):
        assert vak.common.annotation.audio_filename_from_path(
            annot.notated_path
        ) == vak.common.annotation.audio_filename_from_path(source_path)


@pytest.mark.parametrize(
    "source_type, source_format, annot_format, annotated_ext, method",
    [
        ("audio", "cbin", "notmat", None, "remove"),
        ("audio", "wav", "textgrid", None, "replace"),
    ],
)
def test__map_using_ext(
    source_type,
    source_format,
    annot_format,
    annotated_ext,
    method,
    audio_list_factory,
    spect_list_mat,
    specific_annot_list,
):
    if source_type == "audio":
        annotated_files = audio_list_factory(source_format, annot_format)
    else:
        annotated_files = spect_list_mat
    annot_list = specific_annot_list(annot_format)

    annotated_annot_map = vak.common.annotation._map_using_ext(
        annotated_files=annotated_files, annot_list=annot_list, annot_format=annot_format,
        method=method, annotated_ext=annotated_ext
    )

    # test all the audio paths made it into the map
    annotated_files_from_map = list(annotated_annot_map.keys())
    for source_file in annotated_files:
        assert str(source_file) in annotated_files_from_map

    # test all the annots made it into the map
    annot_list_from_map = list(annotated_annot_map.values())
    for annot in annot_list:
        assert annot in annot_list_from_map

    # test all mappings are correct
    annot_class = crowsetta.formats.by_name(annot_format)
    ext = annot_class.ext
    for source_path, annot in list(annotated_annot_map.items()):
        if method == 'remove':
            assert annot.annot_path.name.replace(ext, '') == pathlib.Path(source_path).name
        elif method == 'replace':
            assert annot.annot_path.name.replace(ext, '') == pathlib.Path(source_path).stem


@pytest.mark.parametrize(
    "source_type, source_format, annot_format, method",
    [
        ("audio", "cbin", "notmat", "remove"),
        ("audio", "cbin", "simple-seq", "remove"),
        ("audio", "wav", "birdsong-recognition-dataset", None),
        ("audio", "wav", "textgrid", "replace"),
        ("spect", "mat", "yarden", None),
        ("audio", "wav", "textgrid", "replace"),
    ],
)
def test_map_annotated_to_annot(
    source_type,
    source_format,
    annot_format,
    method,
    audio_list_factory,
    spect_list_mat,
    specific_annot_list,
):
    if source_type == "audio":
        annotated_files = audio_list_factory(source_format, annot_format)
    else:
        annotated_files = spect_list_mat
    annot_list = specific_annot_list(annot_format)
    annotated_annot_map = vak.common.annotation.map_annotated_to_annot(
        annotated_files=annotated_files, annot_list=annot_list, annot_format=annot_format
    )

    # test all the audio paths made it into the map
    annotated_files_from_map = list(annotated_annot_map.keys())
    for source_file in annotated_files:
        assert str(source_file) in annotated_files_from_map

    # test all the annots made it into the map
    annot_list_from_map = list(annotated_annot_map.values())
    for annot in annot_list:
        assert annot in annot_list_from_map

    # test all mappings are correct
    if all([annot.notated_path for annot in annot_list]):
        for source_path, annot in list(annotated_annot_map.items()):
            assert vak.common.annotation.audio_filename_from_path(
                annot.notated_path
            ) == vak.common.annotation.audio_filename_from_path(source_path)
    else:
        if annot_format == 'simple-seq':
            ext = '.csv'
        else:
            annot_class = crowsetta.formats.by_name(annot_format)
            ext = annot_class.ext

        if method == 'remove':
            for source_path, annot in list(annotated_annot_map.items()):
                assert annot.annot_path.name.replace(ext, '') == pathlib.Path(source_path).name
        elif method == 'replace':
            for source_path, annot in list(annotated_annot_map.items()):
                assert annot.annot_path.name.replace(ext, '') == pathlib.Path(source_path).stem
    annot_class = crowsetta.formats.by_name(annot_format)
    ext = annot_class.ext


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
    """Test ``vak.common.annotation.has_unlabeled``,
    including edge cases as discussed in https://github.com/vocalpy/vak/issues/243
    """
    has_unlabeled = vak.common.annotation.has_unlabeled(annot, duration)
    assert has_unlabeled == expected_result


def test_has_unlabeled_annotation_with_no_segments(annotated_annot_no_segments):
    """Test edge case for `has_unlabeled`,
    see https://github.com/vocalpy/vak/issues/378
    """
    audio_path, annot_path = annotated_annot_no_segments
    data, samp_freq = evfuncs.load_cbin(audio_path)
    dur = data.shape[0] / samp_freq
    scribe = crowsetta.Transcriber(format='notmat')
    annot = scribe.from_file(annot_path).to_annot()

    assert vak.common.annotation.has_unlabeled(annot, dur) is True
