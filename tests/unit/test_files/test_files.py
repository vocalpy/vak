import pytest

import vak.files


@pytest.mark.parametrize(
    ('dir_path', 'ext'),
    [('./tests/test_data/source/audio_wav_annot_textgrid/AGBk/', 'WAV'),
     ('./tests/test_data/source/audio_wav_annot_koumura/Bird0/Wave', 'wav'),
     ]
)
def test_from_dir_is_case_insensitive(dir_path, ext):
    files = vak.files.from_dir(dir_path, ext)
    assert len(files) > 0
    assert all(
        [str(file).endswith(ext) for file in files]
    )


@pytest.mark.parametrize(
    ('dir_path', 'ext'),
    [('./tests/test_data/source/audio_wav_annot_textgrid/', 'WAV'),
     ('./tests/test_data/source/audio_wav_annot_koumura/Bird0', 'wav'),
     ]
)
def test_from_dir_searches_child_dir(dir_path, ext):
    files = vak.files.from_dir(dir_path, ext)
    assert len(files) > 0
    assert all(
        [str(file).endswith(ext) for file in files]
    )
