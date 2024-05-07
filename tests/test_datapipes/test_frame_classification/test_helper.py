import numpy as np
import pytest

import vak.datapipes.frame_classification.helper

from ... import fixtures


@pytest.mark.parametrize(
    'subset',
    [
        'train-dur-4.0-replicate-1',
        'train-dur-4.0-replicate-2'
    ]
)
def test_sample_ids_array_filename_for_subset(subset):
    out = vak.datapipes.frame_classification.helper.sample_ids_array_filename_for_subset(subset)
    assert isinstance(out, str)
    assert out == vak.datapipes.frame_classification.constants.SAMPLE_IDS_ARRAY_FILENAME.replace(
                '.npy', f'-{subset}.npy'
            )


@pytest.mark.parametrize(
    'subset',
    [
        'train-dur-4.0-replicate-1',
        'train-dur-4.0-replicate-2'
    ]
)
def test_inds_in_sample_array_filename_for_subset(subset):
    out = vak.datapipes.frame_classification.helper.inds_in_sample_array_filename_for_subset(subset)
    assert isinstance(out, str)
    assert out == vak.datapipes.frame_classification.constants.INDS_IN_SAMPLE_ARRAY_FILENAME.replace(
                '.npy', f'-{subset}.npy'
            )


@pytest.fixture(params=fixtures.spect.SPECT_LIST_NPZ)
def frames_path(request):
    return request.param


def test_load_frames(frames_path):
    out = vak.datapipes.frame_classification.helper.load_frames(frames_path, input_type="spect")
    assert isinstance(out, np.ndarray)
