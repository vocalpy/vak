"""fixture used to test DAS model"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from .test_data import NETS_TEST_DATA_ROOT


DAS_TEST_DATA_ROOT = NETS_TEST_DATA_ROOT / "das"

DAS_BATCH_DATA_ROOT = DAS_TEST_DATA_ROOT / "batches"

AMP_TO_DB_IN_BATCHES = [
    batch
    for batch in DAS_BATCH_DATA_ROOT.iterdir()
    if 'amp-to-db-input' in str(batch)
]

AMP_TO_DB_OUT_BATCHES = [
    batch
    for batch in DAS_BATCH_DATA_ROOT.iterdir()
    if 'amp-to-db-output' in str(batch)
]

AMP_TO_DB_IN_OUT_TUPLES = list(zip(AMP_TO_DB_IN_BATCHES, AMP_TO_DB_OUT_BATCHES))


@pytest.fixture(params=AMP_TO_DB_IN_OUT_TUPLES)
def amp_to_db_in_out_tuples(request):
    amp_in_path, amp_out_path = request.param
    amp_in = np.load(amp_in_path)
    amp_out = np.load(amp_out_path)
    return amp_in, amp_out


STFT_KERNELS_ROOT = DAS_TEST_DATA_ROOT / "stft_kernels"
NPY_KERNEL_PATHS = STFT_KERNELS_ROOT.glob('*npy')


@pytest.fixture
def specific_stft_kernels_factory():

    def _specific_stft_kernels_factory(nb_pre_dft):
        assert isinstance(nb_pre_dft, int), f"nb_pre_dft should be an integer but was: {nb_pre_dft}"

        these = [
            path
            for path in NPY_KERNEL_PATHS
            if str(nb_pre_dft) in str(path)
        ]

        err = f"did not find two npy kernel paths for nb_pre_dft={nb_pre_dft}, found: {these}"
        assert len(these) == 2, err

        real_path = [path for path in these if 'real' in str(path)]
        imag_path = [path for path in these if 'imag' in str(path)]

        assert len(real_path) == 1, f'did not find single path for real kernel, found: {real}'
        real_path = real_path[0]

        assert len(imag_path) == 1, f'did not find single path for img kernel, found: {img}'
        imag_path = imag_path[0]

        real = np.load(real_path)
        imag = np.load(imag_path)

        return real, imag

    return _specific_stft_kernels_factory


INPUTS_BATCHES = [
    batch
    for batch in DAS_BATCH_DATA_ROOT.iterdir()
    if '-inputs-' in str(batch)
]

CONV2D_OUTPUT_REAL_BATCHES = [
    batch
    for batch in DAS_BATCH_DATA_ROOT.iterdir()
    if 'conv2d-output-real-' in str(batch)
]

CONV2D_OUTPUT_IMAG_BATCHES = [
    batch
    for batch in DAS_BATCH_DATA_ROOT.iterdir()
    if 'conv2d-output-imag-' in str(batch)
]

INPUTS_CONV2D_OUTPUTS_TUPLES = list(zip(
    INPUTS_BATCHES, CONV2D_OUTPUT_REAL_BATCHES, CONV2D_OUTPUT_IMAG_BATCHES)
)


@pytest.fixture(params=INPUTS_CONV2D_OUTPUTS_TUPLES)
def inputs_conv2d_outputs(request):
    inputs_path, conv2d_real_path, conv2d_imag_path = request.param
    inputs = torch.tensor(
        np.load(inputs_path)
    ).to(torch.float32)  # audio is 'half'
    conv2d_real = torch.tensor(
        np.load(conv2d_real_path)
    )
    conv2d_imag = torch.tensor(
        np.load(conv2d_imag_path)
    )
    return inputs, conv2d_real, conv2d_imag


STFT_OUT_BATCHES = [
    batch
    for batch in DAS_BATCH_DATA_ROOT.iterdir()
    if str(batch.name).startswith('stft-layer-out-')
]

INPUTS_STFT_OUT_TUPLES = list(zip(INPUTS_BATCHES, STFT_OUT_BATCHES))


@pytest.fixture(params=INPUTS_STFT_OUT_TUPLES)
def inputs_stft_out_tuples(request):
    inputs_path, stft_path = request.param
    inputs = torch.tensor(
        np.load(inputs_path)
    ).to(torch.float32)  # audio is 'half'
    stft_out = torch.tensor(
        np.load(stft_path)
    )
    return inputs, stft_out
