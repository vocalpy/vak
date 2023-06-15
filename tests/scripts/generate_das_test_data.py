#!/usr/bin/env python
# coding: utf-8
"""
Script that generates the "source" test data
used to test the implementation of the DAS model.

This script should be run in an environment
created by the 'das-test-data-env' files
in ``./tests/scripts/``.
"""
import pathlib
import urllib.request

import das.data
import das.io
import das.tcn  # need to make sure this is loaded before loading model
import das.utils
import numpy as np
from tensorflow.keras import backend as K


HERE = pathlib.Path(
    '/home/pimienta/Documents/repos/coding/bioacoustics/das-experiments/src/scripts'
)
TEST_DATA_ROOT = pathlib.Path(
    "/home/pimienta/Documents/repos/coding/vocalpy/vak-vocalpy"
) / "tests/data_for_tests"
SOURCE_TEST_DATA = TEST_DATA_ROOT / "source"
NETS_TEST_DATA = SOURCE_TEST_DATA / "nets"
DAS_TEST_DATA = NETS_TEST_DATA / "das"
STFT_KERNELS_ROOT = DAS_TEST_DATA / "stft_kernels"
DAS_BATCH_DATA_ROOT = DAS_TEST_DATA / "batches"


DAS_MENAGERIE_ROOT = DAS_TEST_DATA / 'das-menagerie'

BF_MODEL_URL = 'https://github.com/janclemenslab/das-menagerie/releases/download/bengalese_finch_v1.0/fourmales_model.h5'
BF_MODEL_PATH = DAS_MENAGERIE_ROOT / 'bengalese_finch_v1.0/fourmales_model.h5'

if not BF_MODEL_PATH.exists():
    urllib.request.urlretrieve(BF_MODEL_URL, BF_MODEL_PATH)

BF_PARAMS_URL = 'https://github.com/janclemenslab/das-menagerie/releases/download/bengalese_finch_v1.0/fourmales_params.yaml'
BF_PARAMS_PATH = DAS_MENAGERIE_ROOT / 'bengalese_finch_v1.0/fourmales_params.yaml'

if not BF_PARAMS_PATH.exists():
    urllib.request.urlretrieve(BF_PARAMS_URL, BF_PARAMS_PATH)


# TODO: I actually made this dataset, see make-datasets.ipynb -- need to be able to download!
# I think because I couldn't load directly?
data_dir='/home/pimienta/Documents/data/vocal/DAS-datasets/bengalese-finch-train-test/dataset.npy'
x_suffix: str = ""
y_suffix: str = ""
dataset = das.io.load(data_dir, x_suffix=x_suffix, y_suffix=y_suffix)


params = das.utils.load_params(
    str(BF_PARAMS_PATH.parent) + '/' + BF_PARAMS_PATH.stem.replace('_params', '')
)


model = das.utils.load_model(str(BF_MODEL_PATH.parent) + '/' + BF_MODEL_PATH.stem.replace('_model', ''),
                             model_dict={'tcn_stft': das.models.model_dict['tcn_stft']})


data_gen = das.data.AudioSequence(
    dataset["train"]["x"],
    dataset["train"]["y"],
    shuffle=False,
    shuffle_subset=None,
    first_sample=0,
    last_sample=None,
    nb_repeats=1,
    batch_processor=None,
    # TODO: which params do I need to get right output shape?
    **params,  # batch size, other parameters that apparently give us correct shape
)


N_BATCHES = 4

stem = 'bengalese-finch-train-test-dataset.npy'

for batch_ind, batch in enumerate(data_gen):
    if batch_ind > N_BATCHES - 1:
        break
    print(
        f'saving batch: {batch_ind}'
    )
    inputs, targets, masks = batch

    print(
        f'inputs shape: {inputs.shape}, targets shape: {targets.shape}'
    )
    
    for var, val in zip(('inputs', 'targets', 'masks'), (inputs, targets, masks)):
        path = DAS_BATCH_DATA_ROOT / f'{stem}-{var}-{batch_ind}'
        print(f'saving: {path}')
        np.save(path, val)
    
    out_0 = model.layers[0](inputs)

    input_conv2d = out_0[:, :, 0:1]
    input_conv2d = K.expand_dims(input_conv2d, 3)  # add a dummy dimension (channel axis)
    subsample = (model.layers[1].n_hop, 1)
    conv2d_output_real = K.conv2d(input_conv2d, model.layers[1].dft_real_kernels, 
                                  strides=subsample, padding=model.layers[1].padding, 
                                  data_format="channels_last")
    conv2d_output_imag = K.conv2d(input_conv2d, model.layers[1].dft_imag_kernels, 
                                  strides=subsample, padding=model.layers[1].padding, 
                                  data_format="channels_last")
    real_path = DAS_BATCH_DATA_ROOT / f'conv2d-output-real-{batch_ind}'
    imag_path = DAS_BATCH_DATA_ROOT / f'conv2d-output-imag-{batch_ind}'
    np.save(real_path, conv2d_output_real)
    np.save(imag_path, conv2d_output_imag)
    
    spect_mono_out = model.layers[1]._spectrogram_mono(out_0)
    spect_mono_out_path = DAS_BATCH_DATA_ROOT / f'spect-mono-out-{batch_ind}'
    np.save(spect_mono_out_path, spect_mono_out.numpy())

    # FIXME: we recapitulate call method here
    if model.layers[1].power_spectrogram != 2.0:
        amp_to_db_input = K.pow(K.sqrt(spect_mono_out), model.layers[1].power_spectrogram)
    else:
        amp_to_db_input = spect_mono_out
    amb_to_db_input_path = DAS_BATCH_DATA_ROOT / f'amp-to-db-input-{batch_ind}'
    np.save(amb_to_db_input_path, amp_to_db_input.numpy())

    if model.layers[1].return_decibel_spectrogram:
        amp_to_db_output = das.kapre.backend_keras.amplitude_to_decibel(amp_to_db_input)
        amb_to_db_output_path = DAS_BATCH_DATA_ROOT / f'amp-to-db-output-{batch_ind}'
        np.save(amb_to_db_output_path, amp_to_db_output.numpy())
    else:
        amp_to_db_output = None
    
    if amp_to_db_output is not None:
        expected_stft_out = amp_to_db_output
    else:
        expected_stft_out = amp_to_db_input
    
    stft_out = model.layers[1](out_0)
    stft_out_path = DAS_BATCH_DATA_ROOT / f'stft-layer-out-{batch_ind}'
    np.save(stft_out_path, stft_out.numpy())
    
    np.testing.assert_allclose(
        stft_out.numpy(),
        expected_stft_out.numpy()
    )
