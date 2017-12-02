"""load spectrograms and labeled timebin vectors generated in matlab

takes as a command line argument the name of the directory with the .mat files

Each .mat file contains:
    s : ndarray
        the spectrogram
    f : ndarray
        vector of values at bin centers for frequencies in spectrogram
    t : ndarray
        vector of values for bin centers for times in spectrogram
    labels : ndarray
        vector of same length as t where each value is a label for that time bin
containing the spectrograms.

loads spectrograms in the .mat files specified in train_keys
"""

import sys
import os
from glob import glob

from scipy.io import loadmat
import joblib
import numpy as np

data_dir = sys.argv[1]
if not os.path.isdir(data_dir):
    raise ValueError(f'{data_dir} is not recognized as a directory')
else:
    os.chdir(data_dir)

spect_files = glob('*.mat')

spects = []
spect_files_used = []
all_time_bins = []
labeled_timebins = []

for counter, spect_file in enumerate(train_spect_files):
    print(f'loading {spect_file}')
    mat_dict = loadmat(spect_file, squeeze_me=True)

    if spect_file == 'train_keys.mat':
        continue

    if 's' not in mat_dict:
        print(f'Did not find a spectrogram in {spect_file}. Skipping this file.')
        continue

    if 'freq_bins' not in locals() and 'time_bins' not in locals():
        freq_bins = mat_dict['f']        
        time_bins = mat_dict['t']
    else:
        assert np.array_equal(mat_dict['f'], freq_bins)

    spect = mat_dict['s']
    labels = mat_dict['labels']
    # number of freq. bins should equaL number of rows
    assert mat_dict['f'].shape[-1] == spect.shape[0]
    # number of time bins should equal number of columns
    assert mat_dict['t'].shape[-1] == spect.shape[1]
    spects.append(spect)
    all_time_bins.append(time_bins)

    assert labels.shape[-1] == mat_dict['t'].shape[-1]
    labeled_timebins.append(labels)

    spect_files_used.append(spect)

data_dict = {'spects': spects,
             'filenames': spect_files_used,
             'freq_bins': freq_bins,
             'time_bins': all_time_bins,
             'labeled_timebins': labeled_timebins}

print(f'saving data dictionary in {data_dir}')
joblib.dump(data_dict, 'data_dict')

# # get train_keys cell array out of .mat file and convert to list of str
# train_spect_files = loadmat('train_keys.mat',squeeze_me=True)['train_keys'].tolist()
