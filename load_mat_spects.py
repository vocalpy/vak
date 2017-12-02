"""load spectrograms generated in matlab
into a numpy array and save for use as a training set

takes as a command line argument the name of the directory with the .mat files
containing the spectrograms and the trainkeys.mat file that specifies which of
them to use for the training set.

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

# get train_keys cell array out of .mat file and convert to list of str
train_spect_files = loadmat('train_keys.mat',squeeze_me=True)['train_keys'].tolist()

spects = []
labeled_timebins = []

for counter, train_spect_file in enumerate(train_spect_files):
    print(f'loading {train_spect_file}')
    mat_dict = loadmat(train_spect_file, squeeze_me=True)
    if counter == 0:
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

    assert labels.shape[-1] == mat_dict['t'].shape[-1]
    labeled_timebins.append(labels)

training_spects = {'spects': spects,
                   'freq_bins': freq_bins,
                   'time_bins': time_bins}
print(f'saving training_spects dicitonary in {data_dir}')
joblib.dump(training_spects, 'training_spects')
print(f'saving labeled_timebins in {data_dir}')
joblib.dump(labeled_timebins, 'labeled_timebins')
