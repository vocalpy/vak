import os
from glob import glob

from scipy.io import loadmat
import joblib
import numpy as np


def convert_mat_to_spect(mat_spect_files,
                         mat_spects_annotation_file,
                         output_dir,
                         labels_mapping=None,
                         n_decimals_trunc=3):
    """converts .mat files with spectrograms to .spect files
    that are used by make_data.py script and the make_data_dicts
    function that it calls.

    Parameters
    ----------
    mat_spect_files : list
        of str, full path to .mat files
        path to directory containing .mat files
    mat_spects_annotation_file : str
        full path to annotation file containing 'keys' and 'elements'
        where 'keys' are filenames of audio files and 'elements'
        contains additional annotation not found in .mat files
    output_dir : str
        full path to directory where .spect files will be saved
    labels_mapping : dict
        maps str labels to consecutive integer values {0,1,2,...N} where N
        is the number of classes / label types.
        Default is None -- currently not implemented.
    n_decimals_trunc : int
        number of decimal places to keep when truncating timebin_dur
        default is 3

    Returns
    -------
    spect_files_path : str
        Full path to file called 'spect_files'
        which contains a list of three-element tuples:
            spect_filename : str, filename of `.spect` file
            spect_dur : float, duration of the spectrogram from cbin
            labels : str, labels from .cbin.not.mat associated with .cbin
                     (string labels for syllables in spectrogram)
        Used when building data sets of a specific duration.

    The fucntion saves a .spect file for each .mat file containing a
    spectrogram. Each .mat file with a spectrogram should contains the following
    keys:
        s : ndarray
            the spectrogram
        f : ndarray
            vector of values at bin centers for frequencies in spectrogram
        t : ndarray
            vector of values for bin centers for times in spectrogram
        labels : ndarray
            vector of same length as t where each value is a label for that time bin
    containing the spectrograms.

    If a .mat file does not contain these keys, the function skips that file.

    """
    decade = 10**n_decimals_trunc  # used below for truncating floating point

    annotations = loadmat(mat_spects_annotation_file, squeeze_me=True)
    annotations = dict(zip(annotations['keys'],
                          annotations['elements']))
    spect_files = []
    num_spect_files = len(mat_spect_files)
    for filenum, matspect_filename in enumerate(mat_spect_files):
        print('loading annotation info from {}, file {} of {}'
              .format(matspect_filename, filenum, num_spect_files))
        wav_filename = os.path.basename(matspect_filename).replace('.mat',
                                                                '.wav')
        annotation = annotations[wav_filename]
        # below does not actually create list
        # instead gets ndarray out of a zero-length ndarray of dtype=object
        labels = annotation['segType'].tolist()

        matspect = loadmat(matspect_filename, squeeze_me=True)
        if 's' not in matspect:
            print('Did not find a spectrogram in {}. '
                  'Skipping this file.'.format(matspect_filename))
            continue

        if 'freq_bins' not in locals() and 'time_bins' not in locals():
            freq_bins = matspect['f']
            time_bins = matspect['t']
            timebin_dur = np.around(np.mean(np.diff(time_bins)), decimals=3)
            # below truncates any decimal place past decade
            timebin_dur = np.trunc(timebin_dur * decade) / decade
        else:
            if not np.array_equal(matspect['f'], freq_bins):
                raise ValueError('freq_bins in {} does not freq_bins from '
                                 'other .mat files'.format(matspect_filename))
            curr_file_timebin_dur = np.around(np.mean(np.diff(matspect['t'])),
                                              decimals=3)
            # below truncates any decimal place past decade
            curr_file_timebin_dur = np.trunc(curr_file_timebin_dur
                                             * decade) / decade
            if not np.allclose(curr_file_timebin_dur, timebin_dur):
                raise ValueError('duration of timebin in file {} did not '
                                 'match duration of timebin from other .mat '
                                 'files.'.format(matspect_filename))

        # number of freq. bins should equal number of rows
        if matspect['f'].shape[-1] != matspect['s'].shape[0]:
            raise ValueError('length of freq_bins in {} does not match '
                             'number of rows in spectrogram'
                             .format(matspect_filename))
        # number of time bins should equal number of columns
        if matspect['t'].shape[-1] != matspect['s'].shape[1]:
            raise ValueError('length of time_bins in {} does not match '
                             'number of columns in spectrogram'
                             .format(matspect_filename))
        labeled_timebins = matspect['labels']
        if labeled_timebins.shape[-1] != matspect['t'].shape[-1]:
            raise ValueError("length of 'labels' (labeled timebins vector)"
                             " in {} does not match number of time bins"
                             .format(matspect_filename))
        if labeled_timebins.ndim != 2:
            if labeled_timebins.ndim == 1:
                # make same shape as output of utils.make_labeled_timebins
                # so learn_curve.py doesn't crash when concatenating
                # with zero-pad vector
                labeled_timebins = labeled_timebins[:, np.newaxis]
            else:
                raise ValueError('labeled timebins vector from {} has '
                                 'invalid number of dimensions: {}'
                                 .format(matspect_filename, labels.ndim))

        spect_dict = {'spect': matspect['s'],
                      'freq_bins': matspect['f'],
                      'time_bins': matspect['t'],
                      'labels': labels,
                      'labeled_timebins': labeled_timebins,
                      'timebin_dur': timebin_dur,
                      'spect_params': 'matlab',
                      'labels_mapping': 'matlab'}

        spect_dict_filename = os.path.join(
            os.path.normpath(output_dir),
            os.path.basename(matspect_filename) + '.spect')
        joblib.dump(spect_dict, spect_dict_filename)

        spect_dur = matspect['s'].shape[-1] * timebin_dur
        spect_files.append((matspect_filename, spect_dur, labels))

    spect_files_path = os.path.join(output_dir, 'spect_files')
    joblib.dump(spect_files, spect_files_path)

    return spect_files_path


def convert_train_keys_to_txt(train_keys_path,
                              txt_filename = 'spect_files'):
    """get train_keys cell array out of .mat file, convert to list of str, save as .txt

    Parameters
    ----------
    train_keys_path : str
        path to folder with train_keys.mat file
    txt_filename : str
        filename for .txt file that contains list of .mat filenames
        Default is `training_filenames`

    Returns
    -------
    None. Saves .txt file in train_keys_path.
    """
    train_spect_files = loadmat(train_keys_path,
                                squeeze_me=True)['train_keys'].tolist()
    txt_filename = os.path.join(os.path.split(train_keys_path)[0],
                                txt_filename)
    with open(txt_filename, 'w') as fileobj:
        fileobj.write('\n'.join(train_spect_files))