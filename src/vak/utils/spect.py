"""spectrogram utilities
filters adapted from SciPy cookbook
https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
spectrogram adapted from code by Kyle Kastner and Tim Sainburg
https://github.com/timsainb/python_spectrograms_and_inversion
"""
import os

import joblib
import numpy as np
from scipy.io import loadmat, wavfile
from scipy.signal import butter, lfilter
from matplotlib.mlab import specgram
from tqdm import tqdm

from .. import evfuncs
from ..koumura_utils import load_song_annot
from .labels import label_timebins


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def spectrogram(data, samp_freq, fft_size=512, step_size=64, thresh=None, transform_type=None):
    """creates a spectrogram

    Parameters
    ----------
    data : ndarray
        audio signal
    log_transform: bool
        if True, take the log of the spectrogram
    thresh: int
        threshold minimum power for log spectrogram

    Return
    ------
    spec : ndarray

    freqbins : ndarray

    timebins : ndarray
    """

    noverlap = fft_size - step_size

    # below only take [:3] from return of specgram because we don't need the image
    spec, freqbins, timebins = specgram(data, fft_size, samp_freq, noverlap=noverlap)[:3]

    if transform_type:
        if transform_type == 'log_spect':
            spec /= spec.max()  # volume normalize to max 1
            spec = np.log10(spec)  # take log
            if thresh:
                # I know this is weird, maintaining 'legacy' behavior
                spec[spec < -thresh] = -thresh
        elif transform_type == 'log_spect_plus_one':
            spec = np.log10(spec + 1)
            if thresh:
                spec[spec < thresh] = thresh
    else:
        if thresh:
            spec[spec < thresh] = thresh  # set anything less than the threshold as the threshold

    return spec, freqbins, timebins


# adapted from:
# https://github.com/NickleDave/hybrid-vocal-classifier/blob/master/hvc/neuralnet/utils.py
class SpectScaler:
    """class that scales spectrograms that all have the
    same number of frequency bins. Any input spectrogram
    will be scaled by subtracting off the mean of each
    frequency bin from the 'fit' set of spectrograms, and
    then dividing by the standard deviation of each
    frequency bin from the 'fit' set.
    """
    def __init__(self):
        self.columnMeans = None
        self.columnStds = None
        self.nonZeroStd = None

    def fit(self, spect):
        """fit a SpectScaler.
        Input should be spectrogram,
        oriented so that the columns are frequency bins.
        Fit function finds the mean and standard deviation of
        each frequency bin, which are used by `transform` method
        to scale other spectrograms.

        Parameters
        ----------
        spect : 2-d numpy array
            with dimensions (time bins, frequency bins)
        """
        if spect.ndim != 2:
            raise ValueError('input spectrogram should be a 2-d array')

        self.columnMeans = np.mean(spect, axis=0)
        self.columnStds = np.std(spect, axis=0)
        assert self.columnMeans.shape[-1] == spect.shape[-1]
        assert self.columnStds.shape[-1] == spect.shape[-1]
        self.nonZeroStd = np.argwhere(self.columnStds != 0)

    def _transform(self, spect):
        """transforms input spectrogram by subtracting off fit mean
        and then dividing by standard deviation
        """
        transformed = spect - self.columnMeans
        # to keep any zero stds from causing NaNs
        transformed[:, self.nonZeroStd] = (
            transformed[:, self.nonZeroStd] / self.columnStds[self.nonZeroStd])
        return transformed

    def transform(self, spects):
        """normalizes input spectrograms with fit parameters
        Assumes spectrograms are oriented with columns being frequency bins
        and rows being time bins.

        Parameters
        ----------
        spects : 2-d numpy array or list of 2-d numpy arrays
            with dimensions (time bins, frequency bins)

        """
        if any([not hasattr(self, attr) for attr in ['columnMeans',
                                                     'columnStds']]):
            raise AttributeError('SpectScaler properties are set to None,'
                                 'must call fit method first to set the'
                                 'value of these properties before calling'
                                 'transform')

        if type(spects) != np.ndarray and type(spects) != list:
            raise TypeError('type {} is not valid for spects'
                            .format(type(spects)))

        if type(spects) == np.ndarray:
            if spects.shape[-1] != self.columnMeans.shape[-1]:
                raise ValueError('number of columns in spects, {}, '
                                 'does not match shape of self.columnMeans, {},'
                                 'i.e. the number of columns from the spectrogram'
                                 'to which the scaler was fit originally')
            return self._transform(spects)

        elif type(spects) == list:
            z_norm_spects = []
            for spect in spects:
                z_norm_spects.append(self._transform(spect))

            return z_norm_spects

    def fit_transform(self, spects):
        """first calls fit and then returns normalized spects
        transformed using the fit parameters"""

        if type(spects) != np.ndarray:
            raise TypeError('spects passed to fit_transform '
                            'should be numpy array, not {}'
                            .format(type(spects)))

        if spects.ndim != 2:
            raise ValueError('ndims of spects should be 2, not {}'
                             .format(spects.ndim))

        self.fit(spects)
        return self.transform(spects)


def from_list(filelist,
              spect_params,
              output_dir,
              labels_mapping,
              skip_files_with_labels_not_in_labelset=True,
              annotation_file=None,
              n_decimals_trunc=3,
              is_for_predict=False):
    """makes spectrograms from a list of audio files

    Parameters
    ----------
    filelist : list
        of str, full paths to .wav or .cbin files
        from which to make spectrograms
    spect_params : dict
        parameters for computing spectrogram, from .ini file
    output_dir : str
        directory in which to save .spect file generated for each .cbin file,
        as described below
    labels_mapping : dict
        maps str labels to consecutive integer values {0,1,2,...N} where N
        is the number of classes / label types
    skip_files_with_labels_not_in_labelset : bool
        if True, skip .cbin files where the 'labels' array in the corresponding
        .cbin.not.mat file contains str labels not found in labels_mapping
    annotation_file : str
        full path to file with annotations. Required for .wav files. Expected to
        be a .mat file containing the variables 'keys' and 'elements', where 'keys'
        is a list of .wav filenames and 'elements' are the associated annotations
        for each filename.
        Default is None.
    n_decimals_trunc : int
        number of decimal places to keep when truncating timebin_dur
        default is 3
    is_for_predict : bool
        if True, we are making spectrograms for prediction of labels. This tells function not
        to worry about annotation and to just insert dummy vector of labeled timebins
        into .spect files. Default is False.

    Returns
    -------
    spects_used_path : str
        Full path to file called 'spect_files'
        which contains a list of three-element tuples:
            spect_filename : str, filename of `.spect` file
            spect_dur : float, duration of the spectrogram from cbin
            labels : str, labels from .cbin.not.mat associated with .cbin
                     (string labels for syllables in spectrogram)
        Used when building data sets of a specific duration.

    For each .wav or .cbin filename in the list, a '.spect' file is saved.
    Each '.spect' file contains a "pickled" Python dictionary
    with the following key, value pairs:
        spect : ndarray
            spectrogram
        freq_bins : ndarray
            vector of centers of frequency bins from spectrogram
        time_bins : ndarray
            vector of centers of tme bins from spectrogram
        labeled_timebins : ndarray
            same length as time_bins, but value of each element is a label
            corresponding to that time bin
    """
    decade = 10** n_decimals_trunc  # used below for truncating floating point

    if all(['.cbin' in filename for filename in filelist]):
        filetype = 'cbin'
    elif all(['.wav' in filename for filename in filelist]):
        filetype = 'wav'
    else:
        raise ValueError('Could not determine whether filelist is '
                         'a list of .cbin files or of .wav files. '
                         'All files must be of the same type.')

    if filetype == 'wav':
        if annotation_file is None and is_for_predict is False:
            raise ValueError('annotation_file is required when using .wav files')
        elif annotation_file and is_for_predict is True:
            if annotation_file.endswith('.mat'):
                # the complicated nested structure of the annotation.mat files
                # (a cell array of Matlab structs)
                # makes it hard for loadmat to load them into numpy arrays.
                # Some of the weird looking things below, like accessing fields and
                # then converting them to lists, are work-arounds to deal with
                # the result of loading the complicated structure.
                # Setting squeeze_me=True gets rid of some but not all of the weirdness.
                annotations = loadmat(annotation_file, squeeze_me=True)
                # 'keys' here refers to filenames, which are 'keys' for the 'elements'
                keys_key = [key for key in annotations.keys() if 'keys' in key]
                elements_key = [key for key in annotations.keys() if 'elements' in key]
                if len(keys_key) > 1:
                    raise ValueError('Found more than one `keys` in annotations.mat file')
                if len(elements_key) > 1:
                    raise ValueError('Found more than one `elements` in annotations.mat file')
                if len(keys_key) < 1:
                    raise ValueError('Did not find `keys` in annotations.mat file')
                if len(elements_key) < 1:
                    raise ValueError('Did not find `elements` in annotations.mat file')
                keys_key = keys_key[0]
                elements_key = elements_key[0]
                annot_keys = annotations[keys_key].tolist()
                annot_elements = annotations[elements_key]
            elif annotation_file.endswith('.xml'):
                annotation_dict = load_song_annot(filelist, annotation_file)
        elif annotation_file is None and is_for_predict is True:
            pass
        else:
            ValueError('make_spects_from_list_of_files received annotation file '
                       'but is_for_predict is True')

    # need to keep track of name of files used since we may skip some.
    # (cbins_used is actually a list of tuples as defined in docstring)
    spect_files = []

    pbar = tqdm(filelist)
    for filename in pbar:
        basename = os.path.basename(filename)
        if filetype == 'cbin':
            if not is_for_predict:
                try:
                    notmat_dict = evfuncs.load_notmat(filename)
                except FileNotFoundError:
                    pbar.set_description(
                        f'Did not find .not.mat file for {basename}, skipping file.'
                    )
                    continue
                this_labels_str = notmat_dict['labels']
                onsets = notmat_dict['onsets'] / 1000
                offsets = notmat_dict['offsets'] / 1000
            dat, fs = evfuncs.load_cbin(filename)

        elif filetype == 'wav':
            fs, dat = wavfile.read(filename)
            if not is_for_predict:
                if annotation_file.endswith('.mat'):
                    ind = annot_keys.index(os.path.basename(filename))
                    annotation = annot_elements[ind]
                    # The .tolist() methods calls below are to get the
                    # array out of the weird lengthless object array
                    # that scipy.io.loadmat produces when trying to load
                    # the annotation files.
                    this_labels_str = annotation['segType'].tolist()
                    onsets = annotation['segFileStartTimes'].tolist()
                    offsets = annotation['segFileEndTimes'].tolist()
                elif annotation_file.endswith('.xml'):
                    filename_key = os.path.basename(filename)
                    this_labels_str = annotation_dict[filename_key]['labels']
                    onsets = annotation_dict[filename_key]['onsets'] / fs
                    offsets = annotation_dict[filename_key]['offsets'] / fs

        if not is_for_predict:
            if skip_files_with_labels_not_in_labelset:
                labels_set = set(this_labels_str)
                # below, set(labels_mapping) is a set of that dict's keys
                if not labels_set.issubset(set(labels_mapping)):
                    # because there's some label in labels
                    # that's not in labels_mapping
                    pbar.set_description(
                        f'found labels in {basename} not in labels_mapping, skipping file'
                    )
                    continue

        pbar.set_description(f'making .spect file for {basename}')

        if 'freq_cutoffs' in spect_params:
            dat = spect_utils.butter_bandpass_filter(dat,
                                                     spect_params['freq_cutoffs'][0],
                                                     spect_params['freq_cutoffs'][1],
                                                     fs)

        # have to make a new dictionary with args to spectrogram
        # instead of e.g. using spect_params._asdict()
        # because we don't want spect_params.freq_cutoffs as an arg to spect_utils.spectrogram
        specgram_params = {'fft_size': spect_params.fft_size,
                           'step_size': spect_params.step_size}
        if 'thresh' in spect_params:
            specgram_params['thresh'] = spect_params.thresh
        if 'transform_type' in spect_params:
            specgram_params['transform_type'] = spect_params.transform_type

        spect, freq_bins, time_bins = spect_utils.spectrogram(dat, fs,
                                                              **specgram_params)

        if 'freq_cutoffs' in spect_params:
            f_inds = np.nonzero((freq_bins >= spect_params['freq_cutoffs'][0]) &
                                (freq_bins < spect_params['freq_cutoffs'][1]))[0]  # returns tuple
            spect = spect[f_inds, :]
            freq_bins = freq_bins[f_inds]

        if not is_for_predict:
            this_labels = [labels_mapping[label]
                           for label in this_labels_str]
            this_labeled_timebins = label_timebins(this_labels,
                                                   onsets,
                                                   offsets,
                                                   time_bins,
                                                   labels_mapping['silent_gap_label'])
        elif is_for_predict:
            this_labels = []
            this_labels_str = ''
            # below 1 because rows = freq bins, cols = time_bins before we take transpose (as we do in main loop)
            this_labeled_timebins = np.zeros((spect.shape[1], 1))

        if not 'timebin_dur' in locals():
            timebin_dur = np.around(np.mean(np.diff(time_bins)), decimals=3)
        else:
            curr_timebin_dur = np.around(np.mean(np.diff(time_bins)), decimals=3)
            # below truncates any decimal place past decade
            curr_timebin_dur = np.trunc(curr_timebin_dur
                                        * decade) / decade
            if not np.allclose(curr_timebin_dur, timebin_dur):
                raise ValueError("duration of timebin in file {}, {}, did not "
                                 "match duration of timebin from other .mat "
                                 "files, {}.".format(curr_timebin_dur,
                                                     filename,
                                                     timebin_dur))
        spect_dur = time_bins.shape[-1] * timebin_dur

        spect_dict = {'spect': spect,
                      'freq_bins': freq_bins,
                      'time_bins': time_bins,
                      'labels': this_labels_str,
                      'labeled_timebins': this_labeled_timebins,
                      'timebin_dur': timebin_dur,
                      'spect_params': spect_params,
                      'labels_mapping': labels_mapping}

        spect_dict_filename = os.path.join(
            os.path.normpath(output_dir),
            os.path.basename(filename) + '.spect')
        joblib.dump(spect_dict, spect_dict_filename)

        spect_files.append((spect_dict_filename, spect_dur, this_labels_str))

    spect_files_path = os.path.join(output_dir, 'spect_files')
    joblib.dump(spect_files, spect_files_path)

    return spect_files_path
