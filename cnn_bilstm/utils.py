from glob import glob
import random

import numpy as np

import hvc


def make_labels_mapping(data_dir):
    """make mapping
    from the set of unique string labels: [i,a,b,c,d,h,j,k]
    to a sequence of integers: [0,1,2,3,...
    for converting labels into integers
    that can then be converted to one-hot vectors
    for training outputs of a neural network"""
    notmats = glob(data_dir + '*.not.mat')
    labels = []
    for notmat in notmats:
        notmat_dict = hvc.evfuncs.load_notmat(notmat)
        label_arr = np.asarray([ord(label)
                                for label in notmat_dict['labels']]
                               )
        labels.append(label_arr)
    labels = np.concatenate(labels)
    uniq_labels = np.unique(labels)
    labels_to_map_to = range(1, uniq_labels.shape[-1] + 1)
    # skip 0 so 0 can be used as label for 'silent gap' across training/testing data
    return dict(zip(uniq_labels, labels_to_map_to))


def make_labeled_timebins_vector(labels,
                                 onsets,
                                 offsets,
                                 time_bins,
                                 silent_gap_label=0):
    """makes a vector of labels for each timebin from a spectrogram,
    given labels for syllables plus onsets and offsets of syllables

    Parameters
    ----------
    labels : ints
        should be mapping returned by make_labels_mapping
    onsets : ndarray
        1d vector of floats, syllable onsets in seconds
    offsets : ndarray
        1d vector of floats, offsets in seconds
    time_bins : ndarray
        1d vector of floats,
        time in seconds for each time bin of a spectrogram
    silent_gap_label : int
        label assigned to silent gaps
        default is 0

    Returns
    -------

    """

    labels = [int(label) for label in labels]
    label_vec = np.ones((time_bins.shape[-1], 1), dtype='int8') * silent_gap_label
    onset_inds = [np.argmin(np.abs(time_bins - onset))
                  for onset in onsets]
    offset_inds = [np.argmin(np.abs(time_bins - offset))
                   for offset in offsets]
    for label, onset, offset in zip(labels, onset_inds, offset_inds):
        label_vec[onset:offset+1] = label
        # offset_inds[ind]+1 because of Matlab one-indexing
    return label_vec


def load_data(labelset, data_dir, number_files):
    """

    Parameters
    ----------
    labelset : list
        all labels to consider in song
        e.g., 'iabcdefghjk'
    data_dir : str
        directory of data
    number_files : int
        number of files in list of song files to process
        assumes files is cbins

    Returns
    -------
    spects : list
        of 2-d ndarrays, spectrograms
    labels : list
        of strings, labels corresponding to each spectrogram
    timebin_dur : float
        duration of a timebin in seconds from spectrograms
        estimated from last spectrogram processed
    """

    spect_maker = hvc.audiofileIO.Spectrogram(nperseg=1024,
                                              noverlap=992,
                                              freq_cutoffs=[500, 10000])
    labels_mapping = make_labels_mapping(data_dir)
    cbins = glob(data_dir + '*.cbin')
    song_spects = []
    all_labels = []
    for cbin in cbins[:number_files]:
        dat, fs = hvc.evfuncs.load_cbin(cbin)
        spect, freqbins, timebins = spect_maker.make(dat, fs)
        song_spects.append(spect.T)
        notmat_dict = hvc.evfuncs.load_notmat(cbin)
        labels = [labels_mapping[ord(label)]
                  for label in notmat_dict['labels']]
        labels = make_labeled_timebins_vector(labels,
                                              notmat_dict['onsets']/1000,
                                              notmat_dict['offsets']/1000,
                                              timebins)
        all_labels.append(labels)
    timebin_dur = np.around(np.mean(np.diff(timebins)), decimals=3)
    return song_spects, all_labels, timebin_dur


def get_inds_for_dur(song_durations,
                     target_duration,
                     timebin_dur_in_s=0.001):
    """for getting a training set with random songs but constant duration
    draws songs at random and adds to list
    until total duration of all songs => target_duration
    then truncates at target duration

    Parameters
    ----------
    song_durations : list
        list of song durations,
        where duration is number of rows in a spectrogram
        e.g.,
        [song_spect.shape[0] for song_spect in song_spectrograms]
        (rows are time instead of frequency,
        because network is set up with input this way)
    target_duration : float
        target duration of training set in s
    timebind_dur_in_s : float
        duration of each timebin, i.e. each column in spectrogram,
        in seconds.
        default is 0.001 s (1 ms)

    Returns
    -------
    inds_to_use : bool
        numpy boolean vector, True where row in X_train gets used
        (assumes X_train is one long spectrogram, consisting of all
        training spectrograms concatenated, and each row being one timebin)
    """

    for song_ind, song_duration in enumerate(song_durations):
        inds = np.ones((song_duration,), dtype=int) * song_ind
        if 'song_inds_arr' in locals():
            song_inds_arr = np.concatenate((song_inds_arr, inds))
        else:
            song_inds_arr = inds

    song_id_list = []
    total_dur_in_timebins = 0
    num_songs = len(song_durations)
    while 1:
        song_id = random.randrange(num_songs)
        if song_id in song_id_list:
            continue
        else:
            song_id_list.append(song_id)
            song_id_inds = np.where(song_inds_arr == song_id)[0]  # 0 because where np.returns tuple
            if 'inds_to_use' in locals():
                inds_to_use = np.concatenate((inds_to_use, song_id_inds))
            else:
                inds_to_use = song_id_inds
            total_dur_in_timebins = total_dur_in_timebins + song_durations[song_id]
            if total_dur_in_timebins * timebin_dur_in_s >= target_duration:
                if total_dur_in_timebins * timebin_dur_in_s > target_duration:
                    correct_length = np.round(target_duration / timebin_dur_in_s).astype(int)
                    inds_to_use = inds_to_use[:correct_length]
                break

    return inds_to_use


def reshape_data_for_batching(X, Y, batch_size, time_steps, input_vec_size):
    """reshape to feed to network in batches"""
    # need to loop through train data in chunks, can't fit on GPU all at once
    # First zero pad
    num_batches = X.shape[0] // batch_size // time_steps
    rows_to_append = ((num_batches + 1) * time_steps * batch_size) - X.shape[0]
    X = np.concatenate((X, np.zeros((rows_to_append, input_vec_size))),
                       axis=0)
    Y = np.concatenate((Y, np.zeros((rows_to_append, 1), dtype=int)), axis=0)
    num_batches = num_batches + 1
    X = X.reshape((batch_size, num_batches * time_steps, -1))
    Y = Y.reshape((batch_size, -1))
    return X, Y, num_batches