"""
evfuncs
Python implementations of functions used with EvTAF and evsonganaly.m
adapated from hybrid-vocal-classifier
https://github.com/NickleDave/hybrid-vocal-classifier
under BSD license
https://github.com/NickleDave/hybrid-vocal-classifier/blob/master/LICENSE
"""

import os

import numpy as np
import scipy
from scipy.io import loadmat


def readrecf(filename):
    """
    reads .rec files output by EvTAF
    """

    rec_dict = {}
    with open(filename, 'r') as recfile:
        line_tmp = ""
        while 1:
            if line_tmp == "":
                line = recfile.readline()
            else:
                line = line_tmp
                line_tmp = ""

            if line == "":  # if End Of File
                break
            elif line == "\n":  # if blank line
                continue
            elif "Catch" in line:
                ind = line.find('=')
                rec_dict['iscatch'] = line[ind + 1:]
            elif "Chans" in line:
                ind = line.find('=')
                rec_dict['num_channels'] = int(line[ind + 1:])
            elif "ADFREQ" in line:
                ind = line.find('=')
                try:
                    rec_dict['sample_freq'] = int(line[ind + 1:])
                except ValueError:
                    rec_dict['sample_freq'] = float(line[ind + 1:])
            elif "Samples" in line:
                ind = line.find('=')
                rec_dict['num_samples'] = int(line[ind + 1:])
            elif "T After" in line:
                ind = line.find('=')
                rec_dict['time_after'] = float(line[ind + 1:])
            elif "T Before" in line:
                ind = line.find('=')
                rec_dict['time before'] = float(line[ind + 1:])
            elif "Output Sound File" in line:
                ind = line.find('=')
                rec_dict['outfile'] = line[ind + 1:]
            elif "Thresholds" in line:
                th_list = []
                while 1:
                    line = recfile.readline()
                    if line == "":
                        break
                    try:
                        th_list.append(float(line))
                    except ValueError:  # because we reached next section
                        line_tmp = line
                        break
                rec_dict['thresholds'] = th_list
                if line == "":
                    break
            elif "Feedback information" in line:
                fb_dict = {}
                while 1:
                    line = recfile.readline()
                    if line == "":
                        break
                    elif line == "\n":
                        continue
                    ind = line.find("msec")
                    time = float(line[:ind - 1])
                    ind = line.find(":")
                    fb_type = line[ind + 2:]
                    fb_dict[time] = fb_type
                rec_dict['feedback_info'] = fb_dict
                if line == "":
                    break
            elif "File created" in line:
                header = [line]
                for counter in range(4):
                    line = recfile.readline()
                    header.append(line)
                rec_dict['header'] = header
    return rec_dict


def load_cbin(filename, channel=0):
    """
    loads .cbin files output by EvTAF.

    arguments
    ---------
    filename : string

    channel : integer
        default is 0

    returns
    -------
    data : numpy array
        1-d vector of 16-bit signed integers

    sample_freq : integer
        sampling frequency in Hz. Typically 32000.
    """

    # .cbin files are big endian, 16 bit signed int, hence dtype=">i2" below
    data = np.fromfile(filename, dtype=">i2")
    recfile = filename[:-5] + '.rec'
    rec_dict = readrecf(recfile)
    data = data[channel::rec_dict['num_channels']]  # step by number of channels
    sample_freq = rec_dict['sample_freq']
    return data, sample_freq


def load_notmat(filename):
    """
    loads .not.mat files created by evsonganaly.m.
    wrapper around scipy.io.loadmat.
    Calls loadmat with squeeze_me=True to remove extra dimensions from arrays
    that loadmat parser sometimes adds.

    Argument
    --------
    filename : string, name of .not.mat file

    Returns
    -------
    notmat_dict : dictionary of variables from .not.mat files
    """

    if ".not.mat" in filename:
        pass
    elif filename[-4:] == "cbin":
        filename += ".not.mat"
    else:
        raise ValueError("Filename should have extension .cbin.not.mat or"
                         " .cbin")

    if not os.path.isfile(filename):
        raise FileNotFoundError
    else:
        return loadmat(filename, squeeze_me=True)


def bandpass_filtfilt(rawsong, samp_freq, freq_cutoffs=None):
    """filter song audio with band pass filter, run through filtfilt
    (zero-phase filter)

    Parameters
    ----------
    rawsong : ndarray
        audio
    samp_freq : int
        sampling frequency
    freq_cutoffs : list
        2 elements long, cutoff frequencies for bandpass filter
        if None, set to [500, 10000]. Default is None.

    Returns
    -------
    filtsong : ndarray
    """

    Nyquist_rate = samp_freq / 2
    if freq_cutoffs is None:
        freq_cutoffs = [500, 10000]
    if rawsong.shape[-1] < 387:
        numtaps = 64
    elif rawsong.shape[-1] < 771:
        numtaps = 128
    elif rawsong.shape[-1] < 1539:
        numtaps = 256
    else:
        numtaps = 512

    cutoffs = np.asarray([freq_cutoffs[0] / Nyquist_rate,
                          freq_cutoffs[1] / Nyquist_rate])
    # code on which this is based, bandpass_filtfilt.m, says it uses Hann(ing)
    # window to design filter, but default for matlab's fir1
    # is actually Hamming
    # note that first parameter for scipy.signal.firwin is filter *length*
    # whereas argument to matlab's fir1 is filter *order*
    # for linear FIR, filter length is filter order + 1
    b = scipy.signal.firwin(numtaps + 1, cutoffs, pass_zero=False)
    a = np.zeros((numtaps+1,))
    a[0] = 1  # make an "all-zero filter"
    padlen = np.max((b.shape[-1] - 1, a.shape[-1] - 1))
    filtsong = scipy.signal.filtfilt(b, a, rawsong, padlen=padlen)
    return filtsong


def smooth_data(rawsong, samp_freq, freq_cutoffs=None, smooth_win=2):
    """filter raw audio and smooth signal
    used to calculate amplitude.

    Parameters
    ----------
    rawsong : 1-d numpy array
        "raw" voltage waveform from microphone
    samp_freq : int
        sampling frequency
    freq_cutoffs: list
        two-element list of integers, [low freq., high freq.]
        bandpass filter applied with this list defining pass band.
        Default is None, in which case bandpass filter is not applied.
    smooth_win : integer
        size of smoothing window in milliseconds. Default is 2.

    Returns
    -------
    smooth : 1-d numpy array
        smoothed waveform

    Applies a bandpass filter with the frequency cutoffs in spect_params,
    then rectifies the signal by squaring, and lastly smooths by taking
    the average within a window of size sm_win.
    This is a very literal translation from the Matlab function SmoothData.m
    by Evren Tumer. Uses the Thomas-Santana algorithm.
    """

    if freq_cutoffs is None:
        # then don't do bandpass_filtfilt
        filtsong = rawsong
    else:
        filtsong = bandpass_filtfilt(rawsong, samp_freq, freq_cutoffs)

    squared_song = np.power(filtsong, 2)
    len = np.round(samp_freq * smooth_win / 1000).astype(int)
    h = np.ones((len,)) / len
    smooth = np.convolve(squared_song, h)
    offset = round((smooth.shape[-1] - filtsong.shape[-1]) / 2)
    smooth = smooth[offset:filtsong.shape[-1] + offset]
    return smooth