"""
Python functions to facilitate interacting with the dataset from Koumura and
Okanoya 2016 [1].

data: https://figshare.com/articles/BirdsongRecognition/3470165

The original code was released under the GNU license:
https://github.com/cycentum/birdsong-recognition/blob/master/
birdsong-recognition/src/computation/ViterbiSequencer.java

[1] Koumura T, Okanoya K (2016) Automatic Recognition of Element Classes and
Boundaries in the Birdsong with Variable Sequences. PLoS ONE 11(7): e0159188.
doi:10.1371/journal.pone.0159188
"""

#from standard library
import os
import xml.etree.ElementTree as ET

#from dependencies
import numpy as np


class Syllable:
    """
    Object that represents a syllable.

    Properties
    ----------
    position : int
        starting sample number ("frame") within .wav file
        *** relative to start of sequence! ***
    length : int
        duration given as number of samples
    label : char
        text representation of syllable as classified by a human
        or a machine learning algorithm
    """
    def __init__(self, position, length, label):
        self.position = position
        self.length = length
        self.label = label

    def __repr__(self):
        rep_str = "Syllable labeled {} at position {} with length {}".format(
                   self.label,self.position,self.length)
        return rep_str


class Sequence:
    """
    Object that represents a sequence of syllables.

    Properties
    ----------
    wavFile : string
        file name of .wav file in which sequence occurs
    position : int
        starting sample number within .wav file
    length : int
        duration given as number of samples
    syls : list
        list of syllable objects that make up sequence
    seqSpect : spectrogram object
    """

    def __init__(self, wav_file,position, length, syl_list):
        self.wavFile = wav_file
        self.position = position
        self.length = length
        self.numSyls = len(syl_list)
        self.syls = syl_list
        self.seqSpect = None

    def __repr__(self):
        rep_str = "Sequence from {} with position {} and length {}".format(
                  self.wavFile,self.position,self.length)
        return rep_str


def parse_xml(xml_file,concat_seqs_into_songs=False):
    """
    parses Annotation.xml files from BirdsongRecognition dataset.

    Parameters
    ----------
    xml_file : string
        filename of .xml file, e.g. 'Annotation.xml'
    concat_seqs_into_songs : Boolean
        if True, concatenate sequences into songs, where each wav file is a
        song. Default is False.

    Returns
    -------
    seq_list : list of Sequence objects
    """

    tree = ET.ElementTree(file=xml_file)
    seq_list = []
    for seq in tree.iter(tag='Sequence'):
        wav_file = seq.find('WaveFileName').text
        position = int(seq.find('Position').text)
        length = int(seq.find('Length').text)
        syl_list = []
        for syl in seq.iter(tag='Note'):
            syl_position = int(syl.find('Position').text)
            syl_length = int(syl.find('Length').text)
            label = syl.find('Label').text

            syl_obj = Syllable(position = syl_position,
                               length = syl_length,
                               label = label)
            syl_list.append(syl_obj)
        seq_obj = Sequence(wav_file = wav_file,
                           position = position,
                           length = length,
                           syl_list = syl_list)
        seq_list.append(seq_obj)

    if concat_seqs_into_songs:
        song_list = []
        curr_wavFile = seq_list[0].wavFile
        new_seq_obj = seq_list[0]
        for syl in new_seq_obj.syls:
            syl.position += new_seq_obj.position

        for seq in seq_list[1:]:
            if seq.wavFile == curr_wavFile:
                new_seq_obj.length += seq.length
                new_seq_obj.numSyls += seq.numSyls
                for syl in seq.syls:
                    syl.position += seq.position
                new_seq_obj.syls += seq.syls

            else:
                song_list.append(new_seq_obj)
                curr_wavFile = seq.wavFile
                new_seq_obj = seq
                for syl in new_seq_obj.syls:
                    syl.position += new_seq_obj.position

        song_list.append(new_seq_obj)  # to append last song

        return song_list

    else:
        return seq_list


def load_song_annot(songfiles, annotation_file):
    """

    Parameters
    ----------
    songfiles: list
        of str, filenames of .wav files from BirdsongRecogntion dataset
    annotation_file: str
        path to Annotation.xml file

    Returns
    -------
    all_songfile_dict: dict
        with keys onsets, offsets, and labels
    """
    seq_list = parse_xml(annotation_file,
                         concat_seqs_into_songs=True)
    wav_files = [seq.wavFile for seq in seq_list]

    annotation_dict = {}
    for songfile in songfiles:
        songfile = os.path.basename(songfile)
        ind = wav_files.index(songfile)
        this_seq = seq_list[ind]
        onsets = np.asarray([syl.position for syl in this_seq.syls])
        offsets = np.asarray([syl.position + syl.length for syl in this_seq.syls])
        labels = [syl.label for syl in this_seq.syls]
        songfile_dict = {
            'onsets' : onsets,
            'offsets' : offsets,
            'labels' : labels
        }
        annotation_dict[songfile] = songfile_dict
    return annotation_dict


def determine_unique_labels(annotation_file):
    """given an annotation.xml file
    from a bird in BirdsongRecognition dataset,
    determine unique set of labels applied to syllables from that bird"""
    annotation = parse_xml(annotation_file,
                           concat_seqs_into_songs=True)
    lbls = [syl.label
            for seq in annotation
            for syl in seq.syls]
    unique_lbls = np.unique(lbls).tolist()
    unique_lbls = ''.join(unique_lbls)  # convert from list to string
    return unique_lbls
