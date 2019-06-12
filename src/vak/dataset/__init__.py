"""module that handles datasets:
spectrograms made from audio files of vocalizations, and associated annotations"""
from .classes import MetaSpect, Vocalization, VocalizationDataset
from . import spect, audio, annotation, split
from .prep import prep
