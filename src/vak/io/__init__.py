"""module that handles datasets:
spectrograms made from audio files of vocalizations, and associated annotations"""
from .classes import MetaSpect, Vocalization, Dataset
from . import dataframe, audio, annotation, split
from .prep import prep
