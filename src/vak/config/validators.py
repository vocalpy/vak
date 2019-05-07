"""validators used by attrs-based classes"""
import os
import pkg_resources


def is_a_directory(instance, attribute, value):
    """check if given path is a directory"""
    if not os.path.isdir(value):
        raise NotADirectoryError(
            f'Value specified for {attribute.name} of {type(instance)} not recognized as a directory:\n'
            f'{value}'
        )


def is_a_file(instance, attribute, value):
    """check if given path is a directory"""
    if not os.path.isfile(value):
        raise NotADirectoryError(
            f'Value specified for {attribute.name} of {type(instance)} not recognized as a file:\n'
            f'{value}'
        )

from ..dataset.audio import VALID_AUDIO_FORMATS


def is_audio_format(instance, attribute, value):
    """check if valid audio format"""
    if value not in VALID_AUDIO_FORMATS:
        raise ValueError(
            f'{value} is not a valid format for audio files'
        )


VALID_ANNOT_FORMATS = {
    entry_point.name for entry_point in pkg_resources.iter_entry_points('crowsetta.format')
}


def is_annot_format(instance, attribute, value):
    """check if valid annotation format"""
    if value not in VALID_ANNOT_FORMATS:
        raise ValueError(
            f'{value} is not a valid format for annotation files.\n'
            f'Valid formats are: {VALID_ANNOT_FORMATS}'
        )


VALID_SPECT_FORMATS = {'mat', 'npz'}
def is_spect_format(instance, attribute, value):
    """check if valid format for spectrograms"""
    if value not in VALID_SPECT_FORMATS:
        raise ValueError(
            f'{value} is not a valid format for spectrogram files.\n'
            f'Valid formats are: {VALID_SPECT_FORMATS}'
        )
