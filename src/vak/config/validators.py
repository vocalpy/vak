"""validators used by attrs-based classes"""
import os

def is_a_directory(instance, attribute, value):
    """check if given path is a directory"""
    if not os.path.isdir(value):
        raise NotADirectoryError(
            f'{value} specified for {attribute} of {instance}, but not recognized as a directory'
        )

def is_a_file(instance, attribute, value):
    """check if given path is a directory"""
    if not os.path.isfile(value):
        raise NotADirectoryError(
            f'{value} specified for {attribute} of {instance}, but not recognized as a file'
        )
