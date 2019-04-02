"""validators used by attrs-based classes"""
import os


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
