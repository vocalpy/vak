#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='cnn-bilstm',
    version='0.1a',
    description=('hybrid convolutional-recurrent neural networks for '
                'segmentation of birdsong and classification of elements'),
    author='Yarden Cohen, David Nicholson',
    author_email='https://github.com/yardencsGitHub/tf_syllable_segmentation_annotation/issues',
    url='https://github.com/yardencsGitHub/tf_syllable_segmentation_annotation/',
    packages=find_packages(),
    license='BSD',
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3.5',
    ]
    )


