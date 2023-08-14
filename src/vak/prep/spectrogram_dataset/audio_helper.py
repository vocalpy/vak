from __future__ import annotations

import logging
import os
from pathlib import Path

import dask.bag as db
import numpy as np
from dask.diagnostics import ProgressBar

from ... import config
from ...common import constants, files
from ...common.annotation import map_annotated_to_annot
from ...common.converters import labelset_to_set
from ...config.spect_params import SpectParamsConfig
from .spect import spectrogram

logger = logging.getLogger(__name__)


def files_from_dir(audio_dir, audio_format):
    """get all audio files of a given format
    from a directory or its sub-directories,
    using the file extension associated with that annotation format.

    Parameters
    ----------
    audio_dir : str
        path to directory containing audio files.
    audio_format : str
        valid audio file format. One of {'wav', 'cbin'}.

    Returns
    -------
    audio_files : list
        of paths to audio files
    """
    if audio_format not in constants.AUDIO_FORMAT_FUNC_MAP:
        raise ValueError(f"'{audio_format}' is not a valid audio format")
    audio_files = files.from_dir(audio_dir, audio_format)
    return audio_files


def make_spectrogram_files_from_audio_files(
    audio_format: str,
    spect_params: dict | config.spect_params.SpectParamsConfig,
    output_dir: str,
    audio_dir: list | None = None,
    audio_files: list | None = None,
    annot_list: list | None = None,
    audio_annot_map: dict | None = None,
    annot_format: str | None = None,
    labelset: str | list | None = None,
    dask_bag_kwargs: dict | None = None,
):
    """Make spectrograms from audio files and save them in npz array files.

    Parameters
    ----------
    audio_format : str
        A :class:`string` representing the format of audio files.
        One of :constant:`vak.common.constants.VALID_AUDIO_FORMATS`.
    spect_params : dict or config.spect_params.SpectParamsConfig
        parameters for computing spectrogram, from .toml file.
        To see all related parameters, run:
        >>> help(vak.config.spect_params.SpectParamsConfig)
        To get a default configuration, create a SpectParamConfig
        with no arguments and then pass that to `to_spect`:
        >>> default_spect_params = vak.config.spect_params.SpectParamsConfig()
        >>> to_spect(audio_format='wav', spect_params=default_spect_params, output_dir='.')
    audio_dir : str
        Path to directory containing audio files from which to make spectrograms.
    audio_files : list
        of str, full paths to audio files from which to make spectrograms
    annot_list : list
        of annotations for array files. Default is None.
    audio_annot_map : dict
        Where keys are paths to array files and value corresponding to each key is
        the annotation for that array file.
        Default is None.
    output_dir : str
        directory in which to save .spect.npz file generated for each audio file.
    labelset : str, list
        of str or int, set of unique labels for vocalizations. Default is None.
        If not None, skip files where the associated annotations contain labels not in ``labelset``.
        ``labelset`` is converted to a Python ``set`` using ``vak.converters.labelset_to_set``.
        See help for that function for details on how to specify labelset.
    dask_bag_kwargs : dict
        Keyword arguments used when calling ``dask.bag.from_sequence``.
        E.g., ``{npartitions=20}``. Allows for finer-grained control
        when needed to process files of different sizes.

    Returns
    -------
    spect_files : list
        of str, full paths to .spect.npz files

    Notes
    -----
    For each audio file, a corresponding 'spect.npz' file will be created.
    Each '.spect.npz' file contains the following arrays:
        s : numpy.ndarray
            spectrogram, a 2-d array
        f : numpy.ndarray
            vector of centers of frequency bins from spectrogram
        t : numpy.ndarray
            vector of centers of tme bins from spectrogram
        audio_path : numpy.ndarray
            path to source audio file used to create spectrogram

    The names of the arrays are defaults, and will change if different values are specified
    in spect_params for 'spect_key', 'freqbins_key', 'timebins_key', or 'audio_path_key'.
    """
    if audio_format not in constants.VALID_AUDIO_FORMATS:
        raise ValueError(
            f"audio format must be one of '{constants.VALID_AUDIO_FORMATS}'; "
            f"format '{audio_format}' not recognized."
        )

    if all([arg is None for arg in (audio_dir, audio_files, audio_annot_map)]):
        raise ValueError(
            "must specify one of: audio_dir, audio_files, audio_annot_map"
        )

    if audio_dir and audio_files:
        raise ValueError(
            "received values for audio_dir and audio_files, unclear which to use"
        )

    if audio_dir and audio_annot_map:
        raise ValueError(
            "received values for audio_dir and audio_annot_map, unclear which to use"
        )

    if audio_files and audio_annot_map:
        raise ValueError(
            "received values for audio_files and audio_annot_map, unclear which to use"
        )

    if annot_list and audio_annot_map:
        raise ValueError(
            "received values for annot_list and array_annot_map, unclear which annotations to use"
        )

    if labelset is not None:
        labelset = labelset_to_set(labelset)

    if type(spect_params) not in [dict, SpectParamsConfig]:
        raise TypeError(
            "type of spect_params must be an instance of vak.config.spect_params.SpectParamsConfig, "
            "or a dict that can be converted to a SpectParamsConfig instance, "
            f"but was {type(spect_params)}"
        )
    if type(spect_params) is dict:
        spect_params = SpectParamsConfig(**spect_params)

    if dask_bag_kwargs is None:
        dask_bag_kwargs = {}  # so ``db.bag(**dask_bag_kwargs)`` works below

    # validate audio files if supplied by user
    if audio_files:
        # make sure audio files are all the same type, and the same as audio format specified
        exts = []
        for audio_file in audio_files:
            root, ext = os.path.splitext(audio_file)
            ext = (
                ext.lower()
            )  # make case-insensitive, e.g. treat .wav and .WAV the same
            exts.append(ext)
        uniq_ext = set(exts)
        if len(uniq_ext) > 1:
            raise ValueError(
                "audio_files should all have the same extension, "
                f"but found more than one: {uniq_ext}"
            )
        else:
            ext_str = uniq_ext.pop()
            if audio_format not in ext_str:
                raise ValueError(
                    f"audio format. '{audio_format}', does not match extensions in audio_files, '{ext_str}''"
                )

    # otherwise get audio files using audio dir (won't need to validate audio files)
    if audio_dir:
        audio_files = files_from_dir(audio_dir, audio_format)

    if annot_list:
        audio_annot_map = map_annotated_to_annot(
            audio_files, annot_list, annot_format
        )

    logger.info("creating array files with spectrograms")

    # use mapping (if generated/supplied) with labelset, if supplied, to filter
    if audio_annot_map:
        if labelset:  # then remove annotations with labels not in labelset
            # note we do this here so it happens regardless of whether
            # user supplied audio_annot_map *or* we constructed it above
            for audio_file, annot in list(audio_annot_map.items()):
                # loop in a verbose way (i.e. not a comprehension)
                # so we can give user warning when we skip files
                annot_labelset = set(annot.seq.labels)
                # below, set(labels_mapping) is a set of that dict's keys
                if not annot_labelset.issubset(set(labelset)):
                    # because there's some label in labels that's not in labelset
                    audio_annot_map.pop(audio_file)
                    extra_labels = annot_labelset - labelset
                    logger.info(
                        f"Found labels, {extra_labels}, in {Path(audio_file).name}, "
                        "that are not in labels_mapping. Skipping file.",
                    )
        audio_files = sorted(list(audio_annot_map.keys()))

    # this is defined here so all other arguments to 'to_spect' are in scope
    def _spect_file(audio_file):
        """helper function that enables parallelized creation of array
        files containing spectrograms.
        Accepts path to audio file, saves .npz file with spectrogram"""
        dat, fs = constants.AUDIO_FORMAT_FUNC_MAP[audio_format](audio_file)
        s, f, t = spectrogram(
            dat,
            fs,
            spect_params.fft_size,
            spect_params.step_size,
            spect_params.thresh,
            spect_params.transform_type,
            spect_params.freq_cutoffs,
        )
        spect_dict = {
            spect_params.spect_key: s,
            spect_params.freqbins_key: f,
            spect_params.timebins_key: t,
            # cast audio_file to str so that we don't
            # end up with a pickled pathlib.Path in .npz file
            spect_params.audio_path_key: str(audio_file),
        }
        basename = os.path.basename(audio_file)
        npz_fname = os.path.join(
            os.path.normpath(output_dir), basename + ".spect.npz"
        )
        np.savez(npz_fname, **spect_dict)
        return npz_fname

    bag = db.from_sequence(audio_files, **dask_bag_kwargs)
    with ProgressBar():
        spect_files = list(bag.map(_spect_file))
    # sort because ordering from Dask not guaranteed
    spect_files = sorted(spect_files)
    return spect_files
