import os
from glob import glob

from tqdm import tqdm
from crowsetta import Transcriber

from ..utils.spect import from_list
from ..evfuncs import load_cbin
from scipy.io import wavfile

AUDIO_FORMAT_FUNC_MAP = {
    'cbin': load_cbin,
    'wav': wavfile.read
}

VALID_AUDIO_FORMATS = list(AUDIO_FORMAT_FUNC_MAP.keys())


def _get_audio_files(audio_format, data_dir):
    """helper function to fetch all audio files of a given format
    from a directory and its sub-directories

    Parameters
    ----------
    audio_format : str
    data_dir : str
        path to directory

    Returns
    -------
    audio_files : list
        of paths to audio files
    """
    if audio_format not in VALID_AUDIO_FORMATS:
        raise ValueError(f"'{audio_format}' is not a valid audio format")
    wildcard_with_extension = f'*.{audio_format}'
    audio_files = glob(os.path.join(data_dir, wildcard_with_extension))
    if len(audio_files) == 0:
        # if we don't any audio files, look in sub-directories
        audio_files = []
        subdirs = glob(os.path.join(data_dir, '*/'))
        for subdir in subdirs:
            audio_files.extend(glob(os.path.join(data_dir,
                                                 subdir,
                                                 wildcard_with_extension)))
    if len(audio_files) == 0:
        raise FileNotFoundError(
            f'No audio files with format {audio_format} found in '
            f'{data_dir} or immediate sub-directories'
        )

    return audio_files


def from_dir(audio_dir,
             audio_format,
             annot_files,
             annot_format,
             spect_params,
             labels_mapping,
             output_dir=None,
             skip_files_with_labels_not_in_labelset=True,
             annotation_file=None,
             n_decimals_trunc=3,
             is_for_predict=False
             ):
    """create .vak.dat files from already-made spectrograms that are in files containing arrays
    (e.g. a .mat file created by Matlab or a .npy file created by numpy)

    Parameters
    ----------
    audio_dir : str
        path to directory containing audio files
    audio_format : str
        format of audio files. One of {'wav', 'cbin'}
    annot_format : str
        format of annotations
    annot_files : str, list


    Returns
    -------
    vakdat_path
    """
    audio_files = glob(
        os.path.join(audio_dir, '*' + audio_format)
    )

    scribe = Transcriber()
    seq = Transcriber.to_seq(file=annot_files, file_format=annot_format)

    # need to keep track of name of files used since we may skip some.
    # (cbins_used is actually a list of tuples as defined in docstring)
    spect_files = []

    pbar = tqdm(audio_files)
    for filename in pbar:
        basename = os.path.basename(filename)


    vakdat_path = from_list(filelist=audio_files)

    return vakdat_path
