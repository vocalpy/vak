import os
from glob import glob

from ..utils.data import make_spects_from_list_of_files

def from_audio(audio_dir, audio_format, annot_files, annot_format):
    """create .vak.dat files from already-made spectrograms that are in files containing arrays
    (e.g. a .mat file created by Matlab or a .npy file created by numpy)

    Parameters
    ----------
    audio_dir
    audio_format
    annot_files
    annot_format

    Returns
    -------
    vakdat_path
    """
    audio_files = glob(
        os.path.join(audio_dir, '*' + audio_format)
    )

    vakdat_path = make_spects_from_list_of_files(filelist=audio_files)

    return vakdat_path
