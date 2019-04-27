def from_audio(audio_dir, audio_format, annot_format):
    """create .vak.dat files from already-made spectrograms that are in files containing arrays
    (e.g. a .mat file created by Matlab or a .npy file created by numpy)

    Parameters
    ----------
    audio_dir
    audio_format
    annot_format

    Returns
    -------
    vakdat_path
    """
    return vakdat_path