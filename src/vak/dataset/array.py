from crowsetta import Transcriber


def from_array(array_dir, array_format, annot_format):
    """create .vak.dat files from already-made spectrograms that are in files containing arrays
    (e.g. a .mat file created by Matlab or a .npy file created by numpy)

    Parameters
    ----------
    array_dir : str
        path to directory of files containing spectrograms as arrays.
    array_format : str
        format of array files. One of {'mat', 'npy'}
    annot_format : str
        format of annotation.

    Returns
    -------
    vakdat_path
    """
    scribe = Transcriber()
    seqs = scribe.to_seq(file=annotation_files, format=annot_format)

    return vakdat_path
