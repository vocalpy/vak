import os
import logging
from datetime import datetime

from crowsetta import Transcriber

from . import annot, array, audio


def prep(labelset,
         data_dir,
         annot_format,
         skip_files_with_labels_not_in_labelset=True,
         output_dir=None,
         save_vocds=False,
         vocds_fname=None,
         return_vocds=True,
         return_path=True,
         annot_file=None,
         audio_format=None,
         array_format=None,
         spect_params=None):
    """prepare a VocalizationDataset from a directory of audio or spectrogram files
    containing vocalizations, and (optionally) annotation for those files

    Parameters
    ----------
    labelset : set, list
        of str or int, set of labels for syllables
    data_dir : str
        path to directory with audio or spectrogram files from which to make dataset
    skip_files_with_labels_not_in_labelset : bool
        if True, skip a file if the labels variable contains labels not
        found in 'labelset'. Default is True.
    output_dir : str
        path to location where data sets should be saved. Default is None,
        in which case data sets is saved in data_dir.
    save_vocds : bool
        if True, save the VocalizationDataset created as a .json file.
    vocds_fname : str
        filename for VocalDataset, which will be saved as a .json file.
        If filename does not end in .json, then that extension will be appended.
        Default is None. If None, then the filename will be
        'vocalization_dataset_prepared_{timestamp}.json'.
    return_vocds : bool
        if True, return
    audio_format : str
        format of audio files. One of {'wav', 'cbin'}.
    array_format : str
        format of array files containing spectrograms as 2-d matrices.
        One of {'mat', 'npz'}.
    annot_format : str
        format of annotations. Any format that can be used with the
        crowsetta library is valid.
    annot_file : str
        Path to a single annotation file. Default is None.
        Used when a single file contains annotations for multiple audio files.
    spect_params : dict
        Dictionary of parameters for creating spectrograms.
        Default is None (implying that spectrograms are already made).

    Returns
    -------
    vocds : vak.dataset.VocalizationDataset
        the VocalizationDataset prepared from the directory specified
    vocds_path : str
        path to where VocalizationDataset was saved

    Notes
    -----
    If dataset is created from audio files, then .spect.npz files will be
    generated from the audio files and saved in output_dir.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel('INFO')

    if type(labelset) not in (set, list):
        raise TypeError(
            f"type of labelset must be set or list, but type was: {type(labelset)}"
        )

    if type(labelset) == list:
        labelset_set = set(labelset)
        if len(labelset) != len(labelset_set):
            raise ValueError(
                'labelset contains repeated elements, should be a set (i.e. all members unique.\n'
                f'Labelset was: {labelset}'
            )
        else:
            labelset = labelset_set

    if vocds_fname is not None:
        if type(vocds_fname) != str:
            raise TypeError(
                f"vocds_fname should be a string, but type was: {type(vocds_fname)}"
            )

    if output_dir:
        if not os.path.isdir(output_dir):
            raise NotADirectoryError(
                f'output_dir not found: {output_dir}'
            )
    elif output_dir is None:
        output_dir = data_dir

    if audio_format is None and array_format is None:
        raise ValueError("Must specify either audio_format or array_format")

    if audio_format and array_format:
        raise ValueError("Cannot specify both audio_format and array_format, "
                         "unclear whether to create spectrograms from audio files or "
                         "use already-generated spectrograms from array files")

    if annot_file is None:
        annot_files = annot.files_from_dir(annot_dir=data_dir,
                                           annot_format=annot_format)
        scribe = Transcriber(voc_format=annot_format)
        annot_list = scribe.to_seq(file=annot_files)
    else:
        scribe = Transcriber(voc_format=annot_format)
        annot_list = scribe.to_seq(file=annot_file)

    # ------ if making dataset from audio files, need to make into array files first! ----------------------------
    if audio_format:
        logger.info(
            f'making array files containing spectrograms from audio files in: {data_dir}'
        )
        audio_files = audio.files_from_dir(data_dir, audio_format)
        array_files = audio.to_arr_files(audio_format=audio_format,
                                         spect_params=spect_params,
                                         output_dir=output_dir,
                                         audio_files=audio_files,
                                         annot_list=annot_list,
                                         labelset=labelset,
                                         skip_files_with_labels_not_in_labelset=skip_files_with_labels_not_in_labelset
                                         )
        array_format = 'npz'
    else:
        array_files = None

    from_arr_kwargs = {
        'array_format': array_format,
        'labelset': labelset,
        'skip_files_with_labels_not_in_labelset': skip_files_with_labels_not_in_labelset,
        'load_arr': False,
        'annot_list': annot_list,
    }

    if array_files:
        from_arr_kwargs['array_files'] = array_files
        logger.info(
            f'creating VocalDataset from array files in: {output_dir}'
        )
    else:
        from_arr_kwargs['array_dir'] = data_dir
        logger.info(
            f'creating VocalDataset from array files in: {data_dir}'
        )

    vocds = array.from_arr_files(**from_arr_kwargs)

    if save_vocds:
        if vocds_fname is None:
            timenow = datetime.now().strftime('%y%m%d_%H%M%S')
            vocds_fname = f'vocalization_dataset_prepared_{timenow}.json'

        if not vocds_fname.endswith('.json'):
            vocds_fname += '.json'

        if output_dir:
            vocds_path = os.path.join(output_dir, vocds_fname)
        else:
            vocds_path = os.path.join(os.getcwd(), vocds_fname)

        vocds.save(json_fname=vocds_path)
    else:
        vocds_path = None

    if return_vocds and return_path:
        return vocds, vocds_path
    elif return_path:
        return vocds_path
    elif return_vocds:
        return vocds

