import os
from pathlib import Path
import logging
from datetime import datetime

from crowsetta import Transcriber

from . import annotation, audio, spect
from .annotation import source_annot_map


def prep(data_dir,
         annot_format=None,
         labelset=None,
         output_dir=None,
         save_vds=False,
         vds_fname=None,
         return_vds=True,
         return_path=True,
         load_spects=True,
         annot_file=None,
         audio_format=None,
         spect_format=None,
         spect_params=None,
         spect_output_dir=None):
    """prepare a VocalizationDataset from a directory of audio or spectrogram files
    containing vocalizations, and (optionally) annotation for those files

    Parameters
    ----------
    data_dir : str
        path to directory with audio or spectrogram files from which to make dataset
    annot_format : str
        format of annotations. Any format that can be used with the
        crowsetta library is valid. Default is None.
    labelset : set, list
        of str or int, set of labels for vocalizations. Default is None.
        If not None, then files will be skipped where the 'labels' array in the
        corresponding annotation contains labels that are not found in labelset
    output_dir : str
        path to location where data sets should be saved. Default is None,
        in which case data sets is saved in data_dir.
    save_vds : bool
        if True, save the VocalizationDataset created as a .json file. Default is False.
    vds_fname : str
        filename for VocalDataset, which will be saved as a .json file.
        If filename does not end in .json, then that extension will be appended.
        Default is None. If None, then the filename will be
        'prep_{timestamp}.vds.json'.
    return_vds : bool
        if True, return prepared VocalizationDataset. Default is True.
    return_path : bool
        if True, return path to saved VocalizationDataset. Default is True.
    load_spects : bool
        if True, load spectrograms. If False, return a VocalDataset without spectograms loaded.
        Default is True. Set to False when you want to create a VocalDataset for use
        later, but don't want to load all the spectrograms into memory yet.
    audio_format : str
        format of audio files. One of {'wav', 'cbin'}.
    spect_format : str
        format of array files containing spectrograms as 2-d matrices.
        One of {'mat', 'npz'}.
    annot_file : str
        Path to a single annotation file. Default is None.
        Used when a single file contains annotations for multiple audio files.
    spect_params : dict
        Dictionary of parameters for creating spectrograms.
        Default is None (implying that spectrograms are already made).
    spect_output_dir : str
        path to location where spectrogram files should be saved. Default is None,
        in which case it defaults to 'spectrograms_generated_{time stamp}'.

    Returns
    -------
    vds : vak.dataset.VocalizationDataset
        the VocalizationDataset prepared from the directory specified
    vds_path : str
        path to where VocalizationDataset was saved

    Notes
    -----
    If dataset is created from audio files, then .spect.npz files will be
    generated from the audio files and saved in output_dir.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel('INFO')

    if labelset is not None:
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

    if vds_fname is not None:
        if type(vds_fname) != str:
            raise TypeError(
                f"vds_fname should be a string, but type was: {type(vds_fname)}"
            )

    if output_dir:
        if not os.path.isdir(output_dir):
            raise NotADirectoryError(
                f'output_dir not found: {output_dir}'
            )
    elif output_dir is None:
        output_dir = data_dir

    if audio_format is None and spect_format is None:
        raise ValueError("Must specify either audio_format or spect_format")

    if audio_format and spect_format:
        raise ValueError("Cannot specify both audio_format and spect_format, "
                         "unclear whether to create spectrograms from audio files or "
                         "use already-generated spectrograms from array files")

    if spect_output_dir:
        if not os.path.isdir(spect_output_dir):
            raise NotADirectoryError(
                f'spect_output_dir not found: {spect_output_dir}'
            )

    if annot_format is not None:
        if annot_file is None:
            annot_files = annotation.files_from_dir(annot_dir=data_dir,
                                                    annot_format=annot_format)
            scribe = Transcriber(voc_format=annot_format)
            annot_list = scribe.to_seq(file=annot_files)
        else:
            scribe = Transcriber(voc_format=annot_format)
            annot_list = scribe.to_seq(file=annot_file)
    else:  # if annot_format not specified
        annot_list = None

    # ------ if making dataset from audio files, need to make into array files first! ----------------------------
    if audio_format:
        logger.info(
            f'making array files containing spectrograms from audio files in: {data_dir}'
        )
        audio_files = audio.files_from_dir(data_dir, audio_format)
        if annot_list:
            audio_annot_map = source_annot_map(audio_files, annot_list)
            if labelset:  # then remove annotations with labels not in labelset
                # do this here instead of inside function call so that items get removed
                # from annot_list here and won't cause an error because they're still
                # in this list when we call spect.from_files
                for audio_file, annot in list(audio_annot_map.items()):
                    # loop in a verbose way (i.e. not a comprehension)
                    # so we can give user warning when we skip files
                    annot_labelset = set(annot.labels)
                    # below, set(labels_mapping) is a set of that dict's keys
                    if not annot_labelset.issubset(set(labelset)):
                        # because there's some label in labels that's not in labelset
                        audio_annot_map.pop(audio_file)
                        logger.info(
                            f'found labels in {annot.file} not in labels_mapping, '
                            f'skipping audio file: {audio_file}'
                        )
                audio_files = []
                annot_list = []
                for k,v in audio_annot_map.items():
                    audio_files.append(k)
                    annot_list.append(v)

        timenow = datetime.now().strftime('%y%m%d_%H%M%S')
        if spect_output_dir is None:
            spect_output_dir = os.path.join(output_dir,
                                            f'spectrograms_generated_{timenow}')
            os.makedirs(spect_output_dir)
        spect_files = audio.to_spect(audio_format=audio_format,
                                     spect_params=spect_params,
                                     output_dir=spect_output_dir,
                                     audio_files=audio_files,
                                     annot_list=annot_list,
                                     labelset=labelset)
        spect_format = 'npz'
    else:  # if audio format is None
        spect_files = None

    from_files_kwargs = {
        'spect_format': spect_format,
        'labelset': labelset,
        'load_spects': load_spects,
        'annot_list': annot_list,
    }

    if spect_files:
        from_files_kwargs['spect_files'] = spect_files
        logger.info(
            f'creating VocalDataset from spectrogram files in: {output_dir}'
        )
    else:
        from_files_kwargs['spect_dir'] = data_dir
        logger.info(
            f'creating VocalDataset from spectrogram files in: {data_dir}'
        )

    vds = spect.from_files(**from_files_kwargs)
    if save_vds:
        if vds_fname is None:
            timenow = datetime.now().strftime('%y%m%d_%H%M%S')
            vds_fname = f'prep_{timenow}.vds.json'

        if not vds_fname.endswith('.json'):
            vds_fname += '.json'

        if output_dir:
            vds_path = os.path.join(output_dir, vds_fname)
        else:
            vds_path = os.path.join(os.getcwd(), vds_fname)

        vds.save(json_fname=vds_path)
    else:
        vds_path = None

    if return_vds and return_path:
        return vds, vds_path
    elif return_path:
        return vds_path
    elif return_vds:
        return vds
