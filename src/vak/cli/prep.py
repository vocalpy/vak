import os
import logging
from configparser import ConfigParser
from datetime import datetime

from .. import dataset

VDS_JSON_EXT = '.vds.json'


def prep(labelset,
         data_dir,
         total_train_set_dur,
         test_dur,
         config_file,
         annot_format,
         val_dur=None,
         skip_files_with_labels_not_in_labelset=True,
         output_dir=None,
         audio_format=None,
         spect_format=None,
         annot_file=None,
         spect_params=None):
    """command-line function that prepares datasets
    for training, validating, and/or testing networks

    Parameters
    ----------
    labelset : list
        of str or int, set of labels for syllables
    data_dir : str
        path to directory with audio files from which to make dataset
    total_train_set_dur : float
        total duration of training set, in seconds.
        Training subsets of shorter duration will be drawn from this set.
    val_dur : float
        total duration of validation set, in seconds.
    test_dur : float
        total duration of test set, in seconds.
    skip_files_with_labels_not_in_labelset : bool
        if True, skip a file if the labels variable contains labels not
        found in 'labelset'. Default is True.
    output_dir : str
        Path to location where data sets should be saved. Default is None,
        in which case data sets are saved in the current working directory.
    audio_format : str
        format of audio files. One of {'wav', 'cbin'}.
    spect_format : str
        format of array files containing spectrograms as matrices, and
        vectors representing frequency bins and time bins of spectrogram.
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
    None

    Saves a VocalizationDataset generated from data_dir, as well as training, test, and
    validation sets created from that VocalizationDataset.
    """
    if audio_format is None and spect_format is None:
        raise ValueError("Must specify either audio_format or spect_format")

    if audio_format and spect_format:
        raise ValueError("Cannot specify both audio_format and spect_format, "
                         "unclear whether to create spectrograms from audio files or "
                         "use already-generated spectrograms from array files")

    logger = logging.getLogger(__name__)
    logger.setLevel('INFO')

    timenow = datetime.now().strftime('%y%m%d_%H%M%S')
    _, tail = os.path.split(data_dir)
    vds_stem = f'{tail}_prep_{timenow}'
    vds_fname = os.path.join(output_dir, vds_stem + VDS_JSON_EXT)

    vds, vds_path = dataset.prep(labelset=labelset,
                                 data_dir=data_dir,
                                 annot_format=annot_format,
                                 skip_files_with_labels_not_in_labelset=skip_files_with_labels_not_in_labelset,
                                 output_dir=output_dir,
                                 save_vds=True,
                                 vds_fname=vds_fname,
                                 return_vds=True,
                                 return_path=True,
                                 load_spects=False,
                                 annot_file=annot_file,
                                 audio_format=audio_format,
                                 spect_format=spect_format,
                                 spect_params=spect_params)

    if val_dur is not None:
        train_vds, test_vds, val_vds = dataset.split.train_test_dur_split(vds,
                                                                          labelset=labelset,
                                                                          train_dur=total_train_set_dur,
                                                                          val_dur=val_dur,
                                                                          test_dur=test_dur)
    else:
        train_vds, test_vds = dataset.split.train_test_dur_split(vds,
                                                                 labelset=labelset,
                                                                 train_dur=total_train_set_dur,
                                                                 test_dur=test_dur)
        val_vds = None

    vds_to_save_keys = ['train', 'test']
    vds_to_save_vals = [train_vds, test_vds]
    if val_vds:
        vds_to_save_keys.append('val')
        vds_to_save_vals.append(val_vds)

    saved_vds_dict = {}
    for key, a_vds in zip(vds_to_save_keys, vds_to_save_vals):
        json_fname = os.path.join(output_dir, vds_stem + f'.{key}' + VDS_JSON_EXT)
        a_vds.save(json_fname=json_fname)
        saved_vds_dict[key] = json_fname

    # rewrite config file with paths where VocalizationDatasets were saved
    config = ConfigParser()
    config.read(config_file)
    for key, path in saved_vds_dict.items():
        config.set(section='TRAIN',
                   option=key + '_data_path',
                   value=path)

    with open(config_file, 'w') as config_file_rewrite:
        config.write(config_file_rewrite)
