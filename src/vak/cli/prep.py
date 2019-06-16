import os
import logging
from configparser import ConfigParser
from datetime import datetime
from .. import dataset

VDS_JSON_EXT = '.vds.json'


def prep(data_dir,
         config_file,
         annot_format=None,
         train_dur=None,
         val_dur=None,
         test_dur=None,
         labelset=None,
         output_dir=None,
         audio_format=None,
         spect_format=None,
         annot_file=None,
         spect_params=None):
    """command-line function that prepares datasets from vocalizations

    Datasets are used to train neural networks that segment audio files into
    vocalizations, and then predict labels for those segments.
    The function also prepares datasets so neural networks can predict the
    segmentation and annotation of vocalizations in them.
    It can also split a dataset into training, validation, and test sets,
    e.g. for benchmarking different neural network architectures.

    If no durations for any of the training sets are specified, then the
    function assumes all the vocalizations constitute a single training
    dataset. If the duration of either the training or test set is provided,
    then the function attempts to split the dataset into training and test
    sets; the

    Parameters
    ----------
    data_dir : str
        path to directory with audio files or spectrogram files from which to make dataset
    config_file : str
        path to config.ini file. Used to rewrite file with options determined by
        this function and needed for other functions (e.g. cli.summary)
    annot_format : str
        format of annotations. Any format that can be used with the
        crowsetta library is valid. Default is None.
    train_dur : float
        duration of training set, in seconds. Default is None.
        If None, this function assumes all vocalizations should be made into
        a single training set.
    val_dur : float
        duration of validation set, in seconds. Default is None.
    test_dur : float
        total duration of test set, in seconds. Default is None.
    labelset : list
        of str or int, set of labels for syllables. Default is None.
        If not None, then files will be skipped where the 'labels' array in the
        corresponding annotation contains labels that are not found in labelset
    output_dir : str
        Path to location where data sets should be saved. Default is None,
        in which case data sets are saved in the current working directory.
    audio_format : str
        format of audio files. One of {'wav', 'cbin'}.
    spect_format : str
        format of array files containing spectrograms as matrices, and
        vectors representing frequency bins and time bins of spectrogram.
        One of {'mat', 'npz'}.
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

    logger = logging.getLogger(__name__)
    logger.setLevel('INFO')

    timenow = datetime.now().strftime('%y%m%d_%H%M%S')
    _, tail = os.path.split(data_dir)
    vds_fname_stem = f'{tail}_prep_{timenow}'

    if all([dur is None for dur in (train_dur, val_dur, test_dur)]):
        # assume the whole dataset is a training set
        do_split = False
        save_vds = False  # because we'll save it in the loop below
        vds_fname = None
    else:
        if val_dur is not None and train_dur is None and test_dur is None:
            raise ValueError('cannot specify only val_dur, unclear how to split dataset into training and test sets')
        else:
            # save before splitting, jic duration args are not valid (we can't know until we make dataset)
            do_split = True
            save_vds = True
            vds_fname = os.path.join(output_dir, f'{vds_fname_stem}{VDS_JSON_EXT}')

    vds = dataset.prep(labelset=labelset,
                       data_dir=data_dir,
                       annot_format=annot_format,
                       output_dir=output_dir,
                       save_vds=save_vds,
                       vds_fname=vds_fname,
                       return_vds=True,
                       return_path=False,
                       load_spects=False,
                       annot_file=annot_file,
                       audio_format=audio_format,
                       spect_format=spect_format,
                       spect_params=spect_params)

    if do_split:
        if val_dur is not None:
            train_vds, val_vds, test_vds = dataset.split.train_test_dur_split(vds,
                                                                              labelset=labelset,
                                                                              train_dur=train_dur,
                                                                              val_dur=val_dur,
                                                                              test_dur=test_dur)
        else:
            train_vds, test_vds = dataset.split.train_test_dur_split(vds,
                                                                     labelset=labelset,
                                                                     train_dur=train_dur,
                                                                     test_dur=test_dur)
            val_vds = None

        vds_to_save_keys = ['train', 'test']
        vds_to_save_vals = [train_vds, test_vds]

        if val_dur is not None:
            vds_to_save_keys.append('val')
            vds_to_save_vals.append(val_vds)

    elif do_split is False:
        # we assumed the whole dataset is a training set
        vds_to_save_keys = ['train']
        vds_to_save_vals = [vds]  # was returned above, but hasn't been saved yet

    saved_vds_dict = {}
    for key, a_vds in zip(vds_to_save_keys, vds_to_save_vals):
        json_fname = os.path.join(output_dir, f'{vds_fname_stem}.{key}{VDS_JSON_EXT}')
        a_vds.save(json_fname=json_fname)
        saved_vds_dict[key] = json_fname

    # rewrite config file with paths where VocalizationDatasets were saved
    config = ConfigParser()
    config.read(config_file)
    for key, path in saved_vds_dict.items():
        config.set(section='TRAIN',
                   option=f'{key}_vds_path',
                   value=path)

    with open(config_file, 'w') as config_file_rewrite:
        config.write(config_file_rewrite)
