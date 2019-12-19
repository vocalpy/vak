import os
import logging
from configparser import ConfigParser
from datetime import datetime

from ..io import dataframe
from ..utils import train_test_dur_split


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
    """prepare datasets from vocalizations.
    Function called by command-line interface.

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
        in which case data sets is saved in data_dir.
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

    Notes
    -----
    Saves a .csv file representing the dataset generated from data_dir.
    If durations were specified for validation and test sets, then the .csv
    has a column representing which files belong to the training, test, and
    validation sets created from that Dataset.

    Datasets are used to train neural networks that segment audio files into
    vocalizations, and then predict labels for those segments.
    The function also prepares datasets so neural networks can predict the
    segmentation and annotation of vocalizations in them.
    It can also split a dataset into training, validation, and test sets,
    e.g. for benchmarking different neural network architectures.

    If no durations for any of the training sets are specified, then the
    function assumes all the vocalizations constitute a single training
    dataset. If the duration of either the training or test set is provided,
    then the function attempts to split the dataset into training and test sets.
    """
    # pre-conditions ---------------------------------------------------------------------------------------------------
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

    if output_dir:
        if not os.path.isdir(output_dir):
            raise NotADirectoryError(
                f'output_dir not found: {output_dir}'
            )
    elif output_dir is None:
        output_dir = data_dir

    # ---- logging -----------------------------------------------------------------------------------------------------
    logger = logging.getLogger(__name__)
    logger.setLevel('INFO')

    # ---- figure out what section we will be saving path in, and prefix of option name in that section ----------------
    # (e.g., if it's PREDICT section then the prefix will be 'predict' for 'predict_vds_path' option
    config = ConfigParser()
    config.read(config_file)
    if config.has_section('TRAIN'):
        section = 'TRAIN'
    elif config.has_section('LEARNCURVE'):
        section = 'LEARNCURVE'
    elif config.has_section('PREDICT'):
        section = 'PREDICT'
    else:
        raise ValueError(
            'Did not find a section named TRAIN, LEARNCURVE, or PREDICT in config.ini file;'
            ' unable to determine which section to add paths to prepared datasets to'
        )

    # ---- figure out file name ----------------------------------------------------------------------------------------
    timenow = datetime.now().strftime('%y%m%d_%H%M%S')
    _, tail = os.path.split(data_dir)
    csv_fname_stem = f'{tail}_prep_{timenow}'
    csv_path = os.path.join(output_dir, f'{csv_fname_stem}.csv')

    # ---- figure out if we're going to split into train / val / test sets ---------------------------------------------
    if all([dur is None for dur in (train_dur, val_dur, test_dur)]):
        # then we're not going to split
        do_split = False
    else:
        if val_dur is not None and train_dur is None and test_dur is None:
            raise ValueError('cannot specify only val_dur, unclear how to split dataset into training and test sets')
        else:
            do_split = True

    # ---- actually make the dataset -----------------------------------------------------------------------------------
    vak_df = dataframe.from_files(labelset=labelset,
                                  data_dir=data_dir,
                                  annot_format=annot_format,
                                  output_dir=output_dir,
                                  annot_file=annot_file,
                                  audio_format=audio_format,
                                  spect_format=spect_format,
                                  spect_params=spect_params)

    if do_split:
        # save before splitting, jic duration args are not valid (we can't know until we make dataset)
        vak_df.to_csv(csv_path)
        vak_df = train_test_dur_split(vak_df,
                                      labelset=labelset,
                                      train_dur=train_dur,
                                      val_dur=val_dur,
                                      test_dur=test_dur)

    elif do_split is False:
        # add a split column, but assign everything to the same 'split'
        vak_df = dataframe.add_split_col(vak_df, split=section.lower())

    vak_df.to_csv(csv_path)

    # use config and section from above to add csv_path to config.ini file
    config.set(section=section,
               option=f'csv_path',
               value=csv_path)

    with open(config_file, 'w') as config_file_rewrite:
        config.write(config_file_rewrite)
