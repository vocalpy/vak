from datetime import datetime
from pathlib import Path
import warnings

from .. import split
from ..converters import expanded_user_path, labelset_to_set
from ..io import dataframe
from ..logging import log_or_print


VALID_PURPOSES = frozenset(['eval',
                            'learncurve',
                            'predict',
                            'train',
                            ])


def prep(data_dir,
         purpose,
         output_dir=None,
         audio_format=None,
         spect_format=None,
         spect_params=None,
         spect_output_dir=None,
         annot_format=None,
         annot_file=None,
         labelset=None,
         train_dur=None,
         val_dur=None,
         test_dur=None,
         logger=None,
         ):
    """prepare datasets from vocalizations.
    High-level function that prepares datasets to be used by other
    high-level functions like vak.train, vak.predict, and vak.learncurve

    Parameters
    ----------
    data_dir : str, Path
        path to directory with files from which to make dataset
    purpose : str
        one of {'train', 'predict', 'learncurve'}
    output_dir : str
        Path to location where data sets should be saved.
        Default is None, in which case data sets to `data_dir`.
    audio_format : str
        format of audio files. One of {'wav', 'cbin'}.
        Default is None, but either audio_format or spect_format must be specified.
    spect_format : str
        format of files containg spectrograms as 2-d matrices. One of {'mat', 'npz'}.
        Default is None, but either audio_format or spect_format must be specified.
    spect_params : dict, vak.config.SpectParams
        parameters for creating spectrograms. Default is None.
    spect_output_dir : str
        path to location where spectrogram files should be saved.
        Default is None, in which case it defaults to ``data_dir``.
        A new directory will be created in ``spect_output_dir`` with
        the name 'spectrograms_generated_{time stamp}'.
    annot_format : str
        format of annotations. Any format that can be used with the
        crowsetta library is valid. Default is None.
    annot_file : str
        Path to a single annotation file. Default is None.
        Used when a single file contains annotations for multiple audio files.
    labelset : str, list, set
        of str or int, set of unique labels for vocalizations. Default is None.
        If not None, then files will be skipped where the associated annotation
        contains labels not found in ``labelset``.
        ``labelset`` is converted to a Python ``set`` using ``vak.converters.labelset_to_set``.
        See help for that function for details on how to specify labelset.
    train_dur : float
        total duration of training set, in seconds. When creating a learning curve,
        training subsets of shorter duration will be drawn from this set. Default is None.
    val_dur : float
        total duration of validation set, in seconds. Default is None.
    test_dur : float
        total duration of test set, in seconds. Default is None.

    Other Parameters
    ----------------
    logger : logging.Logger
        instance created by vak.logging.get_logger. Default is None.

    Returns
    -------
    vak_df : pandas.DataFrame
        that represents a dataset of vocalizations
    csv_path : Path
        to csv saved from vak_df

    Notes
    -----
    Saves a .csv file representing the dataset generated from data_dir.

    Datasets are used to train neural networks that segment audio files into
    vocalizations, and then predict labels for those segments.
    The function also prepares datasets so neural networks can predict the
    segmentation and annotation of vocalizations in them.
    It can also split a dataset into training, validation, and test sets,
    e.g. for benchmarking different neural network architectures.

    If the 'purpose' is set to 'train' or 'learncurve', and/or
    the duration of either the training or test set is provided,
    then the function attempts to split the dataset into training and test sets.
    A duration can also be specified for a validation set
    (used to measure performance during training).
    In these cases, the 'split' column in the .csv
    identifies which files (rows) belong to the training, test, and
    validation sets created from that Dataset.

    If the 'purpose' is set to 'predict' or 'eval',
    or no durations for any of the training sets are specified,
    then the function assumes all the vocalizations constitute a single
    dataset, and for all rows the 'split' columns for that dataset
    will be 'predict' or 'test' (respectively).
    """
    # pre-conditions ---------------------------------------------------------------------------------------------------
    if purpose not in VALID_PURPOSES:
        raise ValueError(
            f'purpose must be one of: {VALID_PURPOSES}\nValue for purpose was: {purpose}'
        )

    if audio_format is None and spect_format is None:
        raise ValueError("Must specify either audio_format or spect_format")

    if audio_format and spect_format:
        raise ValueError("Cannot specify both audio_format and spect_format, "
                         "unclear whether to create spectrograms from audio files or "
                         "use already-generated spectrograms from array files")

    if labelset is not None:
        labelset = labelset_to_set(labelset)

    data_dir = expanded_user_path(data_dir)
    if not data_dir.is_dir():
        raise NotADirectoryError(
            f'data_dir not found: {data_dir}'
        )

    if output_dir:
        output_dir = expanded_user_path(output_dir)
    else:
        output_dir = data_dir

    if not output_dir.is_dir():
        raise NotADirectoryError(
            f'output_dir not found: {output_dir}'
        )

    if spect_output_dir:
        spect_output_dir = expanded_user_path(spect_output_dir)
        if not spect_output_dir.is_dir():
            raise NotADirectoryError(
                f'spect_output_dir not found: {spect_output_dir}'
            )

    if purpose == 'predict':
        if labelset is not None:
            warnings.warn(
                "purpose set to 'predict', but a labelset was provided."
                "This would cause an error because the dataframe.from_files section will attempt to "
                f"check whether the files in the data_dir ({data_dir}) have labels in "
                "labelset, even though those files don't have annotation.\n"
                "Setting labelset to None."
            )
            labelset = None
    log_or_print(
        msg=f'purpose for dataset: {purpose}',
        logger=logger, level='info'
    )
    # ---- figure out file name ----------------------------------------------------------------------------------------
    data_dir_name = data_dir.name
    timenow = datetime.now().strftime('%y%m%d_%H%M%S')
    csv_fname_stem = f'{data_dir_name}_prep_{timenow}'
    csv_path = output_dir.joinpath(f'{csv_fname_stem}.csv')

    # ---- figure out if we're going to split into train / val / test sets ---------------------------------------------
    # catch case where user specified duration for just training set, raise a helpful error instead of failing silently
    if ((purpose == 'train' or purpose == 'learncurve') and
            ((train_dur is not None and train_dur > 0) and
             (val_dur is None or val_dur == 0) and
             (test_dur is None or val_dur == 0)
             )):
        raise ValueError(
            'duration specified for just training set, but prep function does not currently support creating a '
            'single split of a specified duration. Either remove the train_dur option from the prep section and '
            'rerun, in which case all data will be included in the training set, or specify values greater than '
            'zero for test_dur (and val_dur, if a validation set will be used)'
        )

    if all([dur is None for dur in (train_dur, val_dur, test_dur)]) or purpose in ('eval', 'predict'):
        # then we're not going to split
        log_or_print(msg='will not split dataset', logger=logger, level='info')
        do_split = False
    else:
        if val_dur is not None and train_dur is None and test_dur is None:
            raise ValueError('cannot specify only val_dur, unclear how to split dataset into training and test sets')
        else:
            log_or_print(msg='will split dataset', logger=logger, level='info')
            do_split = True

    # ---- actually make the dataset -----------------------------------------------------------------------------------
    vak_df = dataframe.from_files(labelset=labelset,
                                  data_dir=data_dir,
                                  annot_format=annot_format,
                                  annot_file=annot_file,
                                  audio_format=audio_format,
                                  spect_format=spect_format,
                                  spect_output_dir=spect_output_dir,
                                  spect_params=spect_params,
                                  logger=logger)

    if do_split:
        # save before splitting, jic duration args are not valid (we can't know until we make dataset)
        vak_df.to_csv(csv_path)
        vak_df = split.dataframe(vak_df,
                                 labelset=labelset,
                                 train_dur=train_dur,
                                 val_dur=val_dur,
                                 test_dur=test_dur,
                                 logger=logger)

    elif do_split is False:  # add a split column, but assign everything to the same 'split'
        # ideally we would just say split=purpose in call to add_split_col, but
        # we have to special case, because "eval" looks for a 'test' split (not an "eval" split)
        if purpose == 'eval':
            split_name = 'test'  # 'split_name' to avoid name clash with split package
        elif purpose == 'predict':
            split_name = 'predict'

        vak_df = dataframe.add_split_col(vak_df, split=split_name)

    log_or_print(msg=f'saving dataset as a .csv file: {csv_path}', logger=logger, level='info')
    vak_df.to_csv(csv_path, index=False)  # index is False to avoid having "Unnamed: 0" column when loading

    return vak_df, csv_path
