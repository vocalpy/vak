def prep(purpose,
         labelset,
         data_dir=None,
         spect_files=None,
         spect_params=None,
         annotation_file=None,
         total_train_set_dur=None,
         val_dur=None,
         test_dur=None,
         config_file=None,
         silent_gap_label=0,
         skip_files_with_labels_not_in_labelset=True,
         output_dir=None):
    """prepare datasets for training and testing networks,
    or for making predictions with trained networks

    Parameters
    ----------
    purpose : str
        one of {'train', 'predict', 'learncurve'}
    labelset : list
        of str or int, set of unique labels used for segments
        (e.g. unique set of syllables in a bird's song,
        phonemes in speech, etc.)
    data_dir : str
        Path to a directory that contains audio files and annotation
        files. Default is None.
    spect_files_path : str
        Path to a directory of files that contain spectrograms.
        Default is None.

    Note that either `data_dir` or `spect_files_path` must be specified.

    audio_format : str
        Audio format of files in `data_dir`. One of {'.wav', '.cbin'}.
        Default is None, in which case vak tries to determine the format.
    annot_format : str
        Format of annotation files in `data_dir`. One of {'csv', 'koumura', 'notmat'}.
        `vak` uses the `crowsetta` library to parse annotations.
        Default is None, in which case vak tries to determine the format.
    spect_params
    annotation_file

    Other Parameters
    ----------------
    total_train_set_dur
    val_dur
    test_dur
    config_file
    silent_gap_label
    skip_files_with_labels_not_in_labelset
    output_dir

    Returns
    -------

    """