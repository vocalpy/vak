# specification of options in config files

Below is a hopefully clear explanation in plain English of the possible
options in the `config.ini` file that the scripts use.
For a template config file that you can modify, see 
[./template_config.ini](./template_cnfig.ini)

## [NETWORK] section
Hyperparameters for network and related options such as batch size

```
[NETWORK]
batch_size = 11
time_steps = 88
learning_rate = 0.001
```

`batch_size` : int  
    Number of spectrogram 'chunks' in a batch, where a 'chunk' will be of
    size (number of frequency bins in spectrogram * `time_steps`)  
`time_steps` : int  
    Number of time bins that networks gets to see.  
`learning_rate` : float  
    For optimizer

## [SPECTROGRAM] section
Parameters used when making spectrograms from audio files.
** Note this section is *not* required if you are using .mat
files that contain already generated spectrograms. Instead you
should specify the location of those .mat files in the [DATA]
section below.
```
[SPECTROGRAM]
fft_size=512
step_size=64
freq_cutoffs = 500, 10000
thresh = 6.25
transform_type=log_spect
```

`fft_size` : int  
    number of samples to use in FFT  
`step_size` : int  
    A.K.A. "hop size", distance to move forward to grab next segment for FFT  
`freq_cutoffs` : int  
    Bandpass frequencies, becomes a two-element list of ints  
`thresh` : float  
    Threshold. Spectrogram elements with value below threshold are set to
    the threshold value so this becomes the "floor" value.  
    Optional, if not specified then no thresholding is applied.  
`transform_type` : str  
    One of the following:  
    {'log_spect', 'log_spect_plus_one'}  
    'log_spect' is log of spectrogram  
    'log_spect_plus_one' is log(spectrogram + 1)  
    Optional, if not specified then no transform is applied.  

*Note that if `transform_type` is `log_spect` and a `thresh` value is specified,
then the threshold used is the **negative** value of `thresh`.* (This is to 
maintain the behavior of the spectrogram function as originally written.)

## [DATA] section

```
[DATA]
labelset = iabcdefghjk
data_dir = /home/user/data/subdir/
# durations of training, validation, and test sets are given in seconds
total_train_set_duration = 400
train_set_durs = 5, 15, 30, 45, 60, 75, 90, 105
validation_set_duration = 100
test_set_duration = 400
skip_files_with_labels_not_in_labelset = Yes
```

`labelset` : str or a "range string"  
    set of labels that should appear in data set  
    e.g., 'iabcde' if those are the labels, or '012345'  
    In the case that labels are a set of integers, they can be
    specified with a range using a dash and commas, such as
    '1-27' or '1-10,12,13'.  
    (You don't have to type the quotations, those are just to indicate
    strings.)  
`data_dir` : str  
    absolute path to directory with audio files from which data sets will
    be made.
    Not required if `mat_spect_files_path` & `mat_spects_annotation_file`
    are defined.
`all_labels_are_int` : bool
    if 'Yes' or 'True', script assumes all labels are of type int.
    Will convert labelset from str to int
    so that error-checking functions properly when it determines
    whether all labels are present in training, validation,  and
    test data sets.
`mat_spect_files_path` : str
    absolute path to directory with `.mat` files containing spectrograms.
    Not required if `data_dir` is defined.
    Those .mat files must contain the following variables:
        s : spectrogram
        t : time bins vector
        f : frequency bins vector
        labels : vector of same length as time bins, where each element
                 is a label for that time bin, 
                 i.e. what class of syllable does this time bin belong to?
                 Class could also be silent gap between syllables 
`mat_spects_annotation_file` : str
    absolute path to `annotation.mat` file that contains `keys` and `elements`
    Not required if `data_dir` is defined.
    `keys` are audio file names, and `elements` contains the `segType`,
    (as well other variables).
    `segType` has the label applied to each segment.
`total_train_set_duration` : int  
    Total duration of training set in seconds.  
    Subsets of fixed sizes will be drawn from this larger set.  
    Used to fix the size of the larger set across experiments/birds/etc.  
`train_set_durs` : int  
    ints separated by commas, becomes a Python list.  
    Each is the target duration of a subset given in seconds.  
`validation_set_duration` : int  
    in seconds  
`test_set_duration` : int  
    in seconds  
`skip_files_with_labels_not_in_labelset` : bool
    if 'Yes' or 'True', skip any file that contains labels not in labelset.  
    Used to skip files with labels that occur very rarely such as noise
    or "mistakes" made by bird.  

## [TRAIN] section

```
[TRAIN]
train_data_path = /home/user/data/subdir/subsubdir1/spects/train_data_dict
val_data_path = /home/user/data/subdir/subsubdir1/spects/val_data_dict
test_data_path = /home/user/data/subdir/subsubdir1/spects/test_data_dict
use_train_subsets_from_previous_run = No
previous_run_path = /home/user/data/subdir/results_
normalize_spectrograms = Yes
n_max_iter = 18000
val_error_step = 150
checkpoint_step = 600
save_only_single_checkpoint_file = True
patience = None
replicates = 5
```

`train_data_path` : str  
    absolute path to `train_data_dict` created by `make_data()` when you run
    `main.py`. The `make_data()` function changes the value of this option
    for you.  
`val_data_path` : str  
    absolute path to `val_data_dict` created by `make_data()` when you run
    `main.py`. The `make_data()` function changes the value of this option
    for you.  
`test_data_path` : str  
    absolute path to `test_data_dict` created by `make_data()` when you run
    `main.py`. The `make_data()` function changes the value of this option
    for you.  
`use_train_subsets_from_previous_run` : bool  
    if `Yes` or `True` then re-use the randomly drawn subsets from a 
    previous run.  
`previous_run_path` : str  
    absolute path to a previous run (the subdirectory created in `results_dir`
    by `learn_curves.py`  
`normalize_spectrograms` : bool  
    If `Yes` or `True`, normalize spectrograms.  
    Normalization is done by subtracting mean off each frequency bin and
    dividing by standard deviation.
    Note that the mean and standard deviation are found for each subset
    of training data and then applied to validation and test data
    when estimating accuracy.
`n_max_iter` : int  
    Number of iterations (AKA epochs) to train.  
`val_error_step` : int  
    step at which to calculate validation error.  
    Every time iter modulo val_error_step is zero, the validation error
    will be calculated.
`checkpoint_step` : int  
    step at which to save checkpoint.  
    Every time iter modulo checkpoint_step is zero, a checkpoint will be
    saved.
`save_only_single_checkpoint_file` : bool
    if 'Yes' or 'True', overwrite checkpoint file at each checkpoint step 
    instead of saving many checkpoint files. To avoid taking up space with
    large checkpoint files.  
`patience` : int or None  
    if int, will stop if loss does not decrease for this number of steps  
`replicates` : int  
    Number of replicates for each train_dur_size.  
    E.g., if 5 then for each of train set size of duration [10,20,30] seconds, 
    generate 5 random subsets of each duration and train a model with those 
    subsets.  


## OUTPUT section

```
[OUTPUT]
root_results_dir = /home/user/data/subdir/
results_dir_made_by_main_script = /home/user/data/subdir/results_
```

`root_results_dir` : str  
    absolute path to directory where you want results from running
    `main.py` to be saved. Each time you run `main.py`, it will
    create a new subdirectory in `root_results_dir`, which will
    have the name:
`results_dir_made_by_main_script` : str  
    absolute path to subdirectory made by `train()` when you ran 
    `main.py`. The `train()` function automatically changes this
    value for you.