# specification of options in config files

Below is a hopefully clear explanation in plain English of the possible
options in the `config.ini` file that the scripts use.
For a template config file that you can modify, see 
[./template_config.ini](./template_cnfig.ini)

## NETWORK section
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

## SPECTROGRAM section
Parameters used when making spectrograms from audio files
```
[SPECTROGRAM]
fft_size=512
step_size=64
freq_cutoffs = 500, 10000
thresh = 6.25
log_transform=True
```

`fft_size` : int  
    number of samples to use in FFT  
`step_size` : int  
    AKA "hop size", distance to move forward to grab next segment for FFT  
`freq_cutoffs` : int  
    Bandpass frequencies, becomes a two-element list of ints  
`thresh` : float or None  
    Threshold. Spectrogram elements with value below threshold are set to
    zero.  
`transform_type` : str  
    {'log_spect', 'log_spect_plus_one', None}  

## DATA section

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

## TRAIN section

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


## OUTPUT section

```
[OUTPUT]
results_dir = /home/user/data/subdir/
# the option below needs to be added after learn_curve.py runs
# because it is the name of the directory generated *in* output_dir
# by main.py that contains all the training records, data, etc.
results_dir_made_by_main_script = /home/user/data/subdir/results_
```

`results_dir` : str  
    absolute path to directory where you want results from `learn_curve.py`
     to be saved.  
`results_dir_made_by_main_script` : str  
    absolute path to subdirectory made by `results_dir` when you ran 
    `learn_curve.py`. You need to specify this before you run `summary.py`.