# Experiments in this paper
Please see the (README.md)[./README.md] for instructions on how to install 
this package. This document describes how to reproduce the experiments in 
the paper.

## usage

To re-run experiments testing the model, use the `main.py` script.
You run it with config.ini files, using one of three command-line flags:
```
  -g GLOB, --glob GLOB  string to use with glob function to search for config
                        files
  -t TXT, --txt TXT     name of .txt file containing list of config files to
                        run
  -c CONFIG, --config CONFIG
                        name of a single config.ini file
```

As the `--help` explains, the `--glob` flag is set with an argument that 
becomes the search string used by `glob` from the Python standard library to 
find config.ini files.  
For example, to re-run all experiments from the 
[repository here](https://figshare.com/articles/BirdsongRecognition/3470165),
you could execute the following:  
`(cnn-bilstm-conda-env)$ python main.py --glob ./configs/config*bird*ini`  
which will match all config.ini files that have the word bird in the title, 
which is true for all the config files for that repository in `./configs`.
(You'd need to download the repository and change the paths in the 
config files to reflect where you have it saved on your system.)

The `--txt` flag is set with an argument that is the name of a .txt file 
containing list of config files to run.  
So, for example, to re-run the experiments on Bengalese Finch song from 
the [repository here](https://figshare.com/articles/Bengalese_Finch_song_repository/4805749), 
you could use the .txt file in `./configs` that contains a list of config.ini 
files for each bird from the repository.
To do so, you'd execute the following at the command line:  
`(cnn-bilstm-conda-env)$ python main.py --txt ./configs/config_list_bf_song_repository_all.txt `  
(Again you need to download the repository and change the paths in the 
config files to reflect where you have it saved on your system.)

Bengalese finch song used as data for these experiments 
is from the following repositories:
https://figshare.com/articles/Bengalese_Finch_song_repository/4805749
https://figshare.com/articles/BirdsongRecognition/3470165

## explanation of how main.py runs

The `main.py` file is a simple script that runs 3 functions consecutively.
It accepts a config.ini file as a command-line argument (see `usage` above); 
the same config.ini file is used by each function, which updates the file as 
appropriate (e.g. with paths to where training data is saved).
Below is a more detailed explanation of each function that `main.py` runs.


### 1. `cnn_bilstm.train_utils.make_data` : makes data sets

Makes data sets for training, validation, and testing.  
Before you run the script you need to create a `config.ini` file. 
You can adapt the `template_config.ini` file that's in this repository.
In the `config` file, set values for the following options in the '[DATA]` section:  
```ini
[DATA]
labelset = iabcdefghjk  # set of labels, str, int, or a range, e.g., '1-20, 22'
# see docstring of cnn_bilstm.utils.range_str in for explanation of valid ranges
data_dir = /home/user/data/subdir/  # directory with audio files
# durations of training, validation, and test sets in seconds
total_train_set_duration = 400
train_set_durs = 5, 15, 30, 45, 60, 75, 90, 105
validation_set_duration = 100
test_set_duration = 400
skip_files_with_labels_not_in_labelset = Yes
```

For more about what each of these options mean, see 
[README_config.md](./README_config.md).


### 2. `cnn_bilstm.train_utils.train` : train models

After making the data sets, you generate trained models for learning curves.
A learning curve is a plot where the x-axis is size of the training set 
(in this case, duration in seconds) and the y axis is error, accuracy, or some 
similar metric. This function grabs random subsets of training data of a fixed size
 (specified by the `train_set_durs` option in the `config` file) and uses the 
 subsets to train the network. This model is then saved and its ability to 
 generalize is estimated by measuring error on a test set, using the 
`learn_curve` function (below).

Before `train()` can run, you will need to have set the following options in the
`config.ini` file. Note that the `make_data()` function will automatically set 
the values of the `data_path`s for you when you run it.

```ini
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

You'll also want to change the first `results_dir` option under the [OUTPUT] 
section to wherever you want to save all the output (Tensorflow checkpoint files, 
copies of training data, etc.).

#### output of `train()`

The `train()` function will make a subdirectory in `results_dir`, 
and in that subdirectory will make one subdirectory for each replicate of 
each duration of training set.  
Like so:
```
tf_syl_seg_results/
    results_01012018_130100/
        records_for_training_set_of_duration_25s_replicate0/
        records_for_training_set_of_duration_25s_replicate1/
        records_for_training_set_of_duration_25s_replicate2/
        ...
        records_for_training_set_of_duration_200s_replicate4/
        records_for_training_set_of_duration_200s_replicate5/
```

Each `records_for_training_set_of_duration_...` directory will contain the 
following files. Files that contain a single vector should be loaded with the
`joblib` library (because they were saved using that library).
- Checkpoint files that contains the saved model (`.data`, `.meta`, 
  and `.index` files)
- If the spectrograms were normalized/scaled, 
  a file containing the scaled spectrograms.
- `train_inds`, a file containing a vector of indices that were used to obtain
  the training set by indexing into `X_train` a large matrix consisting of all
  the spectrograms from the training set concatenated together.
- `val_errs`: vector with validation error
- `costs`: vector with costs for each training step
- `iter_order`: vector containing indices used to randomly grab training data
  for each iteration. Instead of just sliding a window along the training set
  and presenting each window in order, the windows are presented in this
  randomized order.

To reload a saved model, you use a checkpoint file saved by the
Tensorflow checkpoint saver. Here's an example of how to do this, taken 
from the `cnn_bilstm.train_utils.learn_curve` function:
```Python
meta_file = glob(os.path.join(training_records_dir, 'checkpoint*meta*'))[0]
data_file = glob(os.path.join(training_records_dir, 'checkpoint*data*'))[0]

model = CNNBiLSTM(n_syllables=n_syllables,
                  input_vec_size=input_vec_size,
                  batch_size=batch_size)

with tf.Session(graph=model.graph) as sess:
    model.restore(sess=sess,
                  meta_file=meta_file,
                  data_file=data_file)
```

### 3. `cnn_bilstm.train_utils.learn_curve` : generate learning curve from results

The `learn_curve()` function is run after the `train()` function.  
Note that `train()` will add the `results_dir_made_by_main_script` option 
to the [OUTPUT] section of the `config.ini` file.

#### output of learn_curve()

The function will make a `summary` subdirectory in the `results_dir` subdirectory 
created by `train()`, then loop through all the results subdirectories (one for 
each replicate of each training set duration) and compute error on the training 
and test sets. The script saves the following variables in `summary`:

```
scaled_reshaped_spects : dict
    Spectrograms from training data set.
    Reshaped as they are when fed to the network for prediction.     
scaled_test_spects : dict
    Spectrograms from test data set.
    Reshaped as they are when fed to the network for prediction. 
train_err : ndarray
    Error on training set for all replicates of all training set durations.
    m x n matrix where m is number of durations of training set and n is
    number of replicates for each duration.
test_arr : ndarray
    Error on test set for all replicates of all training set durations.
    m x n matrix where m is number of durations of training set and n is
    number of replicates for each duration.
Y_pred_train_and_test : dict
    with following key, value pairs:
        Y_pred_train_all : list
            predictions for training set, when network was given 
            scaled_reshaped_spects as inputs.
            m lists of n lists of type ndarray
            where m is number of durations of training set
            and n is number of replicates for each duration
            and each ndarray is the output from a trained network 
        Y_pred_test_all
            predictions for test set, when network was given 
            scaled_test_spects as inputs.
            m lists of n lists of type ndarray
            where m is number of durations of training set
            and n is number of replicates for each duration
            and each ndarray is the output from a trained network 
        Y_pred_train_labels_all
            predictions for training set, when network was given 
            scaled_test_spects as inputs.
            m lists of n lists of type str
            where m is number of durations of training set
            and n is number of replicates for each duration
            and each ndarray is the output from a trained network 
        Y_pred_test_labels_all
            predictions for test set, when network was given 
            scaled_test_spects as inputs.
            m lists of n lists of type str
            where m is number of durations of training set
            and n is number of replicates for each duration
            and each ndarray is the output from a trained network 
        Y_train_labels : list
            of str, labels for training set.
            Used to measure syllable error rate
        Y_test_labels
            of str, labels for test set.
            Used to measure syllable error rate
        train_err : ndarray
            Error on training set for all replicates of all training set 
            durations. m x n matrix where m is number of durations of 
            training set and n is number of replicates for each duration.            
        test_err : ndarray
            Error on test set for all replicates of all training set 
            durations. m x n matrix where m is number of durations of 
            training set and n is number of replicates for each duration.
        train_lev : ndarray
            Levenshtein distance for training set for all replicates of 
            all training set durations. m x n matrix where m is number of 
            durations of training set and n is number of replicates for
            each duration.            
        train_syl_err_rate : ndarray
            train_lev normalized by length of training labels, for comparing
            between strings of different lengths
        test_lev : ndarray
            Levenshtein distance for test set for all replicates of 
            all training set durations. m x n matrix where m is number of 
            durations of training set and n is number of replicates for
            each duration.            
        test_syl_err_rate : ndarray
            train_lev normalized by length of test labels, for comparing
            between strings of different lengths
        train_set_durs : list
            of int, duration of training sets in seconds, as defined in
            config.ini and used by summary.py while calculating errors for 
            each item in this list (i.e. for each duration).
```

## Using spectrograms generated with Matlab

You will follow the same three steps above, but your config file should have the
 same format as `template_config_matlab_spectrograms.ini`.
See `README_config.md` for an explanation of additional options in that template
config file.