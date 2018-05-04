# cnn_bilstm package
For testing the cnn-bilstm neural network for segmentation of birdsong into 
syllables. This repository contains scripts to reproduce results, as well as 
the cnn-bilstm package. The package contains the network and various utility 
functions used by the scripts.

## install

It's probably easiest to use Anaconda. 
First set up a conda environment and clone the repo
`$ conda create -n cnn-bilstm numpy scipy joblib tensorflow-gpu ipython jupyter`
`$ git clone https://github.com/NickleDave/tf_syllable_segmentation_annotation davids_fork_of_tf_sylseg`
`$ source activate cnn-bilstm`

## usage

There are 3 main scripts that are run consecutively.
The scripts accept a config.ini file as a command-line argument; you will use 
the same config.ini file with each script but you will make changes to it after 
running each script.

### 1. Make data sets

You will make data sets for training, validation,
 and testing with the `make_data.py` script.  
Before you run the script you need to create a `config.ini` file. 
You can adapt the `template_config.ini` file that's in this repository.
In the `config` file, set values for the following options in the '[DATA]` section:  
```ini
[DATA]
labelset = iabcdefghjk  # set of labels, str, int, or a 
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

After writing the `config` file, run `make_data.py` at the command line with the
 `config` file specified: `(cnn-bilstm) $ python 
 ./cnn-bilstm/make_data.py config_03218_bird0.ini`

### 2. Generate learning curves

After making the data sets, you generate the data for learning curves,
using the `learn_curve.py` script.
A learning curve is a plot where the x-axis is size of the training set 
(in this case, duration in seconds) and the y axis is error, accuracy, or some 
similar metric. The script grabs random subsets of training data of a fixed size
 (specified by the `train_set_durs` option in the `config` file) and uses the 
 subsets to train the network. This model is then saved and its ability to 
 generalize is estimated by measuring error on a test set, using the 
`summary.py` script (below).

Before running the `learn_curve.py` script you again need to modify some
options in the `config.ini` file.
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

Most importantly, you should change `train_dict_path` to wherever 
'train_data_dict' got saved; the path should include the filename. 
Do the same for `val_data_path` and `test_data_path`.

You'll also want to change the first `results_dir` option under the [OUTPUT] 
section to wherever you want to save all the output (checkpoint files, copies of
 training data, etc.).

After modifying the `config` file, run `learn_curve.py` at the command line with
 the `config` file specified: `(cnn-bilstm) $ CUDA_VISIBLE_DEVICES=0 python 
 ./cnn-bilstm/make_data.py config_03218_bird0.ini`

(Note it is not *required* to specify which GPU to use with `CUDA_VISIBLE_DEVICES`.)

#### output of learn_curves.py

The script will make a subdirectory in `results_dir`, 
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

To reload a saved model, you use those `.meta` and `.data` files saved by the
Tensorflow checkpoint saver:
```Python
meta_file = glob(os.path.join(training_records_dir, 'checkpoint*meta*'))[0]
data_file = glob(os.path.join(training_records_dir, 'checkpoint*data*'))[0]

with tf.Session(graph=tf.Graph()) as sess:
    saver = tf.train.import_meta_graph(meta_file)
    saver.restore(sess, data_file[:-20])  # don't need .data-etc... extension
```

You can then 

### 3.Generate summary of results

The `summary.py` script is run after the `learn_curves.py` script.  
Before running `summary.py`, add the `results_dir_made_by_main_script` option 
to the [OUTPUT] section of the `config.ini` file. This should be the full path
to the results directory that was created by the `learn_curves.py` script.
After adding this option to [OUTPUT], you run `summary.py` from the command line
 with the `config.ini` as an argument, like so:
 
 `(virtual-env) $ CUDA_VISIBLE_DEVICES=0 python
  ./cnn-bilstm/summary.py ./configs/config.ini`

(Note it is not *required* to specify which GPU to use with `CUDA_VISIBLE_DEVICES`.)

#### output of summary.py

The script will make a `summary` subdirectory in `results_dir`, 
then loop through all the results subdirectories (one for each replicate of each
training set duration) and compute error on the training and test sets.
The script saves the following variables in `summary`:

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

After this script finishes, you must change

`(cnn-bilstm) $ CUDA_VISIBLE_DEVICES=0 python ./cnn-bilstm/summary.py config_03218_bird0.ini`

## Using spectrograms generated with Matlab

You will follow the same three steps above, but your config file should have the
 same format as `template_config_matlab_spectrograms.ini`.
See `README_config.md` for an explanation of additional options in that template
config file.