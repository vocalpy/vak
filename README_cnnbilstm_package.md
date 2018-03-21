# cnn_bilstm package
For testing the cnn-bilstm neural network for segmentation of birdsong into syllables.
This repository contains scripts to reproduce results, as well as the cnn-bilstm package.
The package contains the network and various utility functions used by the scripts.

## install

It's probably easiest to use Anaconda. First set up a conda environment and clone the repo
`$ conda create -n cnn-bilstm numpy scipy joblib tensorflow-gpu ipython jupyter`
`$ git clone https://github.com/NickleDave/tf_syllable_segmentation_annotation davids_fork_of_tf_sylseg`
`$ source activate cnn-bilstm`

## usage

There are 3 main scripts that are run consecutively.
The scripts accept a config.ini file; you will use the same config.ini file with each script
but you will make changes to it after running each script.

### 1. Make data sets

You will make data sets for training, validation, and testing with the `make_data.py` script.  
Before you run the script you need to create a `config.ini` file. You can adapt the 
`template_config.ini` file that's in this repository.
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
For more about what each of these options mean, see [README_config.md](./README_config.md).

After writing the `config` file, run `make_data.py` at the command line with the `config` file specified: 
`(cnn-bilstm) $ python ./cnn-bilstm/make_data.py config_03218_bird0.ini`

### 2. Generate learning curves

After making the data sets, you generate the data for learning curves,
using the `learn_curve.py` script.
A learning curve is a plot where the x-axis is size of the training set 
(in this case, duration in seconds) and the y axis is error, accuracy, or some similar metric.
The script grabs random subsets of training data of a fixed size (specified by the 
`train_set_durs` option in the `config` file) and uses the subsets to train the network.
This model is then saved and its ability to generalize is estimated by measuring error on 
a test set, using the `summary.py` script (below).

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

Most importantly, you should change `train_dict_path` to wherever 'train_data_dict' got saved; 
the path should include the filename. Do the same for `val_data_path` and `test_data_path`.

You'll also want to change the first `results_dir` option under the [OUTPUT] section to 
wherever you want to save all the output (checkpoint files, copies of training data, etc.).



After modifying the `config` file, run `learn_curve.py` at the command line with the `config` file specified:
`(cnn-bilstm) $ CUDA_VISIBLE_DEVICES=0 python ./cnn-bilstm/make_data.py config_03218_bird0.ini`

(Note it is not *required* to specify which GPU to use with `CUDA_VISIBLE_DEVICES`.)

### 3.Generate summary of results
After this script finishes, you must change

`(cnn-bilstm) $ CUDA_VISIBLE_DEVICES=0 python ./cnn-bilstm/summary.py config_03218_bird0.ini`

## Using your own spectrograms

use the mat_utils functions on the .mat form of the data, to make a data_dict that the main.py function can use.
One function, convert_train_keys_to_txt, makes a .txt file that contains the .mat filenames in train_keys.mat.
The other function, make_data_from_matlab_spects, uses that training_filenames.txt file to create a Python dictionary
 containing the spectrograms and labeled timebin vectors, and some associated metadata. 
 This dictionary has the same format as the dictionary the main.py function uses when cnn_bilstm.utils generates 
 the data directly from the .cbin files.
/Users/yarden/davids_fork_of_tf_sylseg $ activate learn_curve
(learn_curve) /Users/yarden/davids_fork_of_tf_sylseg $ ipython
[0] import cnn_bilstm
[1] cd directory_with_mat_and_train_keys
[2] cnn_bilstm.mat_utils.convert_train_keys_to_txt('.', 'training_filenames)
[3] cnn_bilstm.mat_tuils.make_data_from_matlab_spects('.', 'training_filenames', 'train_data_dict')