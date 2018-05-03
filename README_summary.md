# output of summary file

The `summary.py` script is run after the `learn_curves.py` script.  
Before running `summary.py`, add the `results_dir_made_by_main_script` option 
to the [OUTPUT] section of the `config.ini` file. This should be the full path
to the results directory that was created by the `learn_curves.py` script.
After adding this option to [OUTPUT], you run `summary.py` from the command line
 with the `config.ini` as an argument, like so:
 
 `(virtual-env) $ CUDA_VISIBLE_DEVICES=0 python
  ./cnn-bilstm/summary.py ./configs/config.ini`

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