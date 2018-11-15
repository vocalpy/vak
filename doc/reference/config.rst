=====================
config.ini files spec
=====================


Valid Sections
==============
Following is the set of valid section names:
{DATA, TRAIN, SPECTROGRAM, PREDICT, OUTPUT}.
In addition, whose name is the name of a class
representing a neural network that subclasses the
AbstractSongdeckClass, e.g., "TweetyNet"
and whose options define hyperparameters for that network.
More detail below.


Valid Options by Section
========================
DATA
-----

TRAIN
-----
1. `train_data_path`

Type str, path to training data

```
train_data_path = /some/path/here
```

2. `val_data_path`

Type str, path to validation dat

```
val_data_path = /some/path/here
```

3. `test_data_path`
Type str, path to test data

```
test_data_path = /some/path/here
```

4. `normalize_spectrograms`
Type bool, whether to normalize spectrograms.

```
normalize_spectrograms = Yes
```

5. `train_set_durs`
list of comma-separated integers
Duration of subsets of training data used for learning curve

```
train_set_durs = 4, 6
```

6. num_epochs

 = 2

7. val_error_step
step/epoch at which to estimate accuracy using validation set.
Default is None, in which case no validation is done.

 = 1

8. checkpoint_step = 1
step/epoch at which to save to checkpoint file.
Default is None, in which case checkpoint is only saved at the last epoch.

9. save_only_single_checkpoint_file = True

10. patience = None
number of epochs to wait without the error dropping before stopping the
training. Default is None, in which case training continues for num_epochs

11. replicates = 2

12. networks = TweetyNet

