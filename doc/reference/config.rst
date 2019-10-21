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
1. `labelset`
Type str, the set of labels that correspond to annotated segments
that a network should learn to segment and classify. Note that
segments that are not annotated, e.g. silent gaps between songbird
syllables, then `vak` will assign a dummy label to those segments
-- you don't have to give them a label here.

.. code-block:: console

    labelset = iabcdefghjk

2. `data_dir`
Type str, path to directory with audio files from which to make dataset

.. code-block:: console

    data_dir = ./tests/test_data/cbins/gy6or6/032312

3. `total_train_set_duration`
Type int, total duration of training set, in seconds.
Training subsets of shorter duration will be drawn from this set.

.. code-block:: console

    total_train_set_duration = 50

4. `validation_set_duration`

.. code-block:: console

    validation_set_duration = 15

5. `test_set_duration`

.. code-block:: console

    test_set_duration = 30

6. `output_dir`

.. code-block:: console

    output_dir = ./tests/test_data/vds/

7. `audio_format`

.. code-block:: console

    audio_format = cbin

8. `annot_format`

.. code-block:: console

    annot_format = notmat

TRAIN
-----
1. `train_data_path`

Type str, path to training data

.. code-block:: console

    train_data_path = /some/path/here


2. `val_data_path`

Type str, path to validation dat

.. code-block:: console

    val_data_path = /some/path/here


3. `test_data_path`
Type str, path to test data

.. code-block:: console

    test_data_path = /some/path/here

4. `normalize_spectrograms`
Type bool, whether to normalize spectrograms.

.. code-block:: console

    normalize_spectrograms = Yes

5. `train_set_durs`
list of comma-separated integers
Duration of subsets of training data used for learning curve

.. code-block:: console

    train_set_durs = 4, 6


6. num_epochs

.. code-block:: console

    num_epochs = 2

7. val_error_step
step/epoch at which to estimate accuracy using validation set.
Default is None, in which case no validation is done.

.. code-block:: console

    val_error_step = 1

8. checkpoint_step = 1
step/epoch at which to save to checkpoint file.
Default is None, in which case checkpoint is only saved at the last epoch.

9. save

.. code-block:: console

    save_only_single_checkpoint_file = True

10.

.. code-block:: console

    patience = None

number of epochs to wait without the error dropping before stopping the
training. Default is None, in which case training continues for num_epochs

11. replicates

.. code-block:: console

    replicates = 2

12. networks

.. code-block:: console

    networks = TweetyNet

