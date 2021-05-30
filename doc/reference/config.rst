======================
config.toml files spec
======================

Valid Sections
==============
Following is the set of valid section names:
{PREP, SPECT_PARAMS, DATALOADER, TRAIN, PREDICT, LEARNCURVE}.
In addition, a section is valid whose name is the name of a class
representing a neural network that subclasses the
``vak``, e.g., "TweetyNet"
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

.. code-block:: toml

    labelset = "iabcdefghjk"

2. `data_dir`
Type str, path to directory with audio files from which to make dataset

.. code-block:: toml

    data_dir = "./tests/test_data/cbins/gy6or6/032312"

3. `traindur`
Type int, total duration of training set, in seconds.
Training subsets of shorter duration will be drawn from this set.

.. code-block:: toml

    total_train_set_duration = 50

4. `valdur`

.. code-block:: toml

    validation_set_duration = 15

5. `testdur`

.. code-block:: toml

    test_set_duration = 30

6. `output_dir`

.. code-block:: toml

    output_dir = "./tests/test_data/vds/"

7. `audio_format`

.. code-block:: toml

    audio_format = "cbin"

8. `annot_format`

.. code-block:: toml

    annot_format = "notmat"

TRAIN
-----
1. `csv_path`

Type str, path to .csv file that represents data for training,
i.e., `train` and `val` splits

.. code-block:: toml

    csv_path = "/some/path/here"

2. `normalize_spectrograms`
Type bool, whether to normalize spectrograms.

.. code-block:: toml

    normalize_spectrograms = true

3. num_epochs

.. code-block:: toml

    num_epochs = 2

4. val_step
step at which to estimate accuracy using validation set.
Default is None, in which case no validation is done.

.. code-block:: toml

    val_step = 500

5. ckpt_step
step at which to save to checkpoint file.
Default is None, in which case checkpoint is only saved at the last epoch.

.. code-block:: toml

    ckpt_step = 200

6. patience

.. code-block:: toml

    patience = 2

number of validation steps to wait without the error dropping before stopping the
training. Default is None, in which case training continues for num_epochs

7. networks

.. code-block:: toml

    networks = "TweetyNet"
