====================
Automated Annotation
====================

``vak`` lets you automate annotation of vocalizations with neural networks.
This tutorial walks you through how you would do that, using an example dataset.
When we say annotation, we mean the the kind produced by a software tool
that researchers studying speech and animal vocalizations use,
like `Praat <http://www.fon.hum.uva.nl/praat/manual/Intro_7__Annotation.html>`_
or `Audacity <https://manual.audacityteam.org/man/creating_and_selecting_labels.html>`_.
Typically the annotation consists of a file that specifies segments defined by their onsets, offsets, and labels.
Below is an example of some annotated Bengalese finch song, which is what we'll use for the tutorial.

.. image:: ../images/annotation_example_for_tutorial.png
   :align: center
   :scale: 50 %
   :alt: spectrogram of Bengalese finch song with amplitude plotted underneath, divided into segments labeled with letters

The tutorial is aimed at beginners: you don't need to know how to code.
To work with ``vak`` you will use simple configuration files that you run from the command line.
If you're not sure what is meant by "configuration file" or "command line",
don't worry, it will all be explained in the following sections.

------
Set-up
------

Before going through this tutorial, you'll need to:

1. have ``vak`` installed (following these :ref:`instructions <installation>`).
2. have a text editor to changing a few options in the configuration files
   such as `sublime <https://www.sublimetext.com/>`_, `gedit <https://wiki.gnome.org/Apps/Gedit>`_,
   or `notepad++ <https://notepad-plus-plus.org/>`_
3. download example data (from this dataset: https://figshare.com/articles/Bengalese_Finch_song_repository/4805749 )

 - one day of birdsong, for
   :download:`training data (click to download) <https://ndownloader.figshare.com/files/9537229>`
 - another day, for
   :download:`prediction data (click to download) <https://ndownloader.figshare.com/files/9537232>`
 - Be sure to extract the files from these archives!
   On Windows you can use programs like `WinRAR <https://www.rarlab.com/>`_
   or `WinZIP <https://www.winzip.com/win/en/tar-gz-file.html>`_,
   on mac you can double click the ``.tar.gz`` file, and on
   (Ubuntu) linux you can right-click and select the ``Extract to`` option.

4. download the corresponding configuration files (click to download):
   :download:`gy6or6_train.toml <../toml/gy6or6_train.toml>`
   and :download:`gy6or6_predict.toml <../toml/gy6or6_predict.toml>`

--------
Overview
--------

There are four steps to using ``vak`` to automate annotating vocalizations

1. :ref:`prepare a training dataset <prepare-training-dataset>`, from
   a small annotated dataset of vocalizations
2. :ref:`train a neural <train-neural-network>` network with that dataset
3. :ref:`prepare a prediction dataset <prepare-prediction-dataset>` of unannotated data
4. :ref:`use the trained network <use-trained-network>` to predict annotations for the prediction dataset

Before doing that, you'll need to be at least a little familiar with the command line,
since that's the main way to work with ``vak`` without writing any code.
The next section introduces the command line.

---------------------------------------
0. Use of ``vak`` from the command line
---------------------------------------

``vak`` uses a command-line interface, meaning you run it from the terminal,
also known as the shell.
Basically any time you run ``vak``, what you type at the prompt
will have the following form:

.. code-block:: console

   $ vak command config.toml

where ``command`` will be an actual command, like ``prep``, and ``config.toml``
will be the name of an actual configuration file, that let you configure
how a command will run.

To see a list of available commands when you are at the command line,
you can say:

.. code-block:: console

   $ vak --help

The ``.toml`` files are set up so that each section corresponds to one
of the commands. For example, there is a section called ``[PREP]`` where you
configure how ``vak prep`` will run.
Each section consists of option-value pairs, i.e. names of option set to the values you assign them.

.. literalinclude:: ../toml/gy6or6_train.toml
   :language: toml
   :lines: 1-9

(The files are in ``.toml`` format,
but for this tutorial and for working with ``vak`` you shouldn't really
need to know anything about that format.)

.. topic:: Why command line?

    A strength of the shell is that it lets you write scripts, so that whatever
    you do with data is (more) reproducible. That includes the things you'll do
    with your data when you're telling ``vak`` how to use it to train a neural
    network. In a machine learning context, you need to reproduce the same steps
    when preparing the data you want to apply the trained network to, so you can
    predict its annotation.

    If you don't have experience with the shell, we
    suggest working through this beginner-friendly tutorial from the Carpentries:

    https://swcarpentry.github.io/shell-novice/

    Although it might seem a bit daunting at first, you can actually work quite
    efficiently in the shell once you get familiar with the cryptic commands.
    There's only a handful you need on a regular basis.

Now that you know how to call ``vak`` from the command line, we'll walk through the first example
of modifying a configuration file and then using it to ``prep`` a dataset.

.. _prepare-training-dataset:

-------------------------------
1. preparing a training dataset
-------------------------------

To train a neural network how to predict annotations,
you'll need to tell ``vak`` where your dataset is.
Do this by opening up the ``gy6or6_train.toml`` file and changing the
value for the ``data_dir`` option in the ``[PREP]`` section to the
path to wherever you downloaded the training data on your computer.

.. code-block:: toml

   [PREP]
   data_dir = /home/users/You/Data/vak_tutorial_data/032212

There is one other option you need to change, ``output_dir``
that tells ``vak`` where to save the file it creates that contains information about the dataset.

.. code-block:: toml

   output_dir = /home/users/You/Data/vak_tutorial_data/vak_output

Make sure that this a directory that already exists on your computer,
or create the directory using the File Explorer or the ``mkdir`` command from the command-line.

After you have changed these two options (we'll ignore the others for now),
you can run the command in the terminal that prepares datasets:

.. code-block:: console

   $ vak prep gy6or6_train.toml

Notice that the command has the structure we described above, ``vak command config.toml`` \.

When you run ``prep``\ , ``vak`` converts the data from ``data_dir`` into a special dataset file, and then
automatically adds the path to that file to the ``[TRAIN]`` section of the ``config.toml`` file, as the option
``csv_path``.

.. _train-neural-network:

-------------------------------
2. training a neural network
-------------------------------

Now that you've prepared the dataset, you can train a neural network with it.

Before we start training, there is one option you have to change in the ``[TRAIN]`` section
of the ``config.toml`` file, ``root_results_dir``,
which tells ``vak`` where to save the files it creates during training.
It's important that you specify this option, so you know
where to find those files when we need them below.

.. code-block:: toml

   root_results_dir = /home/users/You/Data/vak_tutorial_data/vak_output

Here it's fine to use the same directory you created before, or make a new one if you prepare to keep the
training data and the files from training the neural network separate.
``vak`` will make a new directory inside of ``root_results_dir`` to save the files related to training
every time that you run the ``train`` command.

To train a neural network, you simply run:

.. code-block:: console

   $ vak train gy6o6_train.toml

In this example we are training ``TweetyNet``\ (https://github.com/yardencsGitHub/tweetynet_),
the default neural network architecture built into ``vak``.
You will see output to the console as the network trains. The options in the ``[TRAIN]`` section of
the ``config.toml`` file tell ``vak`` to train until the error (measured on a separate "validation" set)
has not improved for four epochs (an epoch is one iteration through the entire training data).
If you let ``vak`` train until then, it will go through roughly ten epochs (~2 hours on an Ubuntu machine with
an NVIDIA 1080 Ti GPU).

You can also just stop after one epoch if you want to go through the rest of the tutorial. The ``[TRAIN]`` section
options also specify that ``vak`` should save a "checkpoint" every epoch, and we need to load our trained network
from that checkpoint later when we predict annotations for new data.

.. _prepare-prediction-dataset:

---------------------------------
3. preparing a prediction dataset
---------------------------------

Next you'll prepare a dataset for predicting. The dataset we downloaded has annotation files with it,
but for the sake of this tutorial we'll pretend that they're not annotated, and we instead want to
predict the annotation using our trained network.
Here we'll use the other configuration file you downloaded above, ``gy6or6_predict.toml``.
We use a separate file with a ``[PREDICT]`` section in it instead of a ``[TRAIN]`` section, so that
``vak`` knows the dataset it's going to prepare will be for prediction--i.e., it figures this out
from the name of the section present in the file.

Just like before, you're going to modify the ``data_dir`` option of the
``[PREP]`` section of the ``config.toml`` file.
This time you'll change it to the path to the directory with the other day of data we downloaded.

.. code-block:: toml

   [PREP]
   data_dir = /home/users/You/Data/vak_tutorial_data/032312

And again, you'll need to change the ``output_dir`` option
to tell ``vak`` where to save the file it creates that contains information about the dataset.

.. code-block:: toml

   output_dir = /home/users/You/Data/vak_tutorial_data/vak_output

This part is the same as before too: after you change these options,
you'll run the ``prep`` command to prepare the dataset for prediction:

.. code-block:: console

   $ vak prep gy6or6_predict.toml

As you might guess from last time, ``vak`` will make files for the dataset and a .csv file that points to those,
and then add the path to that file as the option ``csv_path`` in the ``[PREDICT]`` section of the ``config.toml`` file.

.. _use-trained-network:

-------------------------------------------------
4. using a trained network to predict annotations
-------------------------------------------------

Finally you will use the trained network to predict annotations.
This is the part that requires you to find files saved by vak.

There's two you need. The first is the ``checkpoint_path``, the full
path including filename to the file that contains the weights (AKA parameters) of
the trained neural network, saved by ``vak``.

.. code-block:: toml

   checkpoint_path = /home/users/You/Data/vak_tutorial_data/vak_output/results_timestamp/TweetyNet/checkpoints/ckpt.pth

The second is the path to the file containing a saved ``spect_scaler``. The ``SpectScaler`` represents a transform
applied to the data that helps when training the neural network. You need to apply the same transform to the new
data for which you are predicting labels--otherwise the accuracy will be impaired.

.. code-block:: toml

   spect_scaler = /home/users/You/Data/vak_tutorial_data/vak_output/results_timestamp/TweetyNet/checkpoints/ckpt.pth


Finally you can run the ``predict`` command to generate annotation files from the labels predicted by the
trained neural network.

.. code-block:: console

   $ vak predict gy6or6_predict.toml

That's it! With those four simple steps you can train neural networks and then use the
trained networks to predict annotations for vocalizations.
