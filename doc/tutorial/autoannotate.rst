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

 - one day of birdsong, for :download:`training data <https://ndownloader.figshare.com/files/9537229>`
 - another day, for :download:`prediction data <https://ndownloader.figshare.com/files/9537232>`
 - Be sure to extract the files from these archives!
   On Windows you can use programs like `WinRAR <https://www.rarlab.com/>`_
   or `WinZIP <https://www.winzip.com/win/en/tar-gz-file.html>`_,
   on mac you can double click the ``.tar.gz`` file, and on
   (Ubuntu) linux you can right-click and select the ``Extract to`` option.

4. download the corresponding configuration files for
   :download:`training <../ini/gy6or6_train.ini>`
   and :download:`prediction <../ini/gy6or6_predict.ini>`

--------
Overview
--------

There are four steps to using ``vak`` to automate vocal annotations with neural networks

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
also known as the shell. If you don't have experience with the shell, we
suggest working through this beginner-friendly tutorial from the Carpentries:

https://swcarpentry.github.io/shell-novice/

Although it might seem a bit daunting at first, you can actually work quite
efficiently in the shell once you get familiar with the cryptic commands.
There's only a handful you need on a regular basis.

Why command line?
~~~~~~~~~~~~~~~~~

A strength of the shell is that it lets you write scripts, so that whatever
you do with data is (more) reproducible. That includes the things you'll do
with your data when you're telling ``vak`` how to use it to train a neural
network. In a machine learning context, you need to reproduce the same steps
when preparing the data you want to apply the trained network to, so you can
predict its annotation.

The ``vak`` command-line interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With those preliminary comments out of the way, we introduce the command-line
interface. Basically any time you run ``vak``, what you type at the prompt
will have the following form:

.. code-block:: console

   $ vak command config.ini

where ``command`` will be an actual command, like ``prep``, and ``config.ini``
will be an actual ``.ini`` file in which you specify the options for the different
commands. To see a list of available commands when you are at the command line,
you can say:

.. code-block:: console

   $ vak --help

The ``.ini`` files are set up so that each section corresponds to one
of the commands. For example, there is a section called ``[PREP]`` where you
specify options for preparing a dataset. In the next section we'll prepare
a dataset for training a neural network.

.. _prepare-training-dataset:

-------------------------------
1. preparing a training dataset
-------------------------------

To train a neural network how to predict annotations,
you'll need to tell ``vak`` where your dataset is.
Do this by opening up the `gy6or6_train.ini` file and changing the
value for the ``data_dir`` option in the ``[PREP]`` section to the
path to wherever you downloaded the training data on your computer.

.. code-block:: ini

   [PREP]
   data_dir = /home/users/You/Data/vak_tutorial_data/032212

There is one other option you need to change, ``output_dir``
that tells ``vak`` where to save the file it creates that contains information about the dataset.

.. code-block:: ini

   output_dir = /home/users/You/Data/vak_tutorial_data/vak_output

Make sure that this a directory that already exists on your computer,
or create the directory using the File Explorer or the ``mkdir`` command from the command-line.

After you have changed these two options (we'll ignore the others for now),
you can run the command in the terminal that prepares datasets:

.. code-block:: console

   $ vak prep gy6or6_train.ini

Notice that the command has the structure we described above, ``vak command config.ini`` \.

When you run ``prep``\ , ``vak`` converts the data from ``data_dir`` into a special dataset file, and then
automatically adds the path to that file to the ``[TRAIN]`` section of the config.ini file.

.. _train-neural-network:

-------------------------------
2. training a neural network
-------------------------------

Now that you've prepared the dataset, you can train a neural network with it.

Before we start training, there is one option you have to change in the ``[TRAIN]`` section
of the ``config.ini`` file, ``root_results_dir``,
which tells ``vak`` where to save the files it creates during training.
It's important that you specify this option, so you know
where to find those files when we need them below.

.. code-block:: ini

   root_results_dir = /home/users/You/Data/vak_tutorial_data/vak_output

Here it's fine to use the same directory you created before, or make a new one if you prepare to keep the
training data and the files from training the neural network separate.
``vak`` will make a new directory inside of ``root_results_dir`` to save the files related to training
every time that you run the ``train`` command.

To train a neural network, you simply run:

.. code-block:: console

   $ vak train gy6o6_train.ini

In this example we are training ``TweetyNet``\ (https://github.com/yardencsGitHub/tweetynet_),
the default neural network architecture built into ``vak``.
You will see output to the console as the network trains. The options in the ``[TRAIN]`` section of
the ``config.ini`` file tell ``vak`` to train until the error (measured on a separate "validation" set)
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
Just like before, you're going to modify the `data_dir` option of the ``[PREP]`` section of the ``config.ini`` file.
We use a separate ``config.ini`` file

.. code-block:: ini

   [PREP]
   data_dir = /home/users/You/Data/vak_tutorial_data/032212

.. _use-trained-network:

-------------------------------------------------
4. using a trained network to predict annotations
-------------------------------------------------

Finally you will use the
This is the part that