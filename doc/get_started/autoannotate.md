(autoannotate)=

# Automated Annotation

`vak` lets you automate annotation of vocalizations with neural networks.
This tutorial walks you through how you would do that, using an example dataset.
When we say annotation, we mean the the kind produced by a software tool
that researchers studying speech and animal vocalizations use,
like [Praat](http://www.fon.hum.uva.nl/praat/manual/Intro_7__Annotation.html)
or [Audacity](https://manual.audacityteam.org/man/creating_and_selecting_labels.html).
Typically the annotation consists of a file that specifies segments defined by their onsets, offsets, and labels.
Below is an example of some annotated Bengalese finch song, which is what we'll use for the tutorial.

```{image} ../images/annotation_example_for_tutorial.png
:align: center
:alt: spectrogram of Bengalese finch song with amplitude plotted underneath, divided
:  into segments labeled with letters
:scale: 50 %
```

:::{hint}
`vak` has built-in support for widely-used annotation formats.
Even if your data is not annotated with one of these formats, 
you can use `vak` by converting your annotations to a simple `.csv` format 
that is easy to create with Python libraries like `pandas`.
For more information, please see:  
{ref}`howto-user-annot`
:::

The tutorial is aimed at beginners: you don't need to know how to code.
To work with `vak` you will use simple configuration files that you run from the command line.
If you're not sure what is meant by "configuration file" or "command line",
don't worry, it will all be explained in the following sections.

## Set-up

Before going through this tutorial, you'll need to:

1. Have `vak` installed (following these {ref}`instructions <installation>`).
2. Have a text editor to change a few options in the configuration files
   such as [sublime](https://www.sublimetext.com/), [gedit](https://wiki.gnome.org/Apps/Gedit),
   or [notepad++](https://notepad-plus-plus.org/)
3. Download example data from this dataset: <https://figshare.com/articles/Bengalese_Finch_song_repository/4805749>

   - one day of birdsong, for training data (click to download)  
     {download}`https://figshare.com/ndownloader/files/37509160`
   - another day, to use to predict annotations (click to download)
     {download}`https://figshare.com/ndownloader/files/37509172`
   - Be sure to extract the files from these archives! 
     Please use the program "tar" to extract the archives, 
     on either macOS/Linux or Windows.
     Using other programs like WinZIP on Windows 
     can corrupt the files when extracting them,
     causing confusing errors.
     Tar should be available on newer Windows systems
     (as described 
     [here](https://learn.microsoft.com/en-us/virtualization/community/team-blog/2017/20171219-tar-and-curl-come-to-windows)).
   - Alternatively you can copy the following command and then 
     paste it into a terminal to run a Python script 
     that will download and extract the files for you. 

     :::{eval-rst}
    
     .. tabs::
    
        .. code-tab:: shell macOS / Linux
    
           curl -sSL https://raw.githubusercontent.com/vocalpy/vak/main/src/scripts/download_autoannotate_data.py | python3 -
     
        .. code-tab:: shell Windows
    
           (Invoke-WebRequest -Uri https://raw.githubusercontent.com/vocalpy/vak/main/src/scripts/download_autoannotate_data.py -UseBasicParsing).Content | py -
     :::

4. Download the corresponding configuration files (click to download):
   {download}`gy6or6_train.toml <../toml/gy6or6_train.toml>`
   and {download}`gy6or6_predict.toml <../toml/gy6or6_predict.toml>`

## Overview

There are four steps to using `vak` to automate annotating vocalizations

1. {ref}`prepare a training dataset <prepare-training-dataset>`, from
   a small annotated dataset of vocalizations
2. {ref}`train a neural <train-neural-network>` network with that dataset
3. {ref}`prepare a prediction dataset <prepare-prediction-dataset>` of unannotated data
4. {ref}`use the trained network <use-trained-network>` to predict annotations for the prediction dataset

Before doing that, you'll need to know a little bit about working with the shell,
since that's the main way to work with `vak` without writing any code.
You will enter commands into the shell to run `vak`; this is called the
"command line interface". The next section introduces the command line.

## 0. Use of `vak` from the command line

To use the command-line interface to `vak` you will open a program on your computer
that has a name like "terminal", where you can run programs using the shell.
It will look something like this:

```{image} /images/terminalizer/vak-help.gif
```

Basically any time you run `vak`, what you type at the prompt
will have the following form:

```shell
vak command config.toml
```

where `command` will be an actual command, like `prep`, and `config.toml`
will be the name of an actual configuration file, that let you configure
how a command will run.

To see a list of available commands when you are at the command line,
you can say:

```shell
vak --help
```

The `.toml` files are set up so that each section corresponds to one
of the commands. For example, there is a section called `[PREP]` where you
configure how `vak prep` will run.
Each section consists of option-value pairs, i.e. names of option set to the values you assign them.
For example, here is the `[PREP]` section from the configuration file
downloaded for training.

```{literalinclude} ../toml/gy6or6_train.toml
:language: toml
:lines: 1-9
```

(The files are in `.toml` format;
for this tutorial we will explain
anything specific about that format
you might need to know.)

:::{topic} Why command line?
A strength of the shell is that it lets you write scripts, so that whatever
you do with data is (more) reproducible. That includes the things you'll do
with your data when you're telling `vak` how to use it to train a neural
network. In a machine learning context, you need to reproduce the same steps
when preparing the data you want to apply the trained network to, so you can
predict its annotation.

If you don't have experience with the shell, we
suggest working through this beginner-friendly tutorial from the Carpentries:

<https://swcarpentry.github.io/shell-novice/>

Although it might seem a bit daunting at first, you can actually work quite
efficiently in the shell once you get familiar with the cryptic commands.
There's only a handful you need on a regular basis.
:::

Now that you know how to call `vak` from the command line, we'll walk through the first example
of modifying a configuration file and then using it to `prep` a dataset.

(prepare-training-dataset)=

## 1. preparing a training dataset

To train a neural network how to predict annotations,
you'll need to tell `vak` where your dataset is.
Do this by opening up the `gy6or6_train.toml` file and changing the
value for the `data_dir` option in the `[PREP]` section to the
path to wherever you downloaded the training data on your computer.

The options you need to change in the configuration files
have a dummy value in capital letters
to help you pick them out, like so:

```{literalinclude} ../toml/gy6or6_train.toml
:language: toml
:lines: 1-3
```

Change the part of the path in capital letters to the actual location
on your computer:

```toml
[PREP]
data_dir = "/home/users/You/Data/vak_tutorial_data/032212"
```

:::{note}
Notice that paths are enclosed in quotes; this is required
for paths or any other string (text) in a `toml` file. If you
get an error message about the `toml` file, check that
you have put quotes around the paths.
:::

:::{note}
Note also that you can write paths with just forward slashes,
even on Windows platforms! If you are on Windows,
you might be used to writing paths in Python with two
backwards slashes, like so: `'C:\\Users\\Me\\Data'`,
or placing an `r` in front of text strings representing paths, like
`r'C:\Users\Me\Data'`.
To make paths easier to type and read, we work with them
using the `pathlib` library:
<https://realpython.com/python-pathlib/>.
:::

There is one other option you need to change, `output_dir`
that tells `vak` where to save the file it creates that contains information about the dataset.

```toml
output_dir = "/home/users/You/Data/vak_tutorial_data/vak/prep/train"
```

Make sure that this a directory that already exists on your computer,
or create the directory using the File Explorer or the `mkdir` command from the command-line.

After you have changed these two options (we'll ignore the others for now),
you can run the command in the terminal that prepares datasets:

```shell
vak prep gy6or6_train.toml
```

Notice that the command has the structure we described above, `vak command config.toml` .

When you run `prep`, `vak` converts the data from `data_dir` into a special dataset file, and then
automatically adds the path to that file to the `[TRAIN]` section of the `config.toml` file, as the option
`csv_path`.

You have now prepared a dataset for training a model!  
You'll probably have more questions about 
how to do this later, 
when you start to work with your own data. 
When that time comes, please see the how-to page: 
{ref}`howto-prep-annotate`.
For now, let's move on to training a neural network with this dataset.

(train-neural-network)=
## 2. training a neural network

Now that you've prepared the dataset, you can train a neural network with it.
In this example we will train `TweetyNet`,
a neural network architecture that annotates vocalizations
(see: <https://github.com/yardencsGitHub/tweetynet> ).
Please make sure you have installed it following the steps in
{ref}`install-tweetynet` in {ref}`installation`.

Before we start training, there is one option you have to change in the `[TRAIN]` section
of the `config.toml` file, `root_results_dir`,
which tells `vak` where to save the files it creates during training.
It's important that you specify this option, so you know
where to find those files when we need them below.

```toml
root_results_dir = "/home/users/You/Data/vak_tutorial_data/vak/train/results"
```

Here it's fine to use the same directory you created before, or make a new one if you prepare to keep the
training data and the files from training the neural network separate.
`vak` will make a new directory inside of `root_results_dir` to save the files related to training
every time that you run the `train` command.

:::{note}
If you are not using a computer with a specialized GPU for training neural networks,
you'll need to change one more option in the .toml configuration file.
Please change the value for the option `device` in the `[TRAIN]` section from
`cuda` to `cpu`, to avoid getting an error about "CUDA not available".
Using a GPU can speed up training, but in practice we find it is quite possible
to train models for annotation on a CPU,
with training times ranging from a couple hours to overnight.
:::

To train a neural network, you simply run:

```shell
vak train gy6o6_train.toml
```

You will see output to the console as the network trains. The options in the `[TRAIN]` section of
the `config.toml` file tell `vak` to train until the error (measured on a separate "validation" set)
has not improved for four epochs (an epoch is one iteration through the entire training data).
If you let `vak` train until then, it will go through roughly ten epochs (~2 hours on an Ubuntu machine with
an NVIDIA 1080 Ti GPU).

You can also just stop after one epoch if you want to go through the rest of the tutorial. The `[TRAIN]` section
options also specify that `vak` should save a "checkpoint" every epoch, and we need to load our trained network
from that checkpoint later when we predict annotations for new data.

(prepare-prediction-dataset)=

## 3. preparing a prediction dataset

Next you'll prepare a dataset for prediction. The dataset we downloaded has annotation files with it,
but for the sake of this tutorial we'll pretend that they're not annotated, and we instead want to
predict the annotation using our trained network.
Here we'll use the other configuration file you downloaded above, `gy6or6_predict.toml`.
We use a separate file with a `[PREDICT]` section in it instead of a `[TRAIN]` section, so that
`vak` knows the dataset it's going to prepare will be for prediction--i.e., it figures this out
from the name of the section present in the file.

Just like before, you're going to modify the `data_dir` option of the
`[PREP]` section of the `config.toml` file.
This time you'll change it to the path to the directory with the other day of data we downloaded.

```toml
[PREP]
data_dir = "/home/users/You/Data/vak_tutorial_data/032312"
```

And again, you'll need to change the `output_dir` option
to tell `vak` where to save the file it creates that contains information about the dataset.

```toml
output_dir = "/home/users/You/Data/vak_tutorial_data/vak_output"
```

This part is the same as before too: after you change these options,
you'll run the `prep` command to prepare the dataset for prediction:

```shell
vak prep gy6or6_predict.toml
```

As you might guess from last time, `vak` will make files for the dataset and a .csv file that points to those,
and then add the path to that file as the option `csv_path` in the `[PREDICT]` section of the `config.toml` file.

(use-trained-network)=

## 4. using a trained network to predict annotations

Finally you will use the trained network to predict annotations.
This is the part that requires you to find paths to files saved by `vak`.

There's three you need. All three will be in the `results` directory
created by `vak` when you ran `train`. If you replaced the dummy path in
capital letters in the config file, but kept the rest of the path,
then this will be a location with a name like
`/PATH/TO/DATA/vak/train/results/results_{timestamp}`,
where `PATH/TO/DATA/` will be replaced with a path on your machine,
and where `{timestamp}` is an actual time in the format `yymmdd_HHMMSS`
(year-month-day hour-minute-second).

The first path you need is the `checkpoint_path`. This is the full
path, including filename, to the file that contains the weights (also known as parameters)
of the trained neural network, saved by `vak`.
There will be a directory inside the `results_{timestamp}` directory
with the name of the trained model, `TweetyNet`,
and inside that sits a `checkpoints` directory that has the actual file you want.
Typically there will be two checkpoint files, one named just `checkpoint.pt` that is
saved intermittently as a backup,
and another that is saved only when accuracy on the
validation set improves, named `max-val-acc-checkpoint.pt`.
If you were to use the `max-val-acc-checkpoint.pt` then the path would end
with `TweetyNet/checkpoints/max-val-acc-checkpoint.pt`.

```toml
checkpoint_path = "/home/users/You/Data/vak_tutorial_data/vak_output/results_{timestamp}/TweetyNet/checkpoints/max-val-acc-checkpoint.pt"
```

In some cases, a `max-val-acc-checkpoint.pt` may not get saved;
this depends on the options for training and non-deterministic factors like
the randomly initialized weights of the network.
For the purposes of completing this tutorial, using either checkpoint is fine.

The second path you want is the one to the file containing the `labelmap`.
The `labelmap` is a Python
dictionary that maps the labels from your annotation to a set of consecutive integers, which
are the outputs the neural network learns to predict during training. It is saved in a `.json`
file in the root `results_{timestamp}` directory.

```toml
spect_scaler = "/home/users/You/Data/vak_tutorial_data/vak_output/results_{timestamp}/labelmap.json"
```

The third and last path you need is the path to the file containing a saved `spect_scaler`.
The `SpectScaler` represents a transform
applied to the data that helps when training the neural network.
You need to apply the same transform to the new
data for which you are predicting labels--otherwise the accuracy will be impaired.
Note that the file does not have an extension. (In case you are curious,
it's a pickled Python object saved by the `joblib` library.)
This file will also be found in the root `results_{timestamp}` directory.

```toml
spect_scaler = "/home/users/You/Data/vak_tutorial_data/vak_output/results_{timestamp}/SpectScaler"
```

After adding the paths to these files generated during training,
you can specify an `output_dir` where the predicted annotations are saved.
The annotations are saved as a .csv file created by a separate software tool
for dealing with annotations, `crowsetta`. You can also specify the name
of this .csv file. For this tutorial, you can modify both so that
they point to the place where `prep` put the dataset it created for
predictions, just to have everything in one place.

```toml
output_dir = "/home/users/You/Data/vak_tutorial_data/vak/prep/predict"
annot_csv_filename = "gy6or6.032312.annot.csv"
```

:::{note}
Here, just as above for training, if you're not using a computer with a GPU,
you'll want to change the option `device` in the `[PREDICT]` section
of the .toml configuration file from `cuda` to `cpu`.
:::

Finally, after adding these paths,
you can run the `predict` command to generate annotation files from the labels
predicted by the trained neural network.

```shell
vak predict gy6or6_predict.toml
```

That's it! With those four simple steps you can train neural networks and then use the
trained networks to predict annotations for vocalizations.
