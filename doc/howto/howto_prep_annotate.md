(howto-prep-annotate)=
# How do I prepare datasets of annotated vocalizations for use with vak?

You as a researcher want to automate annotation of 
your dataset of vocalizations, 
using a neural network model. 
To do that, you need to prepare a machine learning dataset 
that is used specifically for training a model. 
And of course you need to prepare a dataset 
of unannotated data so that you can 
predict annotations with a trained model.
You may also want to split your annotated data 
into additional subsets, 
such as *validation* and *test* sets.

```{seealso}
For definitions of these terms, please see 
<https://en.wikipedia.org/wiki/Training,_validation,_and_test_data_sets>
```

This page documents how you prepare datasets 
of annotated vocalizations for use with vak, 
what kind of files go into these datasets, 
and how you might need to name those files, 
depending on what kind of annotations you use.
Below, when we use the term "dataset", 
we mean a dataset for training a machine learning model 
or for prediction, 
unless otherwise specified. 
This lets us avoid repeating the noun stack 
"machine learning dataset" 
to distinguish a dataset prepared for use with vak 
from the larger datasets you have a researcher 
for your actual experiments.

## Calling `vak prep`

Whenever we prepare a dataset for use with vak, 
we call `vak prep` at the command line.
To determine the purpose of the dataset,
i.e., training or prediction, 
vak looks for another section in the configuration file,
e.g. `[TRAIN]` or `[PREDICT]`.
The options in that sections and their values 
determine how the datasets are prepared. 

## What kinds of files are required
To train and evaluate models for annotating vocalizations, 
we need two types of files:
1. A set of annotations, that may be in on ore more files
2. The files annotated by those annotations; we use the generic term "annotated files"
   since they could be audio files or array files containing spectrograms.

## How does vak know I want to prepare a dataset with annotations?
To indicate that vak should include annotations in a 
dataset that it prepares, 
you specify an annotation format in the `[PREP]` section 
of the .toml configuration file.

```toml
[PREP]
data_dir = "~/some/place/on/my/computer"
annot_format = "textgrid"
```

Notice the snippet above also shows a `data_dir` option. 
In most cases, vak assumes the annotation files will be found 
in the data dir.
The one exception is when you specify a separate annotation file 
with the `annot_file` option, 
as explained below in {ref}`one-annot-file-multiple-annotated`.

## How do I know which annotation formats I can use?

To work with annotation formats, vak uses another tool, 
[crowsetta](https://crowsetta.readthedocs.io/en/latest/). 
You can discover which formats are built into crowsetta  
by starting the Python interpreter and then 
running the following code snippet in your environment.

```python
import crowsetta
crowsetta.formats.as_list()
```

For more detail on the annotation formats 
built into crowsetta, please see the documentation:
<https://crowsetta.readthedocs.io/en/latest/>

```{admonition} Working with custom annotation formats
:class: note

If your annotations are in a format 
that is not built into crowsetta, 
please see the how-to on converting them to 
formats that vak can work with: 
{ref}`howto-user-annot`
```

(which-annotations-go-with-which-annotated)=
## How does vak know which annotations go with which annotated files?
The way that vak matches annotations with the files they annotate 
depends on your annotation format. 
There are two ways that annotation files can map to the files they annotate.

(one-annot-file-multiple-annotated)=
### One annotation file, multiple annotated files
The first way that annotation files map to the files they annotate is one-to-many: 
a single annotation file contains annotations for multiple annotated files.
In this case, vak "knows" how the files are related to each other, 
without you needing to do anything, 
because the annotation format by definition will need to specify 
in the annotations themselves which annotation 
corresponds to which file.
This is true, for example, for the 
[`'generic-seq'`](https://crowsetta.readthedocs.io/en/latest/formats/seq/generic-seq.html#generic-seq) 
annotation built into crowsetta.
So in this case you don't need to do anything, 
as long as vak can already parse your annotation format 
with crowsetta.
(If your format is not built in, 
please see this how-to: {ref}`howto-user-annot`).

You tell vak that there is only a single annotation file 
by setting a value for the `annot_file` option in the 
`[PREP]` section of the configuration file.

```toml
[PREP]
annot_format = "generic-seq"
annot_file = "./path/to/mouse1-annotations.csv"
```

If you don't set this option, vak assumes 
there is a one-to-one mapping from annotation file 
to annotated file, 
as we explain next.

(one-annot-file-one-annotated-file)=
### One annotation file per annotated file
The second way annotation files can map to annotated files 
is a one-to-one relationship.
This is true for many annotation formats, 
and it is the default.
All you need to do is specify the `annot_format` 
in the `[PREP]` section of the configuration file 
as described above.
When the annotation format makes use of a 
one-annotation-file-per-annotated-file approach, 
there are two ways that vak tries to map 
the annotations to the annotated files. 

The first is the simplest: 
vak simply replaces the extension of the annotation file 
with the extension of the file that it annotates.
In other words, 
if you have specified `audio_format = "wav"` 
in the `[PREP]` section of the configuration file, 
vak will change `bat1-day1-111403.textgrid` to 
`bat1-day1-111403.wav` and then try to match this 
with one of the annotated files 
that are found in `data_dir`.

The second way vak tries to match annotation files 
and annotated files is slightly more complicated.
It assumes that you have named your annotated files 
so that they contain the name of the original audio 
file. This is of course true for the audio files themselves
(the file `bat1-day1-111403.wav` contains its own name, and nothing else). 
It is also true for array files generated by vak 
that contain spectrograms 
(for example, `bat1-day1-111403.wav.npz`). 

In the past this convention led to subtle bugs
(see [#525](https://github.com/vocalpy/vak/issues/525)),
and for that reason the first way of 
mapping from annotation files to annotated files was added
(see [#563](https://github.com/vocalpy/vak/issues/525#issuecomment-1217408650)),
to avoid those bugs 
and to provide more flexibility for users.
There are two reasons you may want to use 
this second approach though. 
The first is that you want to use the filename to capture metadata, 
e.g., you want it to be clear from the name of your 
spectrogram files which audio file they were derived from. 
The second is that you need to have other files with the same extension 
in the same directory as your raw data.
For example, you have annotations in .csv files, 
but you also want to generate another .csv file 
for each audio file that contains acoustic features you extract 
with some other script you wrote. 
In this case, you can name an annotation file 
so that it includes the audio file, 
`bat1-day1-111403.wav.csv`, 
and name the file with acoustic features something like 
`bat1-day1-111403.ftr.csv`. 
This allows both to peacefully coexist in the same directory, 
and allows you to find all the annotation files 
by looking for `*.wav.csv` (with a wildcard)
---this is basically what vak does---
and to find all the feature files 
by looking for `*.ftr.csv`.
