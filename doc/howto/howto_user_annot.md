(howto-user-annot)=

# How do I use my own vocal annotation format?

To load annotation formats,
`vak` depends on a Python tool,
`crowsetta` (<https://crowsetta.readthedocs.io/en/latest/>).
This tool reads formats that can be represented 
as a sequence of segments, where each segment 
has an onset time, offset time, and a label.
It has built-in support for similar sequence-like formats 
saved by apps like [Praat](http://www.fon.hum.uva.nl/praat/manual/Intro_7__Annotation.html)
or [Audacity](https://manual.audacityteam.org/man/creating_and_selecting_labels.html).

If you have a format that is not currently supported
by `crowsetta`, you can still work with your annotations
by converting them to a generic format that `crowsetta` calls `'csv'`.

There are basically two steps to converting your format to `csv`,
described below.

## Step-by-step

1. Write a Python script that loads the onsets, offsets, and labels
   from your format, and then uses that data to create the `Annotation`s and
   `Sequence`s that `crowsetta` uses to convert between formats.

   :::{note}
   For examples, please see any of the modules for built-in functions
   in the `crowsetta` library.

   E.g., the `notmat` module:
   <https://github.com/NickleDave/crowsetta/blob/main/src/crowsetta/notmat.py>

   That module parses annotations from this dataset:
   <https://figshare.com/articles/dataset/Bengalese_Finch_song_repository/4805749>
   :::

2. Then save your `Annotation`s---converted to the generic
   `crowsetta` format---in a `.csv` file, using the `crowsetta.csv` functions.
   There is a convenience function `crowsetta.csv.annot2csv` that you can use
   if you have already written a function that returns `crowsetta.Annotation`s.
   Again, see examples in the built-in format modules.

   :::{note}
   The one key difference between built-in formats is that,
   when you create your `.csv` file, you need to specify
   the `annot_path` as the path to the `.csv` file itself.
   E.g., if you are saving your annotations in a `.csv` file
   named `bat1_converted.csv`, then the value for every cell in
   the `annot_path` column of your `.csv` should be
   also be `bat1_converted.csv`.

   It is counterintuitive to have the `.csv` refer to itself,
   but this workaround prevents `vak` from trying to open
   the original annotation files.
   :::

Here is a script that carries out steps one and two.
This script can be run on the example 
{download}`data <https://ndownloader.figshare.com/files/9537229>` 
used for training a model in the tutorial {ref}`autoannotate`.
```python
import pathlib

import numpy as np
import scipy.io

import crowsetta

data_dir = pathlib.Path('~/Documents/data/gy6or6/032312').expanduser()  # ``expanduser`` for '~' 
annot_path = sorted(data_dir.glob('*.not.mat'))

# ---- step 1. convert to ``Annotation``s with ``Sequence``s
annot = []
for a_notmat in annot_path:
    notmat_dict = scipy.io.loadmat(a_notmat, squeeze_me=True)
    # in .not.mat files saved by evsonganaly,
    # onsets and offsets are in units of ms, have to convert to s
    onsets_s = notmat_dict['onsets'] / 1000
    offsets_s = notmat_dict['offsets'] / 1000

    audio_pathname = str(a_notmat).replace('.not.mat', '')

    notmat_seq = crowsetta.Sequence.from_keyword(labels=np.asarray(list(notmat_dict['labels'])),
                                                 onsets_s=onsets_s,
                                                 offsets_s=offsets_s)
    annot.append(
        crowsetta.Annotation(annot_path=a_notmat, audio_path=audio_pathname, seq=notmat_seq)
    )

# ---- step 2. save as a .csv
crowsetta.csv.annot2csv(annot, csv_filename='data/annot/gy6or6.032212.annot.csv')
```

## Using the `.csv` file

If you have written a script that saves all your annotations 
in a single .csv file as described above, 
then you need to tell `vak` to use that file.
To do so, you add the `annot_file` option in the `[PREP]` section 
of your .toml configuration file, as shown below:

```{code-block} toml
:emphasize-lines: 6
[PREP]
data_dir = "~/Documents/data/vocal/BFSongRepo-test-csv-format/gy6or6/032212"
output_dir = "./data/prep/train"
audio_format = "cbin"
annot_format = "csv"
annot_file = "./data/annot/gy6or6.032212.annot.csv"
labelset = "iabcdefghjk"
train_dur = 50
val_dur = 15
```
