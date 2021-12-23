.. _spect_file_format:

=======================
Spectrogram file format
=======================

File type
=========
``vak`` uses pre-computed files containing spectrograms.

For these files, it accepts two types, either ``.npz`` or ``.mat``.
``.npz`` is a ``numpy`` library format,
for a file that can contain multiple arrays.
``.mat`` is the Matlab data file format---many labs
have existing codebases that generate spectrograms using Matlab.
To work with one of these formats,
you will specify either ``npz`` or ``vak`` in the ``[PREP]`` section
of your ``.toml`` configuration file.

Conventions
===========
Regardless of whether they are ``.npz`` files or ``.mat`` files,
``vak`` expects any spectrogram files to obey the following conventions.

Content
-------
A spectrogram array files should contain (at least) four items.
Other arrays can be in the file, but they will be ignored.

1. The spectrogram, an *m x n* matrix
2. A vector of *m* frequency bins,
   where the value of each element is the frequency at the center of the bin
3. A vector of *n* time bins,
   where thevalue each element is the time at the center of the bin
4. A string path to the audio file from which the spectrogram was generated.

Array naming
------------
By convention each should be associated with a string key: 's', 'f', 't', and 'audio_path'.
If you are using Matlab then you will need to save your workspace variables with these names.

Spectrogram file naming
-----------------------
The *name* of each spectrogram file *must* be the **same**
as the name of the audio file it was created from, with the spectrogram file format added.
E.g., if your audio file is ``bird1.wav``, then the spectrogram file should be ``bird1.wav.npz``.
