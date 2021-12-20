.. _howto_user_annot:

============================================
How do I use my own vocal annotation format?
============================================

To load annotation formats,
``vak`` depends on a Python tool,
``crowsetta`` (https://crowsetta.readthedocs.io/en/latest/).

If you have a format that is not currently supported
by ``crowsetta``, you can still work with your annotations
by converting them to a format that tool calls ``csv``.

step-by-step
============

1. Write a Python script that loads the onsets, offsets, and labels
   from your format, and then uses that data to create the ``Annotation``\s and
  ``Sequence``\s that ``crowsetta`` uses to convert between formats.

   .. note::
      For examples, please see any of the modules for built-in functions
      in the ``crowsetta`` library.


      E.g., the ``notmat`` module:
      https://github.com/NickleDave/crowsetta/blob/main/src/crowsetta/notmat.py


      That module parses annotations from this dataset:
      https://figshare.com/articles/dataset/Bengalese_Finch_song_repository/4805749


2. Then save your ``Annotation``\s---converted to the generic
   ``crowsetta`` format---in a ``.csv`` file, using the ``crowsetta.csv`` functions.
   There is a convenience function ``crowsetta.csv.annot2csv`` that you can use
   if you have already written a function that returns ``crowsetta.Annotation``\s.
   Again, see examples in the built-in format modules.

    .. note::

       The one key difference between built-in formats is that,
       when you create your ``.csv`` file, you need to specify
       the ``annot_path`` as the path to the ``.csv`` file itself.
       E.g., if you are saving your annotations in a ``.csv`` file
       named ``bat1_converted.csv``, then the value for every cell in
       the ``annot_path`` column of your ``.csv`` should be
       also be ``bat1_converted.csv``.

       It is counterintuitive to have the ``.csv`` refer to itself,
       but this workaround prevents ``vak`` from trying to open
       the original annotation files.
