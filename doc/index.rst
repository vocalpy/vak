``vak`` वाच् *vāc*
=================
a neural network toolbox for animal vocalizations and bioacoustics
------------------------------------------------------------------

.. image:: images/song_with_colored_segments.png

``vak`` is a library for researchers studying animal vocalizations--such as
birdsong, bat calls, and even human speech--although it may be useful
to anyone working with bioacoustics data.
While there are many important reasons to study bioacoustics, the scope of ``vak``
is limited to questions related to **vocal learning**,
"the ability to modify acoustic and syntactic sounds,
acquire new sounds via imitation, and produce vocalizations" [Wikipedia]_.
Research questions related to vocal learning cut across a wide range of fields
including neuroscience, phsyiology, molecular biology, genomics, ecology, and evolution [Wir2019]_.

`vak` has two main goals:

1. make it easier for researchers studying animal vocalizations to
   apply neural network algorithms to their data
2. provide a common framework that will facilitate benchmarking neural
   network algorithms on tasks related to animal vocalizations

Currently the main use is automated **annotation** of vocalizations and other animal sounds,
using artificial neural networks.
By **annotation**, we mean something like the example of annotated birdsong shown below:

.. image:: images/annotation_example_for_tutorial.png

Please see links below for information on how to get started and how to use ``vak`` to
apply neural network models to your data.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   get_started/installation
   tutorial/tutorial
   howto/howto
   reference/reference
   explanations/explanations

.. [Wikipedia] https://en.wikipedia.org/wiki/Vocal_learning

.. [Wir2019] Wirthlin M, Chang EF, Knörnschild M, Krubitzer LA, Mello CV, Miller CT,
             Pfenning AR, Vernes SC, Tchernichovski O, Yartsev MM.
             A modular approach to vocal learning: disentangling the diversity of
             a complex behavioral trait. Neuron. 2019 Oct 9;104(1):87-99.
             https://www.sciencedirect.com/science/article/pii/S0896627319308396
