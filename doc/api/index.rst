.. _api:

API Reference
=============

.. automodule:: vak
.. currentmodule:: vak

This section documents the vak `API <https://en.wikipedia.org/wiki/API>`_.

Command Line Interface
----------------------

The :mod:`vak.cli` module implements the vak command line interface.

.. autosummary::
   :toctree: generated
   :template: module.rst

   cli.cli
   cli.eval
   cli.learncurve
   cli.predict
   cli.prep
   cli.train

Core
----

The :mod:`vak.core` module contains high-level functions called by the
commmand-line interface to prepare datasets, train and evaluate
models, and generate predictions from trained models.

.. autosummary::
   :toctree: generated
   :template: module.rst
   :recursive:

   core.learncurve
   core.eval
   core.predict
   core.prep
   core.train

Configuration files
-------------------

The :mod:`vak.config` module contains functions to parse
the TOML configuration files used with vak,
and dataclasses that represent tables from those files.

.. autosummary::
   :toctree: generated
   :template: module.rst
   :recursive:

   config.config
   config.dataloader
   config.eval
   config.learncurve
   config.model
   config.parse
   config.predict
   config.prep
   config.spect_params
   config.train
   config.validators

Datasets
--------
The :mod:`vak.datasets` module contains datasets built into vak.

.. autosummary::
   :toctree: generated
   :template: module.rst
   :recursive:

   datasets.seq.validators
   datasets.vocal_dataset
   datasets.window_dataset

Files
--------
The :mod:`vak.files` module contains helper functions for working with files.

.. autosummary::
   :toctree: generated
   :template: module.rst
   :recursive:

   files.files
   files.spect

Input-Output
------------
The :mod:`vak.io` module contains functions for generating datasets used by vak
from input files: audio files, spectograms, and/or annotation files.

.. autosummary::
   :toctree: generated
   :template: module.rst
   :recursive:

   io.audio
   io.dataframe
   io.spect

Metrics
-------
The :mod:`vak.metrics` module contains metrics used
when evaluating neural network model performance.

.. autosummary::
   :toctree: generated
   :template: module.rst
   :recursive:

   metrics.classification.classification
   metrics.classification.functional
   metrics.distance.distance
   metrics.distance.functional

Models
-------
The :mod:`vak.models` module contains models
built into vak, and functions for working with models:
declaring them via definition, registering them
as one of a family of models, getting a model instance
for training, predicting, etc.

.. autosummary::
   :toctree: generated
   :template: module.rst
   :recursive:

   models._api
   models.base
   models.decorator
   models.definition
   models.get
   models.teenytweetynet
   models.tweetynet
   models.windowed_frame_classification_model

Nets
----
The :mod:`vak.nets` module contains
neural network architectures built into vak.
All models include a neural network architecture
(along with an optimizer, loss function, and metrics).

.. autosummary::
   :toctree: generated
   :template: module.rst
   :recursive:

   nets.teenytweetynet
   nets.tweetynet

Neural Network Layers and Operations
------------------------------------
The :mod:`vak.nn` module contains
operations, layers, and other graph components
used in neural network architectures.

.. autosummary::
   :toctree: generated
   :template: module.rst
   :recursive:

   nn.loss
   nn.functional

Plotting
--------

Functions for plotting
that are built in to vak live in the :mod:`vak.plot` module.

.. autosummary::
   :toctree: generated
   :template: module.rst
   :recursive:

   plot.annot
   plot.learncurve
   plot.spect

Train-test-validation splits of datasets
----------------------------------------

The :mod:`vak.split` module contains functionality
for generating train-validation-test splits from
datasets. It is called by the :mod:`vak.prep` function
when running ``vak prep`` through the command line interface.

.. autosummary::
   :toctree: generated
   :template: module.rst
   :recursive:

   split.algorithms.bruteforce
   split.algorithms.validate
   split.split

Transforms
----------

The :mod:`vak.transforms` module contains transforms
that can be applied to input or output of neural networks,
i.e., for pre-processing or post-processing.

.. autosummary::
   :toctree: generated
   :template: module.rst
   :recursive:

   transforms.labeled_timebins.functional
   transforms.labeled_timebins.transforms
   transforms.functional
   transforms.transforms

Miscellaneous
-------------

.. autosummary::
   :toctree: generated
   :template: module.rst
   :recursive:

   annotation
   constants
   converters
   curvefit
   device
   labeled_timebins
   labels
   logging
   paths
   spect
   tensorboard
   timebins
   timenow
   trainer
   typing
   validators
