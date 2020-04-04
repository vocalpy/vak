# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [unreleased]
### Added
- add functions `format_from_df` and `from_df` to `vak.util.annotation`
  [#107](https://github.com/NickleDave/vak/pull/107) 
  + `vak.util.annotation.from_from_df` returns annotation format associated with a 
  dataset. Raises an error if more than one annotation format or if format is none.
  + `vak.util.annotation.from_df` function returns list of annotations 
  (i.e. `crowsetta.Annotation` instances), one corresponding to each row in the dataframe `df`.
    - encapsulates control flow logic for getting all labels from a dataset of 
      annotated vocalizations represented as a Pandas DataFrame
      + handles case where each vocalization has a separate annotation file 
      + and the case where all vocalizations have annotations in a single file
- `vak.util.labels.from_df` function [#103](https://github.com/NickleDave/vak/pull/103)
  + checks for single annotation type, load all annotations, and then get just labels from those
  + modified to use `util.annotation.from_df` and `vak.util.annotation.format_from_df` 
    in [#107](https://github.com/NickleDave/vak/pull/107)
- logic in `vak.cli.prep` that raises an informative error message when config.toml file specifies
  a duration for training set, but durations for validation and test sets are zero or None
  [#108](https://github.com/NickleDave/vak/pull/108)
  + since there's no functionality for making only one dataset of a specified dataset
- 3 transform classes, and `vak.transforms.util` module [#112](https://github.com/NickleDave/vak/pull/112)
  + with `get_defaults` function
    - encapsulates logic for building transforms, to make `train`, `predict` etc. less verbose
  + obeys DRY, avoid declaring the same utility transforms like to_floattensor and add_channel in 
    multiple functions 
- add `labelset_from_toml_value` to converters [#115](https://github.com/NickleDave/vak/pull/115)
  + casts any value for the `labelset` option in a .toml config file to a set of characters
    [#127](https://github.com/NickleDave/vak/pull/127)
  + uses `vak.util.general.range_str` so that user can specify 
    set of labels with a "range string", e.g. `range: 1-27, 29` [#115](https://github.com/NickleDave/vak/pull/115)
- add logging module in `vak.util` [#132](https://github.com/NickleDave/vak/pull/132)
- add converters and validators for dataset split durations [#143](https://github.com/NickleDave/vak/pull/143)
- add `logger` parameters to `io` sub-package functions, so they can use logger created by `cli` functions
  [#145](https://github.com/NickleDave/vak/pull/145)

### Changed
- rewrite `vak.util.dataset.has_unlabeled` to use `annotation.from_df` 
  [#107](https://github.com/NickleDave/vak/pull/107)
- use `TweetyNet` version 0.3.1 [#120](https://github.com/NickleDave/vak/pull/120)
  + so that `yarden2annot` function from `TweetyNet` will return annotation labels as string, not int
- rewrite `vak.util.annotation.source_annot_map` so that it maps annotations *to* source files, not 
  vice versa [#130](https://github.com/NickleDave/vak/pull/130)
  + more specifically, it no longer crashes if it can't map every annotation to a source file
  + instead it crashes if it can't map every source file to an annotation
- change `vak.annotation.from_df` to better handle single annotation files 
  [#131](https://github.com/NickleDave/vak/pull/131)
  + no longer crashes if the number of annotations from the file does not exactly match the number of source files
  + instead only requires there at least as many annotations as there are source files
- rewrite `vak.util.labels.from_df` to use `vak.util.annotation.from_df`
  [#131](https://github.com/NickleDave/vak/pull/131)
- rewrite `WindowDataset` to use `annotation.from_df` function [#113](https://github.com/NickleDave/vak/pull/113)
- change default value for util.general.timebin_dur_from_vec parameter n_decimals_trunc from 3 to 5
  [#136](https://github.com/NickleDave/vak/pull/136)
- rewrite + rename `splitalgos.validate.durs` [#143](https://github.com/NickleDave/vak/pull/143)
- parallelize validation of spectrogram files, so it's faster on large datasets 
  [#144](https://github.com/NickleDave/vak/pull/144)

### Fixed
- add missing `shuffle` option to [TRAIN] and [LEARNCURVE] sections in `valid.toml`
  [#109](https://github.com/NickleDave/vak/pull/109)
- bug that prevented filtering out vocalizations from a dataset when labels are present 
  in that vocalization that are not in the specified labelset [#118](https://github.com/NickleDave/vak/pull/118)
- fix logging for `vak.prep` command [#132](https://github.com/NickleDave/vak/pull/132)
- fix how dataset duration splits are validated [#143](https://github.com/NickleDave/vak/pull/143), 
  see issue [#140](https://github.com/NickleDave/vak/issues/140) for details.
- fix error due to calling a Path attribute on a string [#144](https://github.com/NickleDave/vak/pull/144)
  as identified in issue [#123](https://github.com/NickleDave/vak/issues/123)

### Removed
- remove `vak.util.general.safe_truncate` function, no longer used 
  [#137](https://github.com/NickleDave/vak/issues/137)
- remove redundant validation of split durations in `util.split` 
  [#143](https://github.com/NickleDave/vak/pull/143)

## [0.3.0a4]
### Added
- warning when user runs `vak prep` with config.toml file that has a `[PREDICT]` 
  section *and* a `labelset` option in the `[PREP]` section.
- better error handling when parsing a config.toml file fails
  + traceback now ends with clear message about error parsing .toml file, but still 
    includes information from `toml` exception

### Fixed
- tiny capitalization bug that broke configuration parsing

## [0.3.0a3]
### Fixed
- add missing sections and options to .toml file that is used to validate 
  user config.toml files, so that those options don't cause 
  invalid section / option errors 

## [0.3.0a2]
### Fixed
- `vak predict` command now works for command line

### Added
- [PREDICT] section now has `annot_format` option -- user can specify
  whatever format they want, doesn't have to be same as training data
- [PREDICT] section of config now has `to_format_kwargs` option, 
  that lets user specify keyword arguments to `crowsetta.Transcriber.to_format` 
  method for the annotation format of files made from predictions 

## [0.3.0a1]
### Fixed
- path in `PACKAGE_DATA` that captures 'valid.toml'

## [0.3.0a0]
### Removed
- `Dataset` class and related classes that were in `vak.dataset` sub-package
  + see `dataframe` module added below that replaces this abstraction
- dependency on `Tensorflow`
  + switch to `torch` because of consistent API, need to work with "mid-level" 
    abstractions, and preference for Python-first framework
- `core` sub-package
  + the idea is that the `cli` package should just implement all the logic that lets
    people who don't want to program use the main functionality
  + and if you do want to program, the rest of the library should facilitate that 
    *instead of* trying to do all the work for you
    - e.g. give someone w/basic coding skills friendly Python classes to work with
      when writing a torch-vernacular training script, instead of 
      giving them a giant `train` function with 3k arguments that no one will ever use
 - `AbstractVakModel` class -- gets replaced with `vak.Model` in `engine` sub-package, 
   see below
 
### Changed
- `dataset` sub-package becomes `io` sub-package ("input-output", like in `astropy`)
- use `torch` and `torchvision` in place of `tensorflow`  
- use `crowsetta` version 2.0
- switch to `toml` format for config files
  + more flexible than `ini` files, less code to maintain for parsing things that 
    don't fit into the `ini` format very well / not at all
- clean up `vak` package structure wherever possible: move many modules into 
  `util` sub-package

### Added
- `dataframe` module in `vak.io`
  + essentially, data path is audio --> spect --> dataframe --> .csv file that represents 
    a dataset
  + choose to use external libraries that are already well-maintained and established to 
    handle as much of the data processing as possible, i.e. `pandas` + `dask`, instead of
    trying to maintain a `Dataset` class that does all this work and deals with its own
    filetype
- `datasets` sub-package
  + uses `torch` and `torchvision` abstractions to represent datasets + dataloaders
- `transforms` sub-package
  + uses `torchvision` transform abstraction to deal with things like "normalizing" 
    spectrograms
- `engine` sub-package
  + with `Model` class that models should sub-class; helps encourage consistent API for models
- `metrics` sub-package
  + to compute things like accuracy
  + lays groundwork for an `ignite.metrics` / Keras-like functionality

## [0.2.2]
### Fixed
- add missing line break in `installation.rst`
  + needed to show the crucial line about how to install from `--channel nickledave`

## [0.2.1]
### Added
- recipe for `conda` build
- made build available on an Anaconda cloud channel

### Changed
- rewrote Installation page of docs
  + basically saying `conda` is required for install currently

## [0.2.0]
### Added
- `vak.core.learncurve.test_one_model` function that makes it easier to
  measure frame and syllable error, etc., on a single trained model
- add `move_spects` method to `Dataset` so an instance of a `Dataset` is not locked to a 
  particular location

### Changed
- single-source version
  + using the "Warehouse" approach from the PyPA page (thanks Donald Stufft)
    <https://packaging.python.org/guides/single-sourcing-package-version/#single-sourcing-the-version>
- rename `VocalizationDataset`, it's now just `Dataset` and is imported
  at top-level of package; both changes make code concise and reduce typing
  (and the `Vocalization` is implied anyway).

### Fixed
- syllable error rate calculated correctly for test data set by `vak.core.learning_curve.test`
- pin `crowsetta` version to 1.1.1 in setup.py
  + so that `pip` doesn't install version 2.0, which causes an error
- `predict` command in command-line interface now works

## [0.1.0]
### Added
- add helper function to TestLearncurve that multiple unit tests can use to assert all outputs 
  were generated. Now being used to make sure bug fixed in 0.1.0a8 stays fixed.
- error checking in cli that raises ValueError when cli command is `learncurve` and the option
  'results_dir_made_by_main_script' is already defined in [OUTPUT] section, since running
  'learncurve' would overwrite it. 
- `dataset` subpackage that houses `Dataset` and related classes that facilitate creating data sets for training neural networks from heterogeneous data: audio files, files of arrays containing spectrograms, different annotation types, etc.
  - also includes modules for handling each data source
    + e.g. `audio.to_spect` creates spectrograms from audio files
    + `spect.from_files` creates a `Dataset` from spectrogram files
- `core` sub-package that contains / will contain functions that do heavy lifting: `learning_curve`, `train`, `predict`
  + `learning_curve` is a sub-sub-module that does both `train` and `test` of models, instead of having a separate `learncurve` and `summary` function (i.e. train and test). Still will confuse some ML/AI people that this "learning curve" has a test data step but whatevs
  + `cli` sub-package calls / will call these functions and handle any command-line-interface specific logic
     (e.g. making changes to `config.ini` files)

### Changed
- change name of `vak.cli.make_data` to `vak.cli.prep`
- structure of `config.ini` file
  + now specify either `audio_format` or `spect_format` in `[DATA]` section
  + and `annot_format` for annotations
- refactor `utils` sub-package
  + move several functions from `data` and `general` into a `labels` module

### Removed
- remove unused options from command-line interface: `--glob`, `--txt`, `--dataset`
- `skip_files_with_labels_not_in_labelset` option
  + now happens whenever `labelset` is specified; if no `labelset` is given then no filtering is done
- `summary` command-line option, since `learncurve` now runs trains models and also tests them on separate data set
- `silent_label_gap` option, because `Dataset` class determines if a label for unlabeled segments between other segments is needed, and if so automatically assigns this a label of 0 when mapping user labels to consecutive integers
  + this way user does not have to think about it
  + and program doesn't have to keep track of a `labels_mapping` file that saves what user specified

## [0.1.0a8]
### Fixed
- Fix how main loop in `learncurve` re-loads indices for grabbing subsets of training data after 
  generating them, and do so in a way that still allows for re-using subsets from previous runs

## [0.1.0a7]
### Added
- `vak.cli.summary` has `save_transformed_data` parameter and `vak.cli` passed value from
  `config.data.save_transformed_data` as the argument when calling `vak.cli.summary`

### Changed
- `vak.cli.summary` only saves transformed train/test data if `save_transformed_data` is `True`
- move a test from tests/unit_tests/test_utils.py into tests/unit_tests/test_utils/test_data.py

### Removed
- `vak.cli.summary` no longer saves copy of test data in results directory

## [0.1.0a6]
### Added
- add test for utils.data.get_inds_for_dur

### Changed
- learncurve gets indices for all train data subsets before starting training

## [0.1.0a5]
### Added
- Use `attrs`-based classes to represent sections of config.ini files 
 
### Changed
- rewrite `vak.cli` so it can deal with state of config.ini files
  + e.g. doesn't throw an error if `train_data_path` not declared as an option in [TRAIN] when running `vak prep` 
(since training data won't exist yet, doesn't make sense to throw an error).

### Removed
- remove code about `freq_bins` in a couple of places, since the number of frequency bins 
  in spectrograms is now just determined programmatically 
  + `vak.config.data` no longer has `freq_bins` field in DataConfig namedtuple
  + `make_data` no longer adds `freq_bins` option to [DATA] section after making data sets

## [0.1.0a4]
### Fixed
- add missing 'save_transformed_data' option to Data config parsing

## [0.1.0a3]
### Changed
- checkpoints saved in individual directories by `learncurve` so they are more cleanly segregated,
e.g. if user wants to point to a specific checkpoint when calling `predict`
- calling `vak prep config.ini` will run `vak.cli.make_data` function
  + so to generate a learning curve, the three steps now are:
  ```bash
  vak prep config.ini
  vak learncurve config.ini
  vak summary config.ini
  ```

### Fixed
- `vak.cli.train` runs all the way through, passes basic "does not crash" test
- `vak.cli.predict` runs all the way through, passes basic "does not crash" test

## [0.1.0a2]
### Changed
- description in setup.py (matches README + Github)
- move command-line interface logic out of __main__.py, into cli/cli.py
- `make_data` and `learncurve` functions use `tqdm` for progress bars

### Fixed
- main() knows to look for `configfile` command-line argument (not `config`)
- `config` module expands user (on Linux/Mac) for (some) directory names

## [0.1.0a1]
First release; still in pre-release
### Changed
- name change from 'songdeck' to 'vak'
