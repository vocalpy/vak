# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0a9]
### Added
- add helper function to TestLearncurve that multiple unit tests can use to assert all outputs 
  were generated. Now being used to make sure bug fixed in 0.1.0a8 stays fixed.
- error checking in cli that raises ValueError when cli command is `learncurve` and the option
  'results_dir_made_by_main_script' is already defined in [OUTPUT] section, since running
  'learncurve' would overwrite it. 

### Changed
- change name of `vak.cli.make_data` to `vak.cli.prep`

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
