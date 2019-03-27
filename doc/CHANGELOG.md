# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
