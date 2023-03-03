# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.8.1 -- 2023-03-02
### Fixed
- Fix transform that converts labeled timebins to segments
  so that it returns all `None`s when there are no segments 
  in the vector, either before or after applying any 
  post-processing transforms
  [#636](https://github.com/NickleDave/vak/pull/636).
  Bug introduced in
  [#621](https://github.com/NickleDave/vak/pull/621).
  Fixes [#634](https://github.com/NickleDave/vak/issues/634).

## 0.8.0 -- 2023-02-09
### Added
- Add options for how `audio.to_spect` calls `dask.bag`, 
  to help with memory issues when processing large files
  [#611](https://github.com/NickleDave/vak/pull/611).
  Fixes [#580](https://github.com/NickleDave/vak/issues/580).
- Add ability to run evaluation of models with and without post-processing 
  transforms. This is done by specifying an option `post_tfm_kwargs` in the 
  `[EVAL]` or `[LEARNCURVE]` sections of a .toml configuration file.
  If the option is not specified, then models are evaluated as they were
  previously, by converting the predicted label for each time bin 
  to a label for each continuous segment, represented as a string.
  If the option *is* specified, then the post-processing is applied 
  to the model predictions before converting to strings.
  Metrics are computed for outputs with *and* without post-processing,
  to be able to compare the two.
  [#621](https://github.com/NickleDave/vak/pull/621).
  Fixes [#472](https://github.com/NickleDave/vak/issues/472).
- `vak.core.eval` now logs computed evaluation metrics so they can be 
  quickly inspected in the terminal or log files before full analysis
  [#621](https://github.com/NickleDave/vak/pull/621).
  Fixes [#471](https://github.com/NickleDave/vak/issues/471).

### Changed
- Rewrite post-processing transforms applied to network outputs 
  as transforms, with functional and class implementations,
  to make it possible to compose these transforms, and more easily 
  evaluate model performance with and without them
  [#621](https://github.com/NickleDave/vak/pull/621).
  Fixes [#537](https://github.com/NickleDave/vak/issues/537).

## 0.7.0 -- 2022-11-23
### Added
- Add unit tests for `csv.has_unlabled`
  [#541](https://github.com/NickleDave/vak/pull/541).
  Fixes [#102](https://github.com/NickleDave/vak/issues/102).
- Add unit tests for `__main__`
  [#542](https://github.com/NickleDave/vak/pull/542).
  Fixes [#337](https://github.com/NickleDave/vak/issues/337).
- Add validation of `labels` argument to `vak.split.algorithms.brute_force`,
  to prevent conditions where algorithm can fail to converge 
  because of bad input 
  [#562](https://github.com/NickleDave/vak/pull/562).
  Fixes [#288](https://github.com/NickleDave/vak/issues/288).
- Add a "Frequently Asked Questions" page to the documentation, 
  and a page to the "Reference" section on file naming conventions
 [#564](https://github.com/NickleDave/vak/pull/564).
  Fixes [#524](https://github.com/NickleDave/vak/issues/524)
  and [#424](https://github.com/NickleDave/vak/issues/424).
- Add a new way for vak to map annotation files to annotated files 
  when preparing datasets, e.g. for training models. 
  For annotation formats that have one annotation file per 
  annotated file, vak can now recognize when
  the annotation files are named by removing the 
  annotated file extension (e.g., .wav or .npz) 
  and replacing it with the annotation format extension, 
  e.g. .txt or .csv. (Other ways of relating annotations 
  and annotated files are still valid, e.g. by including 
  the original source audio file in both filenames.)
  [#572](https://github.com/NickleDave/vak/pull/572).
  Fixes [#563](https://github.com/NickleDave/vak/issues/563).
- Have runs from command-line interface log version to logfile
  [#587](https://github.com/NickleDave/vak/pull/587).
  Fixes [#216](https://github.com/NickleDave/vak/issues/216).

### Changed
- Rewrite unit tests in `tests/test_cli/` to use mocks for `vak.core` functions
  [#544](https://github.com/NickleDave/vak/pull/544).
  Fixes [#543](https://github.com/NickleDave/vak/issues/543).
- It is now possible to load configuration files 
  and work with them programmatically even if the paths 
  they point to do not exist.
  The `core` functions handle validation instead.
  E.g., the `PrepConfig` class does not check whether 
  `output_dir` exist is a directory, but `vak.core.prep` does.
  [#550](https://github.com/NickleDave/vak/pull/550).
  Fixes [#459](https://github.com/NickleDave/vak/issues/459).
- Refactor and speed up logic for determining whether a 
  dataset with sequence annotations has unlabeled segments 
  that should be assigned a "background" label
 [#559](https://github.com/NickleDave/vak/pull/559).
 Fixes [#243](https://github.com/NickleDave/vak/issues/243).
  - Adds a new sub-sub-package, `datasets.seq`
    with a `validators` module, which is where the 
    re-written `has_unlabeled` function now lives. 
    Replaces the `vak.csv` module which was not well named.
  - Also adds a `has_unlabeled` function to `vak.annotation` 
    that is used by `vak.datasets.seq.validators.has_unlabeled`; 
    this function handles edge cases outlined in
    [#243](https://github.com/NickleDave/vak/issues/243).
- Rename and refactor functions in `vak.annotation` 
  that map annotations to the files that they annotate, 
  so that the purpose of the functions is clearer, 
  and add clearer error messages with links to documentation 
  about file naming conventions 
 [#566](https://github.com/NickleDave/vak/pull/566).
 Fixes [#525](https://github.com/NickleDave/vak/issues/525).
- Revise "autoannotate" tutorial to use .wav audio and .csv 
  annotation files from new release of Bengalese Finch Song 
  Repository, and to suggest that Windows users unpack 
  archives with tar, not other programs such as WinZip
  [#578](https://github.com/NickleDave/vak/pull/578).
  Fixes [#560](https://github.com/NickleDave/vak/issues/560)
  and [#576](https://github.com/NickleDave/vak/issues/576).
- Change `vak.files.find_fname` and `vak.files.spect.find_audio_fname` 
  so they work when spaces are in filename and/or path 
  [#594](https://github.com/NickleDave/vak/pull/594).
  Fixes [#589](https://github.com/NickleDave/vak/issues/589).

### Fixed
- Fix how `vak.core.prep` handles `labelset` parameter.
  Add pre-condition that raises a ValueError
  when `labelset` is `None` but the .toml config is one of 
  {'train', 'learncurve', 'eval'}
  [#545](https://github.com/NickleDave/vak/pull/545).
  Avoids running computationally expensive step of generating 
  and validating spectrograms *before* crashing when trying to 
  split the dataset using `labelset`. Also avoids silent 
  failures for datasets that do not require splitting, 
  e.g., an 'eval' set that could contain labels not in the 
  training set.
  Fixes [#468](https://github.com/NickleDave/vak/issues/468).
- Fix how `cli` and `core` functions that have the `csv_path` parameter
  handles  it. The parameter points to a dataset .csv generated by `vak prep`
  that other `core`/`cli` function use: `train`, `learncurve`, `eval`, `predict`.
  They now validate that it exists, and if it doesn't, the `cli` functions 
  politely suggest running `vak prep` first; the `core` functions 
  raise a FileNotFoundError.
  [#546](https://github.com/NickleDave/vak/pull/546).
  Fixes [#469](https://github.com/NickleDave/vak/issues/469).
- Fix bug where `labelmap_path` parameter was ignored by `core.train`.
  Change function so that either `labelmap_path` or `labelset` must 
  be passed in, both passing in both will raise an error.
  Also change `cli.train` to only pass in one of those and set the other 
  to `None`.
  [#552](https://github.com/NickleDave/vak/pull/552).
  Fixes [#547](https://github.com/NickleDave/vak/issues/547).
- Fix `vak.annotation.has_unlabeled` to handle the edge case where an 
  annotation file has no annotated segments
  [#583](https://github.com/NickleDave/vak/pull/583).
  Fixes [#378](https://github.com/NickleDave/vak/issues/378).
- Fix `StandardizeSpect` method `fit_df` so that it computes
  parameters for standardization from a specific
  split of the dataset--the training split, by default--instead 
  of using the entire dataset, which could technically give rise 
  to data leakage
  [#584](https://github.com/NickleDave/vak/pull/583).
  Fixes [#575](https://github.com/NickleDave/vak/issues/575).
- Fix error message in `vak.core.eval`
  [#589](https://github.com/NickleDave/vak/pull/589).
  Fixes [#588](https://github.com/NickleDave/vak/issues/588).
 
## 0.6.0 -- 2022-07-07
### Added
- better document `conda` install 
  [#528](https://github.com/NickleDave/vak/pull/528).
  Fixes [#527](https://github.com/NickleDave/vak/issues/527).
- Add tests for console script, i.e., the command-line interface 
  [#533](https://github.com/NickleDave/vak/pull/533).
  Fixes [#369](https://github.com/NickleDave/vak/issues/369).

### Changed
- switch from using `make` to `nox` for running tasks 
  [#532](https://github.com/NickleDave/vak/pull/532).
  Fixes [#440](https://github.com/NickleDave/vak/issues/440).
- Refactor logging so that it can be configured by `cli` functions
  when running `vak` through command-line interface, and by users 
  that are working with the API directly
  [#535](https://github.com/NickleDave/vak/pull/535).

### Fixed
- Fix bug that prevented creating spectrogram files with non-default keys
  (e.g. 'spect' instead of the default 's'). Needed to pass keys from `spect_params` 
  into `spect.to_dataframe` inside `vak.io.dataframe.from_files`. 
  [#531](https://github.com/NickleDave/vak/pull/531).
  Fixes [#412](https://github.com/NickleDave/vak/issues/412).
- Fix logging so a single message is not logged multiple times. 
  [#535](https://github.com/NickleDave/vak/pull/535).
  Fixes [#258](https://github.com/NickleDave/vak/issues/258).
- Fix section of contributing docs on setting up a development environment. 
  [#592](https://github.com/NickleDave/vak/pull/592).
  Fixes [#591](https://github.com/NickleDave/vak/issues/591).

## 0.5.0.post1 -- 2022-06-25
### Fixed
- Put upper bound on `crowsetta` dependency
  [89ee7b03](https://github.com/vocalpy/vak/commit/89ee7b03bade2c20da6d4c480cf8c799eacee9fb)

## 0.5.0 -- 2022-06-25
### Added
- add ability to continue training from an existing checkpoint 
  [#505](https://github.com/NickleDave/vak/pull/505).
  Fixes [#5](https://github.com/NickleDave/vak/issues/5).

## Changed
- change minimum required Python to 3.8, 
  to adhere to [NEP-29](https://numpy.org/neps/nep-0029-deprecation_policy.html), in 
  [#513](https://github.com/NickleDave/vak/pull/513).
  Fixes [#512](https://github.com/NickleDave/vak/issues/512).

## Fixed
- fix explanation of naming convention for `'simple-seq'` format in how-to
  on using annotation formats
  [#517](https://github.com/NickleDave/vak/pull/517).
  Fixes [#516](https://github.com/NickleDave/vak/issues/516).
- fix links to images in README so they show up on PyPI
  [db98c30](https://github.com/vocalpy/vak/commit/db98c304db9c380086ef60f9a530cbcfd2a96330)

## [0.4.2](https://github.com/NickleDave/vak/releases/tag/0.4.2) -- 2022-03-29
### Added
- add a [Code of Conduct](https://github.com/NickleDave/vak/blob/main/CODE_OF_CONDUCT.md), 
  a [contributing guide on GitHub](https://github.com/NickleDave/vak/blob/main/.github/CONTRIBUTING.md), 
  and a 
  [Development section of the documentation](https://vak.readthedocs.io/en/latest/development/index.html) 
  [#448](https://github.com/NickleDave/vak/pull/448).
  Fixes [#8](https://github.com/NickleDave/vak/issues/8) and
  [#56](https://github.com/NickleDave/vak/issues/56).
- add pull request templates on GitHub 
  [#445](https://github.com/NickleDave/vak/pull/448).
  Fixes [#85](https://github.com/NickleDave/vak/issues/85).
- add links to page describing format for array files 
  containing spectrograms, on the reference index, and on 
  the how-to page on using your own spectrograms. 
  Also add a link to a small example dataset of 
  spectrogram files 
  [#494](https://github.com/NickleDave/vak/pull/494).
  Fixes [#492](https://github.com/NickleDave/vak/issues/492).
- add more detail to explanation of how to use `'csv'` format 
  for annotation
  [#495](https://github.com/NickleDave/vak/pull/495).
  Fixes [#491](https://github.com/NickleDave/vak/issues/491).

### Changed
- make minor revisions to docs
  [#443](https://github.com/NickleDave/vak/pull/443).
  Fixes [#439](https://github.com/NickleDave/vak/issues/439).
- rewrite docs in Markdown / `MyST` wherever possible; 
  install MyST parser for Sphinx
  [#463](https://github.com/NickleDave/vak/pull/463).
  Fixes [#384](https://github.com/NickleDave/vak/issues/384).
- require `crowsetta` version 3.4.0 or greater; 
  in this version, annotation format `'csv'` is now named `'generic-seq'` 
  (and the name `'csv'` will stop working in the next version);
  format `'simple-csv'` renamed to `'simple-seq'`
  [#496](https://github.com/NickleDave/vak/pull/496).
  Fixes [#497](https://github.com/NickleDave/vak/issues/497).
- revise how-to page on annotation formats, 
  to include vignettes for the `'simple-seq'` and 
  `'generic-seq'` formats.
  [#498](https://github.com/NickleDave/vak/pull/498).
  Fixes [#429](https://github.com/NickleDave/vak/issues/429).

### Fixed
- fix bug that caused `vak prep` to crash 
  when there was only one file in a data directory
  [#483](https://github.com/NickleDave/vak/pull/483).
  Fixes [#467](https://github.com/NickleDave/vak/issues/467).
- fix bug that caused `vak prep` to crash 
  when a `.not.mat` annotation file only had a single annotated segment
  [#488](https://github.com/NickleDave/vak/pull/488).
  Fixes [#466](https://github.com/NickleDave/vak/issues/466).

## [0.4.1](https://github.com/NickleDave/vak/releases/tag/0.4.1) -- 2022-01-07
### Changed
- switch to using `flit` to build/publish, drop `poetry`
  [#434](https://github.com/NickleDave/vak/pull/434).
  Fixes [#433](https://github.com/NickleDave/vak/issues/433).
- raise minimum required `pytorch` version to 1.7.1 and 
  minimum `crowsetta` version to 3.2.0'
  [#437](https://github.com/NickleDave/vak/pull/437).
- do various clean-up steps to development / CI workflows, 
  in the process of getting ready to publish `vak` on `conda-forge`
  [#437](https://github.com/NickleDave/vak/pull/437).
- resolve various minor docs issues
  [#438](https://github.com/NickleDave/vak/pull/438).

## [0.4.0](https://github.com/NickleDave/vak/releases/tag/0.4.0) -- 2021-12-29
### Added
- add a [CITATION.cff](https://citation-file-format.github.io/) file
  [#407](https://github.com/NickleDave/vak/pull/407).
- add an [all-contributors](https://allcontributors.org/) table to README, 
  using their bot to adopt the spec.
  E.g., [#395](https://github.com/NickleDave/vak/pull/395). 
  Fixes [#387](https://github.com/NickleDave/vak/issues/387).
- add description of command-line interface to reference section of documentation.
  [#417](https://github.com/NickleDave/vak/pull/417).
  Fixes [#270](https://github.com/NickleDave/vak/issues/270).
- add how-to on using an annotation format that's not built in
  [#421](https://github.com/NickleDave/vak/pull/421).
  Fixes [#397](https://github.com/NickleDave/vak/issues/397).
- add how-to on using custom spectrograms 
  [#421](https://github.com/NickleDave/vak/pull/421).
  Fixes [#413](https://github.com/NickleDave/vak/issues/413).

### Changed
- updated the .toml configuration files in the tutorial 
  to match what was used for [TweetyNet paper](https://github.com/yardencsGitHub/tweetynet).
  [#416](https://github.com/NickleDave/vak/pull/416).
  Fixes [#414](https://github.com/NickleDave/vak/issues/414).
- move tutorial into "getting started" section of docs, 
  and revise landing page of docs
  [#419](https://github.com/NickleDave/vak/pull/419).
- revise the documentation for the configuration file format.
  Show valid options for each section by including docstrings from the classes
  that represents the different sections
  [#428](https://github.com/NickleDave/vak/pull/428).
  Fixes [#271](https://github.com/NickleDave/vak/issues/271).

### Fixed
- make further fixes + add unit tests for handling predictions where all timebins 
  are the background "unlabeled" class [#409](https://github.com/NickleDave/vak/pull/409).
  Fixes bug in `remove_short_segments` [#403](https://github.com/NickleDave/vak/issues/403).
  Related to [#393](https://github.com/NickleDave/vak/issues/393) 
  and [#386](https://github.com/NickleDave/vak/issues/386).
- fix docs so entries appear in navbar
  [#427](https://github.com/NickleDave/vak/pull/427).
  Fixes [#426](https://github.com/NickleDave/vak/issues/426).

## [0.4.0b6](https://github.com/NickleDave/vak/releases/tag/0.4.0b6) -- 2021-11-23
### Changed
- bump minimum Python to 3.7
  [#388](https://github.com/NickleDave/vak/pull/388).
  Fixes [#380](https://github.com/NickleDave/vak/issues/380).

### Fixed
- fix how `predict` handles annotations that are predicted to have no labeled segments, 
  i.e. where all time bins are predicted to have "background" / "unlabeled" class
  [#394](https://github.com/NickleDave/vak/pull/394).
  For details, see [#393](https://github.com/NickleDave/vak/issues/393) and 
  [#386](https://github.com/NickleDave/vak/issues/386).

## [0.4.0b5](https://github.com/NickleDave/vak/releases/tag/0.4.0b5) -- 2021-10-08
### Changed
- change Python constraint to include 3.9
  [#368](https://github.com/NickleDave/vak/pull/368)

### Fixed
- fix typo in `doc/reference/reference.rst` that broke a link
  [#363](https://github.com/NickleDave/vak/issues/363)
- fix bug in function `lbl_tb2labels` that affect calculation of segment error rate 
  for annotations with digits that had multiple characters (e.g. '21', '22').
  [#377](https://github.com/NickleDave/vak/pull/377).
  For details see [#373](https://github.com/NickleDave/vak/issues/373)

## [0.4.0b4](https://github.com/NickleDave/vak/releases/tag/0.4.0b4) -- 2021-04-25
### Added
- add `events2df` function to `tensorboard` module that converts an "events" file 
  (log) created during training into a `pandas.DataFrame`, to make it easier to 
  work directly with logged scalar values, e.g. plot training history showing loss 
  [#346](https://github.com/NickleDave/vak/pull/346).
- add Dice loss, commonly used for segmentation problems, adapted from `kornia` library 
  for use with 1-D sequences [#357](https://github.com/NickleDave/vak/pull/357).

### Changed
- change name of `summary_writer` module to `tensorboard` to reflect that it contains 
  any function related to `tensorboard` [#346](https://github.com/NickleDave/vak/pull/346).

### Fixed
- fix bug in Levenshtein distance implementation 
  [#356](https://github.com/NickleDave/vak/pull/342).
  For details see issue [#355](https://github.com/NickleDave/vak/issues/355).
  Also added unit tests for Levenshtein distance and segment error rate
  in [#356](https://github.com/NickleDave/vak/pull/342).

## [0.4.0b3](https://github.com/NickleDave/vak/releases/tag/0.4.0b3) -- 2021-04-04
### Changed
- refactor unit tests so they can be run with `TeenyTweetyNet` on Github Actions
  [#339](https://github.com/NickleDave/vak/pull/339).
  For details see issue [#330](https://github.com/NickleDave/vak/issues/330).

### Fixed
- fix how `train_dur_csv_paths.from_dir` orders replicates
  [#342](https://github.com/NickleDave/vak/pull/342).
  For details see issue [#340](https://github.com/NickleDave/vak/issues/340).

## [0.4.0b2] -- 2021-03-21
### Added
- add built-in model `TeenyTweetyNet` 
  [#329](https://github.com/NickleDave/vak/pull/329) 
  that will be used to speed up `vak` test suite.
  For details see issue [#308](https://github.com/NickleDave/vak/issues/308).
- make it so config does not validate other sections when running `vak prep`, 
  to avoid annoying errors due to options that are going to change anyway 
  [#335](https://github.com/NickleDave/vak/pull/335).
  For details see [#314](https://github.com/NickleDave/vak/issues/314) and   
  [#334](https://github.com/NickleDave/vak/issues/334).
- raise clear error message when running `vak prep` and the section that a  
  dataset is being `prep`ared for already has a `csv_path`. Ask user to 
  remove it if they really want to generate a new one 
  [#335](https://github.com/NickleDave/vak/pull/335).
  For details see [#314](https://github.com/NickleDave/vak/issues/314) and   
  [#333](https://github.com/NickleDave/vak/issues/333).
- add `split` parameter to `WindowDataset.spect_vectors_from_df` 
  [#336](https://github.com/NickleDave/vak/pull/336).
  For details see issue [#328](https://github.com/NickleDave/vak/issues/328).

### Changed
- refactor `config` sub-package
  [#335](https://github.com/NickleDave/vak/pull/335).
  For details see [#331](https://github.com/NickleDave/vak/issues/331),  
  [#332](https://github.com/NickleDave/vak/issues/332).

### Fixed
- change `model.load` method, so that `torch.load` uses `map_location` parameter
  [#324](https://github.com/NickleDave/vak/pull/324).
  This way, loading a model trained on a GPU won't 
  cause a RuntimeError if only a CPU is available.
  For details see issue [#323](https://github.com/NickleDave/vak/issues/323).
- fix `train_dur_csv_paths.from_dir` so it uses correct dataset splits to 
  generate `spect_vector`s for `WindowDataset` 
  [#336](https://github.com/NickleDave/vak/pull/336).
  For details see issue [#328](https://github.com/NickleDave/vak/issues/328).

## [0.4.0b1] -- 2021-03-06
### Added
- add ability to save "raw outputs" of network, e.g. the "logits", 
  when running `vak predict` command
  [#320](https://github.com/NickleDave/vak/pull/320).
  For details see issue [#90](https://github.com/NickleDave/vak/issues/90).

### Changed
- change `split.algorithms.validate.validate_split_durations_and_convert_nonnegative` 
  so that it no longer converts all durations to non-negative numbers, because the 
  functions that call it need to "see" when a target split duration is specified as 
  -1 (meaning "use any remaining vocalizations in this split") so they can determine 
  properly when they've finished dividing the dataset into splits.
  Accordingly, rename to `split.algorithms.validate.validate_split_durations`.
  [#300](https://github.com/NickleDave/vak/pull/300)
- refactor code that programmatically builds `results_path` used in `core` and `cli` 
  functions that run `train` and `learncurve`
  [#304](https://github.com/NickleDave/vak/pull/304).
  For details see 
  [comment on pull request](https://github.com/NickleDave/vak/pull/304#issue-576981330). 
- refactor `vak.config.parse.from_toml` function into two others, 
  the original and a new `parse.from_toml_path` 
  [#306](https://github.com/NickleDave/vak/pull/306).
  For details see issue [#305](https://github.com/NickleDave/vak/issues/305)
- switch to using `pytest` to run test suite [#309](https://github.com/NickleDave/vak/pull/309).
- switch to using Github Actions for continuous integration
  [#312](https://github.com/NickleDave/vak/pull/312).
- parametrize `device` fixture so tests run on CPU and, when present, GPU
  [#313](https://github.com/NickleDave/vak/pull/313)
- refactor `cli.learncurve` module into a sub-package with separate module for 
  `train_csv_paths` helper functions used by `learning_curve`
  [#319](https://github.com/NickleDave/vak/pull/319)
- lower the lower bounds on dependencies, so that users can install with earlier 
  versions of `torch`, `torchvision`, etc.
  [9c6ed46](https://github.com/NickleDave/vak/commit/9c6ed46822c53aaa25f66b050b6490657ec5005b)

### Fixed
- fix `split.algorithms.bruteforce` so that it always returns either a list of 
  indices or `None` for each split, instead of sometimes returning an empty list
  instead of a `None`. Also rewrite this function for clarity and to obey DRY 
  principle.
  [#300](https://github.com/NickleDave/vak/pull/300)
- fix unit tests [#309](https://github.com/NickleDave/vak/pull/309).
- fix how runs of `learncurve` that use `previous_run_path` get the 
  "spect vectors" that determine valid windows that can grabbed from 
  the `WindowDataset`
  [#319](https://github.com/NickleDave/vak/pull/319). 
  For details see [#316](https://github.com/NickleDave/vak/issues/316).
  There was a bug with the first attempt to fix this, that was resolved by 
  [#322](https://github.com/NickleDave/vak/pull/322).
  For details see issue [#321](https://github.com/NickleDave/vak/issues/321).

## [0.4.0dev1] - 2021-01-24
### Note this version was "yanked" from PyPI because of issues with how dependencies were specified
### Added
- automate generation of test data.
  [#274](https://github.com/NickleDave/vak/pull/274)
  This pull request also adds concept of 'source' and 'generated' test data, 
  and decouples them from the source code in other ways, e.g. adding 
  a Makefile command that downloads them as .tar.gz files from an 
  Open Science Framework project.
  See details in comment on pull request: 
  https://github.com/NickleDave/vak/pull/274#issue-538992350
- make it possible to specify `spect_output_dir` when `prep`ing datasets, 
  the directory where array files containing spectrograms are saved
  [#290](https://github.com/NickleDave/vak/pull/290).
  Addresses issue [#289](https://github.com/NickleDave/vak/issues/289).
- add ability to specify `previous_run_path` when running `learncurve`,  
  so that training data subsets generated by a previous run are used 
  instead of generating new subsets. Controls for any effect of 
  changing training data across experiments, and makes things faster
  [#291](https://github.com/NickleDave/vak/pull/291)

### Changed
- make it possible for labels in `labelset` to be multiple characters
  [##278](https://github.com/NickleDave/vak/pull/278)
- switch to `crowsetta` version 3.0.0, making it possible to specify 
  `csv` as an annotation format
  [#279](https://github.com/NickleDave/vak/pull/279)
- switch to using `soundfile` to load audio files
  [#281](https://github.com/NickleDave/vak/pull/281)
- switch to using `poetry` for development
  [#283](https://github.com/NickleDave/vak/pull/283)
- move `converters` module out of `config` sub-package up to top level
  [4ad9b93](https://github.com/NickleDave/vak/commit/4ad9b9390be6ac97b3dbe2b459e94d12d35ff051)
- rename `converters.labelset_from_toml_value` to `labelset_to_set` 
  since it will be used throughout package (not just with .toml config files)
  [4ad9b93](https://github.com/NickleDave/vak/commit/4ad9b9390be6ac97b3dbe2b459e94d12d35ff051)
-  make other functions use `converter.labelset_to_set` for `labelset` argument
  [35a67d8](https://github.com/NickleDave/vak/commit/35a67d87aabe82b8485162573777d06ff5571409)
  [902d840](https://github.com/NickleDave/vak/commit/902d8405610e54da4645732353118439e2349946)
  [062902e](https://github.com/NickleDave/vak/commit/062902ed101c8bf5ed6552c2c055a0c15d019396)
  [d4e673c](https://github.com/NickleDave/vak/commit/d4e673c792532e311dfb44118e513a615377b2fb)
- rename `vak/validation.py` -> `vak/validators.py`
  [9df32e2](https://github.com/NickleDave/vak/commit/9df32e24c650057fc34dd7e53c159bae24192f25)
- raise minimum versions for `crowsetta`, at least 3.1.0, and `tweetynet`, at least 0.5.0
  [e1a6fbb](https://github.com/NickleDave/vak/commit/e1a6fbb9d3ccdb63167446684a8aecb3e667fd8a)
- make `vak.io.audio.to_spect` use `vak.logging.log_or_print` function 
  so that logger messages actually appear in terminal and in log files
  [af719b3](https://github.com/NickleDave/vak/commit/af719b30faa4484f2f27a0e0a236310576e8ecb0) 
 
### Fixed
- add missing import of `eval` module to `vak.cli.__init__` and organize import statements 
  [6341c8d](https://github.com/NickleDave/vak/commit/6341c8d4991a4e51565953f8e15d40f13419e6d5)
- fix `vak.files.from_dir` function, that returns list of all files 
  from a directory with specified extension, so that it is case-insensitive
  [#276](https://github.com/NickleDave/vak/pull/276)
- fix `vak.annotation.recursive_stem` function so it is case-insensitive
  [c02bd8a](https://github.com/NickleDave/vak/commit/c02bd8a8d33eadeb5ce04725d63f1d2e520de737)
- fix `vak.io.audio.to_spect` so validation of `audio_files` is case-insensitive
  [cbd08f6](https://github.com/NickleDave/vak/commit/cbd08f6deab7a26fbbb1814fbe6349c578dae20f)
- fix `find_audio_fname` to work with str and Path
  [1480b01](https://github.com/NickleDave/vak/commit/1480b01ebc623a64a5c077c26ffdcaa242f29f3e)
- fix how `labelset_to_set` handles set, and add type-checking as pre-condition, 
  sp that the function doesn't just return `None`
  [6c454cd](https://github.com/NickleDave/vak/commit/6c454cda3aded7c0cf7ac19a6eef6f6831220033)
- use `poetry` in Makefile to run scripts that generate test data, 
  so that development version of `vak` is used, 
  not some other version that might be installed into an environment
  (e.g. a `conda` environment the developer had activated)
  [090c205](https://github.com/NickleDave/vak/commit/090c205e227824eda7c1b156f5320129a4809b6b)
- make `source_annot_map` have no side effects, fixes [#287](https://github.com/NickleDave/vak/issues/287)
  [d1cbe82](https://github.com/NickleDave/vak/commit/d1cbe82132f46f5cc400524dfefdc94de55c430b)

### Removed
- remove `tweetynet` as a core dependency, since this creates a 
  circular dependency (`tweetynet` definitely depends on `vak`) 
  that prevents using `conda-forge`. Instead declare `tweetynet` as 
  a test dependency.
  [74350a7](https://github.com/NickleDave/vak/commit/c26ad08bfd4057324ba55a1902f7dc2845bc6e40)
- remove `output_dir` parameter from `dataframe.from_files` -- not used
  [#286](https://github.com/NickleDave/vak/pull/286)
- remove filtering by `labelset` in `dataframe.from_files`
  [7dbdc23](https://github.com/NickleDave/vak/commit/7dbdc233a0776e6c205a65ee062f2dce9d479af8)

## [0.3.3]
### Fixed
- remove out-of-date install instructions that were confusing people 
  [#268](https://github.com/NickleDave/vak/pull/268)

## [0.3.2]
### Fixed
- fix wrong argument value in call to imshow in `plot.spect_annot` function
  [648b675](https://github.com/NickleDave/vak/commit/648b675221472f6bcd2750262c57dd8a761099e0)
- fix bug that caused `vak.config.parse` to silently fail when parsing the 
  `[SPECT_PARAMS]` section of config.toml files
  [#266](https://github.com/NickleDave/vak/pull/266)

## [0.3.1]
### Fixed
- fix `RuntimeError` under torch 1.6 caused by 
  dividing a tensor by an integer in `Model._eval()` method
  [#250](https://github.com/NickleDave/vak/pull/250).
  Fixes [#249](https://github.com/NickleDave/vak/issues/249).  

## [0.3.0]
### Added
- add functionality to `WindowDataset` that enables training with datasets 
  of specified durations [#188](https://github.com/NickleDave/vak/pull/186)
- add transforms for post-hoc clean up of predicted labels for time bins, 
  that are applied before converting into segments with labels, onsets, and offsets
  + `majority_vote_transform` that find the most frequently occurring label in a segment 
    and assigns it to the entire segment [#227](https://github.com/NickleDave/vak/pull/227)
  + `remove_short_segments` that removes any segments shorter than a specified duration
    [#229](https://github.com/NickleDave/vak/pull/229)
- add logic to `WindowDataset.crop_spect_vectors_keep_classes` method so that it tries 
  to crop a third way, by removing unlabeled segments within vocalizations, if cropping 
  the specified duration from the end or beginning fails
  [#224](https://github.com/NickleDave/vak/pull/224)
- add ability to specify name of .csv file containing annotations produced by 
  `vak.core.predict` [#232](https://github.com/NickleDave/vak/pull/232)
- make it so that ItemTransforms (optionally) return path to array files 
  containing spectrograms, so user can easily link train/test/predict data 
  returned by `DataLoader` to the source file
  [#236](https://github.com/NickleDave/vak/pull/236)
- add functions for plotting spectrograms and annotation to `plot` sub-package
  [#245](https://github.com/NickleDave/vak/pull/245)

### Changed
- refactor to remove `util`s modules [#196](https://github.com/NickleDave/vak/pull/196)
- add `core.predict` module and rewrite `cli.predict` to use it
  [#210](https://github.com/NickleDave/vak/pull/210)
- modify `vak.split.algorithms.brute_force` so that it 
  starts by seeding each split with one instance of each 
  label in the label set. Quick tests found that this 
  improves success rate of splits on one dataset 
  with many (30) classes.
  [#218](https://github.com/NickleDave/vak/pull/218) 
- change `core.predict` so that it always saves 
  predicted annotations as a .csv file 
  [#222](https://github.com/NickleDave/vak/pull/222).
  Removed functionality for converting to other formats.
  See discussion in [#212](https://github.com/NickleDave/vak/issues/211)
- change warning issued by `split.train_test_dur_split_inds` to a log 
  statement [#231](https://github.com/NickleDave/vak/pull/231)
- use `VocalDataset` in `core.predict`,
  see discussion in issue [#206](https://github.com/NickleDave/vak/issues/206)
  [#242](https://github.com/NickleDave/vak/pull/242)
- revise README [#248](https://github.com/NickleDave/vak/pull/248)

### Fixed
- changes references to `config.ini` in docstrings to `config.toml`
  [#190](https://github.com/NickleDave/vak/pull/190)
- fix error type in 'config.predict' [#197](https://github.com/NickleDave/vak/pull/197)
- add missing `to_format_kwargs` attribute to `PredictConfig` docstring
  [#210](https://github.com/NickleDave/vak/pull/210)
- add missing parameter in `transforms.default.get_defaults`
  [#210](https://github.com/NickleDave/vak/pull/210)
- add missing import in `cli.predict`
  [#210](https://github.com/NickleDave/vak/pull/210)
- revise `autoannotate` tutorial to include missing steps in `predict`
  [#210](https://github.com/NickleDave/vak/pull/210)
- fix up `config.toml` files that are used with `autoannotate` tutorial
  [#210](https://github.com/NickleDave/vak/pull/210)
- fix variable name error in `WindowDataset.crop_spect_vectors_keep_classes` method
  [#215](https://github.com/NickleDave/vak/pull/215)
- fix bug in `WindowDataset.crop_spect_vectors_keep_classes`
  [#217](https://github.com/NickleDave/vak/issues/217)
  that caused `x_inds` to have invalid values when the 
  `WindowDataset.crop_spect_vectors_keep_classes` function 
  cropped the vectors to a specified duration "from the front"
  [#219](https://github.com/NickleDave/vak/pull/219)
- remove line that caused `vak predict` to crash
  [#211](https://github.com/NickleDave/vak/issues/211)
  when model was trained without a `SpectStandardizer` transform
  [#221](https://github.com/NickleDave/vak/pull/221)
- fix bugs that prevented `vak eval` cli command from working
  [#238](https://github.com/NickleDave/vak/pull/238) 
- fix bug in `labels.lbl_tb2labels` (https://github.com/NickleDave/vak/issues/239) 
  that resulted from lack of input validation and an indentation error
  [#240](https://github.com/NickleDave/vak/pull/240) 
- fix how segment onsets and offsets are converted from time bin "units" 
  back to seconds [#246](https://github.com/NickleDave/vak/pull/246).
  Fixes [#237](https://github.com/NickleDave/vak/issues/237).
- fix .toml config file used with "autoannotate" tutorial, 
  and revise related section of tutorial on prediction
  [#247](https://github.com/NickleDave/vak/pull/247).
  Fixes [#223](https://github.com/NickleDave/vak/issues/223).

### Removed
- remove `bin/` that contained scripts used with previous version of `vak`
  [#226](https://github.com/NickleDave/vak/pull/226)
- remove mentions of `.ini` config files from documentation
  [#248](https://github.com/NickleDave/vak/pull/248)

## [0.3.0a5]
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
- add `log_or_print` function to `util.logging` that either writes message to logger, 
  or simply prints the message if there is no logger [#147](https://github.com/NickleDave/vak/pull/147)
- add `logger` attribute to `vak.Model` class, used to log if not None 
  [#148](https://github.com/NickleDave/vak/pull/148)
- add Tensorboard `SummaryWriter` to `vak.Model` class so there is an `events` file recording each 
  model's training history [#149](https://github.com/NickleDave/vak/pull/149)
  + and add Tensorboard as a dependency in [#162](https://github.com/NickleDave/vak/pull/162)
- add additional logging to `Model` class [#153](https://github.com/NickleDave/vak/pull/153)
- add initial tutorial on using `vak` for automated annotation of vocalizations 
  [#156](https://github.com/NickleDave/vak/pull/156)
- add `VocalDataset`, more generalized form of a dataset where the input to a network is contained in a source 
  file, e.g. a .npz array file with a spectrogram, and the optional target is the annotation 
  [#165](https://github.com/NickleDave/vak/pull/165)
- add `transforms.defaults` with `ItemTransforms` that return dictionaries. Decouples logic for 
  what will be in returned "items" from the different dataset classes [#165](https://github.com/NickleDave/vak/pull/165)
- add `eval` command to command-line interface [#179](https://github.com/NickleDave/vak/pull/179)
- add `vak.core` sub-package with "core" functions that are called by corresponding functions in 
  `vak.cli`, e.g. `vak.cli.train` calls `vak.core.train`; de-couples high-level functionality from 
  command-line interface, and makes it possible for one high-level function to call another, i.e., 
  `vak.core.learncurve` calls `vak.core.train` and `vak.core.eval`
  [#183](https://github.com/NickleDave/vak/pull/183)
- add computation of distance metrics to `Model._eval` method
  [#185](https://github.com/NickleDave/vak/pull/185)

### Changed
- rewrite `vak.util.dataset.has_unlabeled` to use `annotation.from_df` 
  [#107](https://github.com/NickleDave/vak/pull/107)
- bump minimum version of `TweetyNet` to 0.3.1 in [#120](https://github.com/NickleDave/vak/pull/120)
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
- bump minimum version of `TweetyNet` to 0.4.0 in [#155](https://github.com/NickleDave/vak/pull/155)
  + so `TweetyNetModel.from_class` method accepts `logger` argument
- change checkpointing and validation so that they occur on specific steps, not epochs.
  [#161](https://github.com/NickleDave/vak/pull/161)
  This way models with very large training sets that may run for only 1-2 epochs still intermittently save 
  checkpoints as backups and measure performance on the validation set.
- change names of `TrainConfig` attributes `val_error_step` and `checkpoint_step` to `val_step` and `ckpt_step` 
  for brevity + clarity. [#161](https://github.com/NickleDave/vak/pull/161) Also changed the names of the 
  corresponding `vak.Model.fit` method parameters to match.
- change `vak.Model._eval` method to work like `vak.cli.predict` does, feeding models non-overlapping 
  windows from spectrograms [#165](https://github.com/NickleDave/vak/pull/165)
- change `reshape_to_window` transform to `view_as_window_batch` because it was not working as intended 
  [#165](https://github.com/NickleDave/vak/pull/165)
- bump minimum version of `TweetyNet` to 0.4.1 in [#172](https://github.com/NickleDave/vak/pull/172)
  + version that changes optimizer back to `Adam`
- raise lower bound on `crowsetta` version to 2.2.0, to get fixes for `koumura2annot`
  and avoid errors when `annot_file` is provided as a `pathlib.Path` instead of a `str`
  [#175](https://github.com/NickleDave/vak/pull/175)
- change `Model._eval` method so it returns metrics average across batches, in addition to 
  the value for each batch
  [#185](https://github.com/NickleDave/vak/pull/185)
- raise minimum version of `TweetyNet` to 0.4.2, adds distance metrics to `TweetyNetModel`
  [9626385](https://github.com/NickleDave/vak/commit/96263858efe880f94dc782cd8a66ec1c051f2ea1)

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
- fix indent error in `Model.fit` method (see issue [#151](https://github.com/NickleDave/vak/issues/151)) 
  that stopped training early [#153](https://github.com/NickleDave/vak/pull/153) 
- fix bug [#166](https://github.com/NickleDave/vak/issues/166) 
  that let training continue even after `patience` number of validation steps had elapsed 
  without an increase in accuracy [#168](https://github.com/NickleDave/vak/pull/168) 
- fix `learncurve` functionality so it will work in version `0.3.0` 
  [#183](https://github.com/NickleDave/vak/pull/183)

### Removed
- remove `vak.util.general.safe_truncate` function, no longer used 
  [#137](https://github.com/NickleDave/vak/issues/137)
- remove redundant validation of split durations in `util.split` 
  [#143](https://github.com/NickleDave/vak/pull/143)
- removed `save_only_single_checkpoint_file` option and functionality
  [#161](https://github.com/NickleDave/vak/pull/161). 
  Now save only one checkpoint as backup, and another for best performance on validation set if provided.
  See discussion in pull request and the issues it fixes for more detail.

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
