(contributing)=

# Contributors Guide

## Ways to Contribute

### Ways to Contribute Documentation and/or Code

* Tackle any issue that you wish! Some issues are labeled as **"good first issues"** to
  indicate that they are beginner friendly, meaning that they don't require extensive
  knowledge of the project.
* Make a tutorial or gallery example of how to do something.
* Improve the API documentation.
* Contribute code! This can be code that you already have and it doesn't need to be
  perfect! We will help you clean things up, test it, etc.

### Ways to Contribute Feedback

* Provide feedback about how we can improve the project or about your particular use
  case. Open an [issue](https://github.com/vocalpy/vak/issues) with
  feature requests or bug fixes.
* Help triage issues, or give a "thumbs up" on issues that others reported which are
  relevant to you (using the
  ["Add Your Reaction" icon](https://github.blog/2016-03-10-add-reactions-to-pull-requests-issues-and-comments/)).

### Ways to Contribute to Community Building

* Cite vak when using the project.
  Please see [the CITATION.cff](https://github.com/vocalpy/vak/blob/main/CITATION.cff) file for details.
  To obtain a citation in APA or BibTEX format, click on "Cite this repository" on the
  [GitHub repository](https://github.com/vocalpy/vak).
* Spread the word about vak and star the project on GitHub!

## Providing Feedback

Two of the main ways to contribute are to report a bug or to request a feature.
Both can be done by opening an [Issue](https://github.com/vocalpy/vak/issues)
on GitHub and filling out the template.

* Find the [Issues](https://github.com/vocalpy/vak/issues) tab on the
  top of the GitHub repository and click *New Issue*.
* Choose a template based on the type of feedback

  * For a bug report, click on *Get started* next to *Bug report*.

  * For a feature request, click on *Get started* next to *Feature request*.

* **Please try to fill out the template with as much detail as you can**.
* After submitting your bug report or feature request,
  try to answer any follow up questions as best as you can.

## General Guidelines

(getting-help)=

### Getting Help

Discussion often happens on GitHub issues and pull requests. In addition, there is a
[Discourse forum](https://forum.vocalpy.org/) for
the project where you can ask questions.

(dev-workflow)=

### Workflow for Contributing

We follow the [GitHub pull request workflow](http://www.asmeurer.com/git-workflow)
to make changes to our codebase. Every change made goes through a pull request, even
our own, so that our
[continuous integration](https://the-turing-way.netlify.app/reproducible-research/ci.html)
services have a chance to check that the code is up to standards and passes all
our tests. This way, the *main* branch is always stable.

#### For New Contributors

Please take a look at these resources to learn about Git and pull requests:

* [How to Contribute to Open Source](https://opensource.guide/how-to-contribute/).
* [Git Workflow Tutorial](http://www.asmeurer.com/git-workflow/) by Aaron Meurer.
* [How to Contribute to an Open Source Project on GitHub](https://egghead.io/courses/how-to-contribute-to-an-open-source-project-on-github).

And please don't hesitate to {ref}`ask questions <getting-help>`.

#### Writing commit messages

We follow the convention of beginning a git commit message
with an abbreviation that indicates the reason for the commit.
This convention is used by several Python data science libraries.
The standard abbreviations we use to start the commit message with are

```{eval-rst}
.. include:: commit-abbreviations.rst
```

(dev-env)=

## Setting up a development environment

This section describes how to set up an environment for development.
This is the steps the maintainers follow, and it can also be used by contributors.

### You will need:
1. git, for version control

(you can install git from [Github](https://help.github.com/en/github/getting-started-with-github/set-up-git),
with your operating system package manager, or using conda.)

2. nox, for running tests and other tasks

This project uses the library [nox](https://nox.thea.codes/en/stable/)
as a [task runner](https://scikit-hep.org/developer/tasks),
to automate tasks like setting up a development environment.
Each task is represented as what nox calls a "session",
and you can run a session by invoking nox
at the command-line with the name of the session.
So, to set up a virtual environment for development
with vak installed in "editable" mode,
you would run the "dev" session, as explained below
in {ref}`Steps to set up a development environment`.

We suggest using [pipx](https://github.com/pypa/pipx)
to install nox in its own isolated environment,
so that nox can be accessed system-wide without affecting
anything else on your machine.

To install nox this way:

1. Install pipx, e.g. with the package manager [brew](https://github.com/pypa/pipx#on-macos)
(and [brew works on Linux too](https://docs.brew.sh/Homebrew-on-Linux))

2. Install nox with pipx: `pipx install nox`

For other ways to install nox, please see:
https://nox.thea.codes/en/stable/tutorial.html#installation

### Steps to set up a development environment

#### 1. Clone the repository

Clone the repository from Github using git

```shell
git clone https://github.com/vocalpy/vak
```

#### 2. Create a virtual environment with the development dependencies

The repository includes code that automates setting up a development environment,
using nox as described above.

To create a virtual environment, run the following:

```shell
nox -s dev
```

You can then activate the virtual environment by executing:

```shell
. ./.venv/activate
```

on MacOS and Linux, or

```shell
./.venv/activate.bat
```

on Windows.

#### 3. Download test data

There are three types of data of needed for tests.

1. .toml configuration files
2. source test data
3. generated test data

The .toml configuration files are under version control,
so you will already have them when you clone the repository.
The other two types of data are made up of files
that are too large to keep in a GitHub repository.
Instead, the files are kept on a public
[Open Science Framework](https://osf.io) project, here:
<https://osf.io/vz48c/>
There are `nox` sessions that download this data.
Next we define these two other types of test data.

##### Source test data

The source data consists of files used as input to vak
such as audio and annotation files. These files are less likely to change
as vak develops, so they are kept separate.

To download these files, run:

```shell
nox -s test-data-download-source
```

##### Generated test data

Generated test data consists of files created by vak,
such as .csv files that represent datasets,
and saved neural network checkpoints.

This test data set is generated by a script: `./tests/scripts/generate_data_for_tests.py`.
Generally speaking, the core maintainers are the only ones that should need
to run this script.

To download these files, run:

```shell
nox -s test-data-download-generated-all
```

#### 4. Proceed with development

After completing these steps, you are ready for development!

## Contributing Code

### vak Code Overview


The source code for vak is located in the directory `./src/vak`. When contributing
code, be sure to follow the general guidelines in the
{ref}`dev-workflow` section.

### Code Style

In general, vak code should

In general, crowsetta code should

* follow the [Zen of Python](https://www.python.org/dev/peps/pep-0020/#id2) in terms of implementation
* follow the [PEP8 style guide](https://www.python.org/dev/peps/pep-0008/) for code
* follow the [numpy standard for docstrings](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard)

We also use the tool [Black](https://github.com/psf/black) to format the code, so we don't have to think about it.

### Testing your Code

Automated testing helps ensure that our code is as free of bugs as it can be.
It also lets us know immediately if a change we make breaks any other part of the code.

All of our test code and data are stored in the directory `./tests`,
that is set up as if it were a Python package.
We use the [pytest](https://pytest.org) framework to run the test suite.
While developing, you can run the entire test suite inside an
activated virtual environment by running `pytest` from the command line:

```shell
pytest
```

You can also run tests in just one test script using:

```shell
pytest ./tests/NAME_OF_TEST_FILE.py
```

For more on specifying which test to run with pytest, see
[this page](https://docs.pytest.org/en/7.1.x/how-to/usage.html#specifying-which-tests-to-run) 
in their documentation.

When you are ready to make a pull request,
it is good practice to run the entire test suite
in a newly-created virtual environment.
There is a command in the `nox` file that automates this for you.

```shell
nox -s test
```

Please write tests for your code so that we can be sure that it won't break any of the
existing functionality.
Tests also help us be confident that we won't break your code in the future.

If you're **new to testing**, see existing test files for examples of things to do.
**Don't let the tests keep you from submitting your contribution!**
If you're not sure how to do this or are having trouble, submit your pull request
anyway.
We will help you create the tests and sort out any kind of problem during code review.
It's OK if you can't or don't know how to test something.
Leave a comment in the pull request and we'll help you out.
