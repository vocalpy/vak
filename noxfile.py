import json
import os
import pathlib
import shutil
import tarfile
import urllib.request

import nox


DIR = pathlib.Path(__file__).parent.resolve()
VENV_DIR = pathlib.Path('./.venv').resolve()


with pathlib.Path('./tests/vak.tests.config.json').open('rb') as fp:
    VAK_TESTS_CONFIG = json.load(fp)


nox.options.sessions = ['test', 'coverage']


@nox.session
def build(session: nox.Session) -> None:
    """
    Build an SDist and wheel with ``flit``.
    """
    dist_dir = DIR.joinpath("dist")
    if dist_dir.exists():
        shutil.rmtree(dist_dir)

    session.install(".[dev]")
    session.run("flit", "build")


@nox.session(python="3.10.7")
def dev(session: nox.Session) -> None:
    """
    Sets up a python development environment for the project.

    This session will:
    - Create a python virtualenv for the session
    - Install the `virtualenv` cli tool into this environment
    - Use `virtualenv` to create a global project virtual environment
    - Invoke the python interpreter from the global project environment to install
      the project and all it's development dependencies.
    """
    session.install("virtualenv")
    # VENV_DIR here is a pathlib.Path location of the project virtualenv
    # e.g. .venv
    session.run("virtualenv", os.fsdecode(VENV_DIR), silent=True)

    python = os.fsdecode(VENV_DIR.joinpath("bin/python"))

    # Use the venv's interpreter to install the project along with
    # all it's dev dependencies, this ensures it's installed in the right way
    session.run(python, "-m", "pip", "install", "-e", ".[dev,test,doc]", external=True)


@nox.session(python="3.10")
def lint(session):
    """
    Run the linter.
    """
    session.install("isort", "black", "flake8")
    # run isort first since black disagrees with it
    session.run("isort", "./src")
    session.run("black", "./src", "--line-length=79")
    session.run("flake8", "./src", "--max-line-length", "120")


TEST_PYTHONS = [
    "3.10",
    "3.11",
    "3.12",
]


@nox.session(python=TEST_PYTHONS)
def test(session) -> None:
    """
    Run the unit and regular tests.
    """
    session.install(".[test]")
    if session.posargs:
        session.run("pytest", *session.posargs)
    else:
        session.run("pytest", "-x", "--slow-last")


@nox.session
def coverage(session) -> None:
    """
    Run the unit and regular tests, and save coverage report
    """
    session.install(".[test]")
    session.run(
        "pytest", "--slow-last", "--cov=./", "--cov-report=xml", *session.posargs
    )


@nox.session
def doc(session: nox.Session) -> None:
    """
    Build the docs.

    To run ``sphinx-autobuild``,  do:

    .. code-block::console

       nox -s doc -- autobuild

    Otherwise the docs will be built once using
    """
    session.install(".[doc]")
    if session.posargs:
        if "autobuild" in session.posargs:
            print("Building docs at http://127.0.0.1:8000 with sphinx-autobuild -- use Ctrl-C to quit")
            session.run("sphinx-autobuild", "doc", "doc/_build/html")
        else:
            print("Unsupported argument to docs")
    else:
        session.run("sphinx-build", "-nW", "--keep-going", "-b", "html", "doc/", "doc/_build/html")


# ---- sessions below this all have to do with data for tests ----------------------------------------------------
def clean_dir(dir_path):
    """Helper function that "cleans" a directory by removing all files
    (that are not hidden) without removing the directory itself."""
    dir_path = pathlib.Path(dir_path)
    dir_contents = dir_path.glob('*')
    for content in dir_contents:
        if content.is_dir():
            shutil.rmtree(content)
        else:
            if content.name.startswith('.'):
                # e.g., .gitkeep file we don't want to delete
                continue
            content.unlink()


DATA_FOR_TESTS_DIR = './tests/data_for_tests/'
SOURCE_TEST_DATA_DIR = f"{DATA_FOR_TESTS_DIR}source/"
SOURCE_TEST_DATA_DIRS = [
    dir_ for dir_
    in sorted(pathlib.Path(SOURCE_TEST_DATA_DIR).glob('*/'))
    if dir_.is_dir()
]


@nox.session(name='test-data-clean-source')
def test_data_clean_source(session) -> None:
    """Clean (remove) 'source' test data, used by TEST_DATA_GENERATE_SCRIPT."""
    clean_dir(SOURCE_TEST_DATA_DIR)


def copy_url(url: str, path: str) -> None:
    """Copy data from a url to a local file."""
    urllib.request.urlretrieve(url, path)


SOURCE_TEST_DATA_URL = 'https://osf.io/2ehbp/download'
SOURCE_TEST_DATA_TAR = f'{SOURCE_TEST_DATA_DIR}source_test_data-version-1.x.tar.gz'


@nox.session(name='test-data-tar-source')
def test_data_tar_source(session) -> None:
    """Make a .tar.gz file of just the 'source' test data used to run tests."""
    session.log(f"Making tarfile with source data: {SOURCE_TEST_DATA_TAR}")
    make_tarfile(SOURCE_TEST_DATA_TAR, SOURCE_TEST_DATA_DIRS)


@nox.session(name='test-data-download-source')
def test_data_download_source(session) -> None:
    """Download and extract a .tar.gz file of 'source' test data, used by TEST_DATA_GENERATE_SCRIPT."""
    session.log(f'Downloading: {SOURCE_TEST_DATA_URL}')
    copy_url(url=SOURCE_TEST_DATA_URL, path=SOURCE_TEST_DATA_TAR)
    session.log(f'Extracting downloaded tar: {SOURCE_TEST_DATA_TAR}')
    with tarfile.open(SOURCE_TEST_DATA_TAR, "r:gz") as tf:
        tf.extractall(path='.')


TEST_DATA_GENERATE_SCRIPT = './tests/scripts/generate_data_for_tests.py'


@nox.session(name='test-data-generate', python="3.10")
def test_data_generate(session) -> None:
    """Produced 'generated' test data, by running TEST_DATA_GENERATE_SCRIPT on 'source' test data."""
    session.install(".[test]")
    session.run("python", TEST_DATA_GENERATE_SCRIPT)


GENERATED_TEST_DATA_DIR = f'{DATA_FOR_TESTS_DIR}generated/'


@nox.session(name='test-data-clean-generated')
def test_data_clean_generated(session) -> None:
    """Clean (remove) 'generated' test data."""
    clean_dir(GENERATED_TEST_DATA_DIR)


def make_tarfile(name: str, to_add: list):
    """Helper function that makes a tarfile"""
    with tarfile.open(name, "w:gz") as tf:
        for add_name in to_add:
            tf.add(name=add_name)


CONFIGS_DIR = f'{GENERATED_TEST_DATA_DIR}configs'
PREP_DIR = f'{GENERATED_TEST_DATA_DIR}prep/'
RESULTS_DIR = f'{GENERATED_TEST_DATA_DIR}results/'

PREP_CI: list = []
for model_name in VAK_TESTS_CONFIG['models']:
    PREP_CI.extend(
        sorted(
            pathlib.Path(PREP_DIR).glob(f'*/*/{model_name}')
                 )
    )
RESULTS_CI: list = []
for model_name in VAK_TESTS_CONFIG['models']:
    PREP_CI.extend(
        sorted(
            pathlib.Path(RESULTS_DIR).glob(f'*/*/{model_name}')
                 )
    )

GENERATED_TEST_DATA_CI_TAR = f'{GENERATED_TEST_DATA_DIR}generated_test_data-version-1.x.ci.tar.gz'
GENERATED_TEST_DATA_CI_DIRS = [CONFIGS_DIR] + PREP_CI + RESULTS_CI

GENERATED_TEST_DATA_ALL_TAR = f'{GENERATED_TEST_DATA_DIR}generated_test_data-version-1.x.tar.gz'
GENERATED_TEST_DATA_ALL_DIRS = [CONFIGS_DIR, PREP_DIR, RESULTS_DIR]


@nox.session(name='test-data-tar-generated-all')
def test_data_tar_generated_all(session) -> None:
    """Make a .tar.gz file of all 'generated' test data."""
    session.log(f"Making tarfile with all generated data: {GENERATED_TEST_DATA_ALL_TAR}")
    make_tarfile(GENERATED_TEST_DATA_ALL_TAR, GENERATED_TEST_DATA_ALL_DIRS)


@nox.session(name='test-data-tar-generated-ci')
def test_data_tar_generated_ci(session) -> None:
    """Make a .tar.gz file of just the 'generated' test data used to run tests on CI."""
    session.log(f"Making tarfile with generated data for CI: {GENERATED_TEST_DATA_CI_TAR}")
    make_tarfile(GENERATED_TEST_DATA_CI_TAR, GENERATED_TEST_DATA_CI_DIRS)


GENERATED_TEST_DATA_ALL_URL = 'https://osf.io/xfp6n/download'


@nox.session(name='test-data-download-generated-all')
def test_data_download_generated_all(session) -> None:
    """Download and extract a .tar.gz file of all 'generated' test data"""
    session.install("pandas")
    session.log(f'Downloading: {GENERATED_TEST_DATA_ALL_URL}')
    copy_url(url=GENERATED_TEST_DATA_ALL_URL, path=GENERATED_TEST_DATA_ALL_TAR)
    session.log(f'Extracting downloaded tar: {GENERATED_TEST_DATA_ALL_TAR}')
    with tarfile.open(GENERATED_TEST_DATA_ALL_TAR, "r:gz") as tf:
        tf.extractall(path='.')
    session.log('Fixing paths in .csv files')
    session.install("pandas")


GENERATED_TEST_DATA_CI_URL = 'https://osf.io/un2zs/download'


@nox.session(name='test-data-download-generated-ci')
def test_data_download_generated_ci(session) -> None:
    """Download and extract a .tar.gz file of just the 'generated' test data used to run tests on CI"""
    session.install("pandas")
    session.log(f'Downloading: {GENERATED_TEST_DATA_CI_URL}')
    copy_url(url=GENERATED_TEST_DATA_CI_URL, path=GENERATED_TEST_DATA_CI_TAR)
    session.log(f'Extracting downloaded tar: {GENERATED_TEST_DATA_CI_TAR}')
    with tarfile.open(GENERATED_TEST_DATA_CI_TAR, "r:gz") as tf:
        tf.extractall(path='.')
