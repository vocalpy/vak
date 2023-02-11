import os
import pathlib
import shutil
import tarfile
import urllib.request

import nox


DIR = pathlib.Path(__file__).parent.resolve()
VENV_DIR = pathlib.Path('./.venv').resolve()

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


@nox.session
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


# ---- used by sessions that "clean up" data for tests
def clean_dir(dir_path):
    """
    "clean" a directory by removing all files
    (that are not hidden)
    without removing the directory itself
    """
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
    """
    Clean (remove) 'source' test data, used by TEST_DATA_GENERATE_SCRIPT.
    """
    clean_dir(SOURCE_TEST_DATA_DIR)


def copy_url(url: str, path: str) -> None:
    """Copy data from a url to a local file."""
    urllib.request.urlretrieve(url, path)


SOURCE_TEST_DATA_URL = 'https://osf.io/hbg4k/download'
SOURCE_TEST_DATA_TAR = f'{SOURCE_TEST_DATA_DIR}source_test_data.tar.gz'


@nox.session(name='test-data-tar-source')
def test_data_tar_source(session) -> None:
    """
    Make a .tar.gz file of just the 'generated' test data used to run tests on CI.
    """
    session.log(f"Making tarfile with source data: {SOURCE_TEST_DATA_TAR}")
    make_tarfile(SOURCE_TEST_DATA_TAR, SOURCE_TEST_DATA_DIRS)


@nox.session(name='test-data-download-source')
def test_data_download_source(session) -> None:
    """
    Download and extract a .tar.gz file of 'source' test data, used by TEST_DATA_GENERATE_SCRIPT.
    """
    session.log(f'Downloading: {SOURCE_TEST_DATA_URL}')
    copy_url(url=SOURCE_TEST_DATA_URL, path=SOURCE_TEST_DATA_TAR)
    session.log(f'Extracting downloaded tar: {SOURCE_TEST_DATA_TAR}')
    with tarfile.open(SOURCE_TEST_DATA_TAR, "r:gz") as tf:
        tf.extractall(path='.')


TEST_DATA_GENERATE_SCRIPT = './tests/scripts/generate_data_for_tests.py'


@nox.session(name='test-data-generate')
def test_data_generate(session) -> None:
    """
    Produced 'generated' test data, by running TEST_DATA_GENERATE_SCRIPT on 'source' test data.
    """
    session.install(".[test]")
    session.run("python", TEST_DATA_GENERATE_SCRIPT)


GENERATED_TEST_DATA_DIR = f'{DATA_FOR_TESTS_DIR}generated/'


@nox.session(name='test-data-clean-generated')
def test_data_clean_generated(session) -> None:
    """
    Clean (remove) 'generated' test data.
    """
    clean_dir(GENERATED_TEST_DATA_DIR)


def make_tarfile(name: str, to_add: list):
    with tarfile.open(name, "w:gz") as tf:
        for add_name in to_add:
            tf.add(name=add_name)


CONFIGS_DIR = f'{GENERATED_TEST_DATA_DIR}configs'
PREP_DIR = f'{GENERATED_TEST_DATA_DIR}prep/'
RESULTS_DIR = f'{GENERATED_TEST_DATA_DIR}results/'

PREP_CI = sorted(pathlib.Path(PREP_DIR).glob('*/*/teenytweetynet'))
RESULTS_CI = sorted(pathlib.Path(RESULTS_DIR).glob('*/*/teenytweetynet'))
GENERATED_TEST_DATA_CI_TAR = f'{GENERATED_TEST_DATA_DIR}generated_test_data-version-0.x.ci.tar.gz'
GENERATED_TEST_DATA_CI_DIRS = [CONFIGS_DIR] + PREP_CI + RESULTS_CI

GENERATED_TEST_DATA_ALL_TAR = f'{GENERATED_TEST_DATA_DIR}generated_test_data-version-0.x.tar.gz'
GENERATED_TEST_DATA_ALL_DIRS = [CONFIGS_DIR, PREP_DIR, RESULTS_DIR]


@nox.session(name='test-data-tar-generated-all')
def test_data_tar_generated_all(session) -> None:
    """
    Make a .tar.gz file of all 'generated' test data.
    """
    session.log(f"Making tarfile with all generated data: {GENERATED_TEST_DATA_ALL_TAR}")
    make_tarfile(GENERATED_TEST_DATA_ALL_TAR, GENERATED_TEST_DATA_ALL_DIRS)


@nox.session(name='test-data-tar-generated-ci')
def test_data_tar_generated_ci(session) -> None:
    """
    Make a .tar.gz file of just the 'generated' test data used to run tests on CI.
    """
    session.log(f"Making tarfile with generated data for CI: {GENERATED_TEST_DATA_CI_TAR}")
    make_tarfile(GENERATED_TEST_DATA_CI_TAR, GENERATED_TEST_DATA_CI_DIRS)


GENERATED_TEST_DATA_ALL_URL = 'https://osf.io/532cs/download'


@nox.session(name='test-data-download-generated-all')
def test_data_download_generated_all(session) -> None:
    """
    Download and extract a .tar.gz file of all 'generated' test data
    """
    session.install("pandas")
    session.log(f'Downloading: {GENERATED_TEST_DATA_ALL_URL}')
    copy_url(url=GENERATED_TEST_DATA_ALL_URL, path=GENERATED_TEST_DATA_ALL_TAR)
    session.log(f'Extracting downloaded tar: {GENERATED_TEST_DATA_ALL_TAR}')
    with tarfile.open(GENERATED_TEST_DATA_ALL_TAR, "r:gz") as tf:
        tf.extractall(path='.')
    session.log('Fixing paths in .csv files')
    session.run(
        "python", "./tests/scripts/fix_prep_csv_paths.py"
    )


GENERATED_TEST_DATA_CI_URL = 'https://osf.io/g79sx/download'


@nox.session(name='test-data-download-generated-ci')
def test_data_download_generated_ci(session) -> None:
    """
    Download and extract a .tar.gz file of just the 'generated' test data used to run tests on CI
    """
    session.install("pandas")
    session.log(f'Downloading: {GENERATED_TEST_DATA_CI_URL}')
    copy_url(url=GENERATED_TEST_DATA_CI_URL, path=GENERATED_TEST_DATA_CI_TAR)
    session.log(f'Extracting downloaded tar: {GENERATED_TEST_DATA_CI_TAR}')
    with tarfile.open(GENERATED_TEST_DATA_CI_TAR, "r:gz") as tf:
        tf.extractall(path='.')
    session.log('Fixing paths in .csv files')
    session.run(
        "python", "./tests/scripts/fix_prep_csv_paths.py"
    )


@nox.session
def test(session) -> None:
    """
    Run the unit and regular tests.
    """
    session.install(".[test]")
    session.run("pytest", *session.posargs)


@nox.session
def coverage(session) -> None:
    """
    Run the unit and regular tests, and save coverage report
    """
    session.install(".[test]")
    if session.posargs:
        if "running-on-ci" in session.posargs:
            # on ci, just run `teenytweetynet` model
            session.run(
                "pytest", "--models", "teenytweetynet", "--cov=./", "--cov-report=xml"
            )
            return
        else:
            print("Unsupported argument to coverage")

    session.run(
        "pytest", "--cov=./", "--cov-report=xml", *session.posargs
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
