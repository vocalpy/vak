import json
import pathlib


from . import fixtures
# keep this import here, we need it for fixtures
from .fixtures import *


with pathlib.Path('./tests/vak.tests.config.json').open('r') as fp:
    VAK_TESTS_CONFIG_TOML = json.load(fp)
DEFAULT_MODELS = VAK_TESTS_CONFIG_TOML['models']


def by_slow_marker(item):
    return 1 if item.get_closest_marker('slow') is None else 0



def pytest_addoption(parser):
    parser.addoption(
        "--models",
        action="store",
        default=DEFAULT_MODELS,
        nargs="+",
        help="vak models to test, space-separated list of names",
    )
    parser.addoption('--dtype', action="store", default="float32")
    parser.addoption('--slow-last', action='store_true', default=False)


def pytest_collection_modifyitems(items, config):
    if config.getoption('--slow-last'):
        items.sort(key=by_slow_marker, reverse=True)


def pytest_generate_tests(metafunc):
    models = metafunc.config.option.models
    if isinstance(models, str):
        # wrap a single model name in a list
        models = [models]
    # **note!** fixture name is singular even though cmdopt is plural
    if "model" in metafunc.fixturenames and models is not None:
        metafunc.parametrize("model", models)

    if 'dtype_name' in metafunc.fixturenames:
        raw_value = metafunc.config.getoption('--dtype')
        if raw_value == 'all':
            dtype_names = list(fixtures.torch.TEST_DTYPES.keys())
        else:
            dtype_names = raw_value.split(',')
        if dtype_names is not None:
            metafunc.parametrize('dtype_name', dtype_names)
