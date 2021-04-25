from . import fixtures
from .fixtures import *


def pytest_addoption(parser):
    parser.addoption(
        "--models",
        action="store",
        default="teenytweetynet",
        nargs="+",
        help="vak models to test, space-separated list of names",
    )
    parser.addoption('--dtype', action="store", default="float32")


def pytest_generate_tests(metafunc):
    models = metafunc.config.option.models
    if isinstance(models, str):
        # wrap a single model name in a list
        models = [models]
    # **note!** fixture name is singular even though cmdopt is plural
    if "model" in metafunc.fixturenames and models is not None:
        metafunc.parametrize("model", models)

    dtype_names = None
    if 'dtype_name' in metafunc.fixturenames:
        raw_value = metafunc.config.getoption('--dtype')
        if raw_value == 'all':
            dtype_names = list(fixtures.torch.TEST_DTYPES.keys())
        else:
            dtype_names = raw_value.split(',')
        if dtype_names is not None:
            metafunc.parametrize('dtype_name', dtype_names)
