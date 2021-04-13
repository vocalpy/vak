from .fixtures import *


def pytest_addoption(parser):
    parser.addoption(
        "--models",
        action="store",
        default="teenytweetynet",
        nargs="+",
        help="vak models to test, space-separated list of names",
    )


def pytest_generate_tests(metafunc):
    models = metafunc.config.option.models
    if isinstance(models, str):
        # wrap a single model name in a list
        models = [models]
    # **note!** fixture name is singular even though cmdopt is plural
    if "model" in metafunc.fixturenames and models is not None:
        metafunc.parametrize("model", models)
