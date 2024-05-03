"""fixtures relating to .toml configuration files"""
import json
import shutil

import pytest
import tomlkit

from .test_data import GENERATED_TEST_DATA_ROOT, TEST_DATA_ROOT


TEST_CONFIGS_ROOT = TEST_DATA_ROOT.joinpath("configs")


@pytest.fixture
def test_configs_root():
    """Path that points to data_for_tests/configs

    Two types of config files in this directory:
    1) those used by the tests/scripts/generate_data_for_tests.py script.
       Will be listed in configs.json. See ``specific_config_toml_path`` fixture below
       for details about types of configs.
    2) those used by tests that are static, e.g., ``invalid_table_config.toml``

    This fixture facilitates access to type (2), e.g. in test_config/test_parse
    """
    return TEST_CONFIGS_ROOT


@pytest.fixture
def list_of_schematized_configs(test_configs_root):
    """returns list of configuration files,
    schematized with attributes
    so that fixtures and unit tests can specify those attributes
    to find the filename of a specific configuration file,
    that can then be used to get that file.

    Each element in the list is a dict with the following keys:
    `filename`, `config_type`, `audio_format`, `spect_format`, `annot_format`
    These keys define the schema for config files.

    For example, here is the first one:
    {
      "filename": "test_eval_audio_cbin_annot_notmat.toml",
      "config_type": "eval",
      "audio_format": "cbin",
      "spect_format": null,
      "annot_format": "notmat"
    }

    The ``specific_config_toml_path`` factory fixture returns a function that
    itself return a configuration ``filename``, when provided values for
    all of the other keys.
    """
    with test_configs_root.joinpath("configs.json").open("r") as fp:
        return json.load(fp)["config_metadata"]


@pytest.fixture
def config_that_doesnt_exist(tmp_path):
    return tmp_path / "config_that_doesnt_exist.toml"


@pytest.fixture
def invalid_table_config_path(test_configs_root):
    return test_configs_root.joinpath("invalid_table_config.toml")


@pytest.fixture
def invalid_key_config_path(test_configs_root):
    return test_configs_root.joinpath("invalid_key_config.toml")


@pytest.fixture
def invalid_train_and_learncurve_config_toml(test_configs_root):
    return test_configs_root.joinpath("invalid_train_and_learncurve_config.toml")


GENERATED_TEST_CONFIGS_ROOT = GENERATED_TEST_DATA_ROOT.joinpath("configs")


@pytest.fixture
def generated_test_configs_root():
    return GENERATED_TEST_CONFIGS_ROOT


@pytest.fixture
def specific_config_toml_path(generated_test_configs_root, list_of_schematized_configs, tmp_path):
    """returns a factory function
    that will return the path
    to a specific configuration file, determined by
    characteristics specified by the caller:
    `config_type`, `audio_format`, `spect_format`, `annot_format`

    The factory function actually returns a copy,
    that will be copied into ``tmp_path``,
    so the original remains unmodified.

    If ``root_results_dir`` argument is specified
    when calling the factory function,
    it will convert the value for that key in the table
    corresponding to ``config_type`` to the value
    specified for ``root_results_dir``.
    This makes it possible to dynamically change the ``root_results_dir``
    e.g. to the ``tmp_path`` fixture used by unit tests
    """

    def _specific_config(
        config_type,
        model,
        annot_format,
        audio_format=None,
        spect_format=None,
        keys_to_change=None,
    ):
        """returns path to a specific configuration file,
        determined by characteristics specified by the caller:
        `config_type`, `audio_format`, `spect_format`, `annot_format`

        Parameters
        ----------
        config_type : str
            corresponding to a `vak` cli command
        annot_format : str
            annotation format, recognized by ``crowsetta``
        audio_format : str
        spect_format : str
        keys_to_change : list, dict
            list of dicts with keys 'table', 'key', and 'value'.
            Can be a single dict, in which case only that key is changed.
            If the 'value' is set to 'DELETE-KEY',
            the key will be removed from the config.
            This can be used to test behavior when the key is not set.

        Returns
        -------
        config_path : pathlib.Path
            that points to temporary copy of specified config,
            with any keys changed as specified
        """
        original_config_path = None
        for schematized_config in list_of_schematized_configs:
            if all(
                [
                    schematized_config["config_type"] == config_type,
                    schematized_config["model"] == model,
                    schematized_config["annot_format"] == annot_format,
                    schematized_config["audio_format"] == audio_format,
                    schematized_config["spect_format"] == spect_format,
                ]
            ):
                original_config_path = generated_test_configs_root.joinpath(
                    schematized_config["filename"]
                )
                break

        if original_config_path is None:
            raise ValueError(
                f"did not find a specific config with `config_type`='{config_type}', "
                f"`model`={model}, `annot_format`='{annot_format}', "
                f"`audio_format`='{audio_format}', and `spect_type`='{spect_format}'."
            )
        config_copy_path = tmp_path.joinpath(original_config_path.name)
        config_copy_path = shutil.copy(src=original_config_path, dst=config_copy_path)

        if keys_to_change is not None:
            if isinstance(keys_to_change, dict):
                keys_to_change = [keys_to_change]
            elif isinstance(keys_to_change, list):
                pass
            else:
                raise TypeError(
                    f"invalid type for `keys_to_change`: {type(keys_to_change)}"
                )

            with config_copy_path.open("r") as fp:
                tomldoc = tomlkit.load(fp)

            for table_key_val_dict in keys_to_change:
                table_name = table_key_val_dict["table"]
                key = table_key_val_dict["key"]
                value = table_key_val_dict["value"]
                if isinstance(key, str):
                    if table_key_val_dict["value"] == 'DELETE-KEY':
                        # e.g., to test behavior of config without this key
                        del tomldoc["vak"][table_name][key]
                    else:
                        tomldoc["vak"][table_name][key] = value
                elif isinstance(key, list) and len(key) == 2 and all([isinstance(el, str) for el in key]):
                    # for the case where we need to access a sub-table
                    # right now this applies mainly to ["vak"][table]["dataset"]["path"]
                    # if we end up having to access more / deeper then we'll need something more general
                    if table_key_val_dict["value"] == 'DELETE-KEY':
                        # e.g., to test behavior of config without this key
                        del tomldoc["vak"][table_name][key[0]][key[1]]
                    else:
                        tomldoc["vak"][table_name][key[0]][key[1]] = value
                else:
                    raise ValueError(
                        f"Unexpected value for 'key' in `keys_to_change` dict: {key}.\n"
                        f"`keys_to_change` dict: {table_key_val_dict}"
                    )

            with config_copy_path.open("w") as fp:
                tomlkit.dump(tomldoc, fp)

        return config_copy_path

    return _specific_config


ALL_GENERATED_CONFIG_PATHS = sorted(GENERATED_TEST_CONFIGS_ROOT.glob("*toml"))


# ---- path to config files ----
@pytest.fixture(params=ALL_GENERATED_CONFIG_PATHS)
def a_generated_config_path(request):
    return request.param


def _tomlkit_to_popo(d):
    """Convert tomlkit to "popo" (Plain-Old Python Objects)

    From https://github.com/python-poetry/tomlkit/issues/43#issuecomment-660415820
    """
    try:
        result = getattr(d, "value")
    except AttributeError:
        result = d

    if isinstance(result, list):
        result = [_tomlkit_to_popo(x) for x in result]
    elif isinstance(result, dict):
        result = {
            _tomlkit_to_popo(key): _tomlkit_to_popo(val) for key, val in result.items()
        }
    elif isinstance(result, tomlkit.items.Integer):
        result = int(result)
    elif isinstance(result, tomlkit.items.Float):
        result = float(result)
    elif isinstance(result, tomlkit.items.String):
        result = str(result)
    elif isinstance(result, tomlkit.items.Bool):
        result = bool(result)

    return result


# ----  config dicts from paths ----
def _load_config_dict(toml_path):
    """Return config as dict, loaded from toml file.

    Used to test functions that parse config tables, taking these dicts as inputs.

    Note that we access the topmost table loaded from the toml: config_dict['vak']
    """
    with toml_path.open("r") as fp:
        config_dict = tomlkit.load(fp)
    return _tomlkit_to_popo(config_dict['vak'])


@pytest.fixture
def specific_config_toml(specific_config_toml_path):
    """returns a function that will return a dict
    containing parsed toml from a
    specific configuration file, determined by
    characteristics specified by the caller:
    `config_type`, `audio_format`, `spect_format`, `annot_format`
    """

    def _specific_config_toml(
        config_type,
        model,
        annot_format,
        audio_format=None,
        spect_format=None,
    ):
        config_path = specific_config_toml_path(
            config_type, model, annot_format, audio_format, spect_format
        )
        return _load_config_dict(config_path)

    return _specific_config_toml


@pytest.fixture(params=ALL_GENERATED_CONFIG_PATHS)
def a_generated_config_dict(request):
    # we remake dict every time this gets called
    # so that we're not returning a ``config_dict`` that was
    # already mutated by a `Config.from_config_dict` function,
    # e.g. the value for the 'spect_params' key gets mapped to a SpectParamsConfig
    # by PrepConfig.from_config_dict
    return _load_config_dict(request.param)


ALL_GENERATED_EVAL_CONFIG_PATHS = sorted(
    GENERATED_TEST_CONFIGS_ROOT.glob("*eval*toml")
)

ALL_GENERATED_LEARNCURVE_CONFIG_PATHS = sorted(
    GENERATED_TEST_CONFIGS_ROOT.glob("*learncurve*toml")
)

ALL_GENERATED_PREDICT_CONFIG_PATHS = sorted(
    GENERATED_TEST_CONFIGS_ROOT.glob("*predict*toml")
)

ALL_GENERATED_TRAIN_CONFIG_PATHS = sorted(
    GENERATED_TEST_CONFIGS_ROOT.glob("*train*toml")
)

# as above, we remake dict every time these fixutres get called
# so that we're not returning a ``config_dict`` that was
# already mutated by a `Config.from_config_dict` function,
# e.g. the value for the 'spect_params' key gets mapped to a SpectParamsConfig
# by PrepConfig.from_config_dict
@pytest.fixture(params=ALL_GENERATED_EVAL_CONFIG_PATHS)
def a_generated_eval_config_dict(request):
    return _load_config_dict(request.param)


@pytest.fixture(params=ALL_GENERATED_LEARNCURVE_CONFIG_PATHS)
def a_generated_learncurve_config_dict(request):
    return _load_config_dict(request.param)


@pytest.fixture(params=ALL_GENERATED_PREDICT_CONFIG_PATHS)
def a_generated_predict_config_dict(request):
    return _load_config_dict(request.param)


@pytest.fixture(params=ALL_GENERATED_TRAIN_CONFIG_PATHS)
def a_generated_train_config_dict(request):
    return _load_config_dict(request.param)


@pytest.fixture
def all_generated_learncurve_configs_toml(all_generated_learncurve_configs):
    return [_load_config_dict(config) for config in all_generated_learncurve_configs]


ALL_GENERATED_CONFIGS_TOML_PATH_PAIRS = list(zip(
    [_load_config_dict(config) for config in ALL_GENERATED_CONFIG_PATHS],
    ALL_GENERATED_CONFIG_PATHS,
))


# ---- config toml + path pairs ----
@pytest.fixture
def all_generated_configs_toml_path_pairs():
    """zip of tuple pairs: (dict, pathlib.Path)
    where ``Path`` is path to .toml config file and ``dict`` is
    the .toml config from that path
    loaded into a dict with the ``toml`` library
    """
    # we duplicate the constant above because we need to remake
    # the variables for each unit test. Otherwise tests that modify values
    # for config keys cause other tests to fail
    return zip(
        [_load_config_dict(config) for config in ALL_GENERATED_CONFIG_PATHS],
        ALL_GENERATED_CONFIG_PATHS
    )


@pytest.fixture
def configs_toml_path_pairs_by_model_factory(all_generated_configs_toml_path_pairs):
    """
    factory fixture that returns all generated configs and their paths for a specified model
    """

    def _wrapped(model,
                 table_name=None):
        out = []
        unzipped = list(all_generated_configs_toml_path_pairs)
        for config_toml, toml_path in unzipped:
            if toml_path.name.startswith(model):
                if table_name:
                    if table_name.lower() in toml_path.name:
                        out.append(
                            (config_toml, toml_path)
                        )
                else:
                    out.append(
                     (config_toml, toml_path)
                    )
        return out

    return _wrapped

