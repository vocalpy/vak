"""fixtures relating to .toml configuration files"""
import json
import shutil

import pytest
import toml

from .test_data import GENERATED_TEST_DATA_ROOT


@pytest.fixture
def test_configs_root(test_data_root):
    """Path that points to data_for_tests/configs

    Two types of config files in this directory:
    1) those used by the tests/scripts/generate_data_for_tests.py script.
       Will be listed in configs.json. See ``specific_config`` fixture below
       for details about types of configs.
    2) those used by tests that are static, e.g., ``invalid_section_config.toml``

    This fixture facilitates access to type (2), e.g. in test_config/test_parse
    """
    return test_data_root.joinpath("configs")


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

    The ``specific_config`` factory fixture returns a function that
    itself return a configuration ``filename``, when provided values for
    all of the other keys.
    """
    with test_configs_root.joinpath("configs.json").open("r") as fp:
        return json.load(fp)["configs"]


@pytest.fixture
def config_that_doesnt_exist(tmp_path):
    return tmp_path / "config_that_doesnt_exist.toml"


@pytest.fixture
def invalid_section_config_path(test_configs_root):
    return test_configs_root.joinpath("invalid_section_config.toml")


@pytest.fixture
def invalid_option_config_path(test_configs_root):
    return test_configs_root.joinpath("invalid_option_config.toml")


GENERATED_TEST_CONFIGS_ROOT = GENERATED_TEST_DATA_ROOT.joinpath("configs")


@pytest.fixture
def generated_test_configs_root():
    return GENERATED_TEST_CONFIGS_ROOT


# ---- path to config files ----
@pytest.fixture
def all_generated_configs(generated_test_configs_root):
    return sorted(generated_test_configs_root.glob("*toml"))


@pytest.fixture
def specific_config(generated_test_configs_root, list_of_schematized_configs, tmp_path):
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
    it will convert the value for that option in the section
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
        options_to_change=None,
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
        options_to_change : list, dict
            list of dicts with keys 'section', 'option', and 'value'.
            Can be a single dict, in which case only that option is changed.
            If the 'value' is set to 'DELETE-OPTION',
            the option will be removed from the config.
            This can be used to test behavior when the option is not set.

        Returns
        -------
        config_path : pathlib.Path
            that points to temporary copy of specified config,
            with any options changed as specified
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

        if original_config_path is None:
            raise ValueError(
                f"did not find a specific config with `config_type`='{config_type}', "
                f"`model`={model}, `annot_format`='{annot_format}', "
                f"`audio_format`='{audio_format}', and `spect_type`='{spect_format}'."
            )
        config_copy_path = tmp_path.joinpath(original_config_path.name)
        config_copy_path = shutil.copy(src=original_config_path, dst=config_copy_path)

        if options_to_change is not None:
            if isinstance(options_to_change, dict):
                options_to_change = [options_to_change]
            elif isinstance(options_to_change, list):
                pass
            else:
                raise TypeError(
                    f"invalid type for `options_to_change`: {type(options_to_change)}"
                )

            with config_copy_path.open("r") as fp:
                config_toml = toml.load(fp)

            for opt_dict in options_to_change:
                if opt_dict["value"] == 'DELETE-OPTION':
                    # e.g., to test behavior of config without this option
                    del config_toml[opt_dict["section"]][opt_dict["option"]]
                else:
                    config_toml[opt_dict["section"]][opt_dict["option"]] = opt_dict["value"]

            with config_copy_path.open("w") as fp:
                toml.dump(config_toml, fp)

        return config_copy_path

    return _specific_config


@pytest.fixture
def all_generated_train_configs(generated_test_configs_root):
    return sorted(generated_test_configs_root.glob("test_train*toml"))


@pytest.fixture
def all_generated_learncurve_configs(generated_test_configs_root):
    return sorted(generated_test_configs_root.glob("test_learncurve*toml"))


@pytest.fixture
def all_generated_eval_configs(generated_test_configs_root):
    return sorted(generated_test_configs_root.glob("test_eval*toml"))


@pytest.fixture
def all_generated_predict_configs(generated_test_configs_root):
    return sorted(generated_test_configs_root.glob("test_predict*toml"))


# ----  config toml from paths ----
def _return_toml(toml_path):
    """return config files loaded into dicts with toml library
    used to test functions that parse config sections, taking these dicts as inputs"""
    with toml_path.open("r") as fp:
        config_toml = toml.load(fp)
    return config_toml


@pytest.fixture
def specific_config_toml(specific_config):
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
        config_path = specific_config(
            config_type, model, annot_format, audio_format, spect_format
        )
        return _return_toml(config_path)

    return _specific_config_toml


@pytest.fixture
def all_generated_configs_toml(all_generated_configs):
    return [_return_toml(config) for config in all_generated_configs]


@pytest.fixture
def all_generated_train_configs_toml(all_generated_train_configs):
    return [_return_toml(config) for config in all_generated_train_configs]


@pytest.fixture
def all_generated_learncurve_configs_toml(all_generated_learncurve_configs):
    return [_return_toml(config) for config in all_generated_learncurve_configs]


@pytest.fixture
def all_generated_eval_configs_toml(all_generated_eval_configs):
    return [_return_toml(config) for config in all_generated_eval_configs]


@pytest.fixture
def all_generated_predict_configs_toml(all_generated_predict_configs):
    return [_return_toml(config) for config in all_generated_predict_configs]


# ---- config toml + path pairs ----
@pytest.fixture
def all_generated_configs_toml_path_pairs(all_generated_configs):
    """zip of tuple pairs: (dict, pathlib.Path)
    where ``Path`` is path to .toml config file and ``dict`` is
    the .toml config from that path
    loaded into a dict with the ``toml`` library
    """
    return zip(
        [_return_toml(config) for config in all_generated_configs],
        all_generated_configs,
    )


@pytest.fixture
def configs_toml_path_pairs_by_model_factory(all_generated_configs_toml_path_pairs):
    """
    factory fixture that returns all generated configs and their paths for a specified model
    """

    def _wrapped(model,
                 section_name=None):
        out = []
        unzipped = list(all_generated_configs_toml_path_pairs)
        for config_toml, toml_path in unzipped:
            if toml_path.name.startswith(model):
                if section_name:
                    if section_name.lower() in toml_path.name:
                        out.append(
                            (config_toml, toml_path)
                        )
                else:
                    out.append(
                     (config_toml, toml_path)
                    )
        return out

    return _wrapped


@pytest.fixture
def all_generated_train_configs_toml_path_pairs(all_generated_train_configs):
    """zip of tuple pairs: (dict, pathlib.Path)
    where ``Path`` is path to .toml config file and ``dict`` is
    the .toml config from that path
    loaded into a dict with the ``toml`` library
    """
    return zip(
        [_return_toml(config) for config in all_generated_train_configs],
        all_generated_train_configs,
    )


@pytest.fixture
def all_generated_learncurve_configs_toml_path_pairs(all_generated_learncurve_configs):
    """zip of tuple pairs: (dict, pathlib.Path)
    where ``Path`` is path to .toml config file and ``dict`` is
    the .toml config from that path
    loaded into a dict with the ``toml`` library
    """
    return zip(
        [_return_toml(config) for config in all_generated_learncurve_configs],
        all_generated_learncurve_configs,
    )


@pytest.fixture
def all_generated_eval_configs_toml_path_pairs(all_generated_eval_configs):
    """zip of tuple pairs: (dict, pathlib.Path)
    where ``Path`` is path to .toml config file and ``dict`` is
    the .toml config from that path
    loaded into a dict with the ``toml`` library
    """
    return zip(
        [_return_toml(config) for config in all_generated_eval_configs],
        all_generated_eval_configs,
    )


@pytest.fixture
def all_generated_predict_configs_toml_path_pairs(all_generated_predict_configs):
    """zip of tuple pairs: (dict, pathlib.Path)
    where ``Path`` is path to .toml config file and ``dict`` is
    the .toml config from that path
    loaded into a dict with the ``toml`` library
    """
    return zip(
        [_return_toml(config) for config in all_generated_predict_configs],
        all_generated_predict_configs,
    )
