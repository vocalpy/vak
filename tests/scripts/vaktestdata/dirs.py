"""Helper functions for setting up directories."""
from . import constants


def make_subdirs_in_generated(config_paths):
    """make sub-directories inside ./tests/data_for_tests/generated

    do this after copying configs,
    before using those configs to generate results.
    We use configs to decide which dirs we need to make

    makes three directories in data_for_tests/generated:
    configs, prep, and results.
    prep has one sub-directory for every data "type".
    results does also, but in addition will have sub-directories
    within those for models.
    """
    for top_level_dir in constants.TOP_LEVEL_DIRS:
        for command_dir in constants.COMMAND_DIRS:
            for data_dir in constants.DATA_DIRS:
                if not any(
                        [f'{command_dir}_{data_dir}' in str(config_path) for config_path in config_paths]
                ):
                    continue  # no need to make this dir

                for model in constants.MODELS_RESULTS:
                    subdir_to_make = (
                        constants.GENERATED_TEST_DATA / top_level_dir / command_dir / data_dir / model
                    )
                    subdir_to_make.mkdir(parents=True)
