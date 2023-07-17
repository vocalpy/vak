# Do this here to suppress warnings before we import vak
import logging
import warnings

from numba.core.errors import NumbaDeprecationWarning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)

import vak

from . import constants


logger = logging.getLogger(__name__)


def run_prep():
    """Run ``vak prep`` to prepare datasets used with tests.

    This function runs ``prep`` for **only** the configuration files
    in configs.json that have ``null`` for the field in their metadata
    ``'use_dataset_from_config'``. The ``null`` indicates that these
    configs do **not** re-use a dataset prepared from another config.
    """
    configs_to_prep = [
        config_metadata
        for config_metadata in constants.CONFIG_METADATA
        if config_metadata.use_dataset_from_config is None
    ]

    for config_metadata in configs_to_prep:
        config_path = constants.GENERATED_TEST_CONFIGS_ROOT / config_metadata.filename
        if not config_path.exists():
            raise FileNotFoundError(f"{config_path} not found")
        logger.info(
            f"\nRunning vak prep to generate data for tests, using config:\n{config_path.name}"
        )
        vak.cli.prep.prep(toml_path=config_path)
