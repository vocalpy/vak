# Do this here to suppress warnings before we import vak
import logging
import warnings

from numba.core.errors import NumbaDeprecationWarning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)

import vak

from . import constants


logger = logging.getLogger(__name__)


def get_or_make_source_files():
    """Run :func:`vak.prep.frame_classification.get_or_make_source_files`
    to generate files used with tests.

    This runs *only* for configs where the metadata has the model family
    ``"frame_classification"`` and the metadata has a value for
    ``spect_output_dir``.
    """
    configs_to_prep = [
        config_metadata
        for config_metadata in constants.CONFIG_METADATA
        if config_metadata.use_dataset_from_config is None
    ]

    for config_metadata in configs_to_prep:
        if config_metadata.model_family != "frame_classification":
            continue

        if config_metadata.spect_output_dir is None:
            continue

        spect_output_dir = constants.GENERATED_SPECT_OUTPUT_DIR / config_metadata.spect_output_dir
        spect_output_dir.mkdir()

        config_path = constants.GENERATED_TEST_CONFIGS_ROOT / config_metadata.filename
        if not config_path.exists():
            raise FileNotFoundError(f"{config_path} not found")
        logger.info(
            f"\nRunning :func:`vak.prep.frame_classification.get_or_make_source_files` to generate data for tests, using config:\n{config_path.name}"
        )
        cfg = vak.config.parse.from_toml_path(config_path)

        vak.prep.frame_classification.get_or_make_source_files(
            data_dir=cfg.prep.data_dir,
            input_type=cfg.prep.input_type,
            audio_format=cfg.prep.audio_format,
            spect_format=cfg.prep.spect_format,
            spect_params=cfg.spect_params,
            spect_output_dir=spect_output_dir,
            annot_format=cfg.prep.annot_format,
            annot_file=cfg.prep.annot_file,
            labelset=cfg.prep.labelset,
            audio_dask_bag_kwargs=cfg.prep.audio_dask_bag_kwargs,
        )
        vak.cli.prep.prep(toml_path=config_path)
