from pathlib import Path

import attr
from attr.validators import instance_of, optional
import toml
from toml.decoder import TomlDecodeError

from .dataloader import parse_dataloader_config, DataLoaderConfig
from .eval import parse_eval_config, EvalConfig
from .learncurve import parse_learncurve_config, LearncurveConfig
from .predict import parse_predict_config, PredictConfig
from .prep import parse_prep_config, PrepConfig
from .spect_params import parse_spect_params_config, SpectParamsConfig
from .train import parse_train_config, TrainConfig

from .validators import are_sections_valid, are_options_valid


@attr.s
class Config:
    """class to represent config.toml file

    Attributes
    ----------
    prep : vak.config.prep.PrepConfig
        represents [PREP] section of config.toml file
    spect : vak.config.spectrogram.SpectConfig
        represents [SPECTROGRAM] section of config.toml file
    dataloader : vak.config.dataloader.DataLoaderConfig
        represents [DATALOADER] section of config.toml file
    train : vak.config.train.TrainConfig
        represents [TRAIN] section of config.toml file
    eval : vak.config.eval.EvalConfig
        represents [EVAL] section of config.toml file
    predict : vak.config.predict.PredictConfig
        represents [PREDICT] section of config.toml file.
    learncurve : vak.config.learncurve.LearncurveConfig
        represents [LEARNCURVE] section of config.toml file
    """
    spect_params = attr.ib(validator=instance_of(SpectParamsConfig), default=SpectParamsConfig())
    dataloader = attr.ib(validator=instance_of(DataLoaderConfig), default=DataLoaderConfig())

    prep = attr.ib(validator=optional(instance_of(PrepConfig)), default=None)
    train = attr.ib(validator=optional(instance_of(TrainConfig)), default=None)
    eval = attr.ib(validator=optional(instance_of(EvalConfig)), default=None)
    predict = attr.ib(validator=optional(instance_of(PredictConfig)), default=None)
    learncurve = attr.ib(validator=optional(instance_of(LearncurveConfig)), default=None)


SECTION_PARSERS = {
    'SPECT_PARAMS': parse_spect_params_config,
    'DATALOADER': parse_dataloader_config,
    'PREP': parse_prep_config,
    'EVAL': parse_eval_config,
    'TRAIN': parse_train_config,
    'LEARNCURVE': parse_learncurve_config,
    'PREDICT': parse_predict_config,
}


def from_toml(toml_path):
    """parse a TOML configuration file

    Parameters
    ----------
    toml_path : str, Path
        path to a configuration file in TOML format

    Returns
    -------
    config : vak.config.parse.Config
        instance of Config class, whose attributes correspond to
        sections in a config.toml file.
    """
    # check config_path is a file,
    # because if it doesn't ConfigParser will just return an "empty" instance w/out sections or options
    toml_path = Path(toml_path)
    if not toml_path.is_file():
        raise FileNotFoundError(f'path not recognized as a file: {toml_path}')

    try:
        with toml_path.open('r') as fp:
            config_toml = toml.load(fp)
    except TomlDecodeError as e:
        raise Exception(f'Error when parsing .toml config file: {toml_path}') from e

    are_sections_valid(config_toml, toml_path)

    if 'TRAIN' in config_toml:
        if 'LEARNCURVE' in config_toml:
            raise ValueError(
                'a single config.toml file cannot contain both TRAIN and LEARNCURVE sections, '
                'because it is unclear which of those two sections to add paths to when running '
                'the "prep" command to prepare datasets'
            )

        if 'PREP' in config_toml:
            if 'test_set_duration' in config_toml['PREP']:
                raise ValueError(
                    "cannot define 'test_set_duration' option for PREP section when using with vak 'train' command, "
                    "'test_set_duration' is not a valid option for the TRAIN section. "
                    "Were you trying to use the 'learncurve' command instead?"
                )

    config_dict = {}
    for section_name in SECTION_PARSERS.keys():
        if section_name in config_toml:
            are_options_valid(config_toml, section_name, toml_path)
            section_parser = SECTION_PARSERS[section_name]
            config_dict[section_name.lower()] = section_parser(config_toml, toml_path)

    return Config(**config_dict)
