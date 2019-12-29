from configparser import ConfigParser
from configparser import MissingSectionHeaderError, ParsingError,\
    DuplicateOptionError, DuplicateSectionError
from pathlib import Path

import attr
from attr.validators import instance_of, optional

from .dataloader import parse_dataloader_config, DataLoaderConfig
from .learncurve import parse_learncurve_config, LearncurveConfig
from .predict import parse_predict_config, PredictConfig
from .prep import parse_prep_config, PrepConfig
from .spectrogram import parse_spect_config, SpectConfig
from .train import parse_train_config, TrainConfig

from .validators import are_sections_valid, are_options_valid


@attr.s
class Config:
    """class to represent config.ini file

    Attributes
    ----------
    prep : vak.config.prep.PrepConfig
        represents [PREP] section of config.ini file
    spect : vak.config.spectrogram.SpectConfig
        represents [SPECTROGRAM] section of config.ini file
    dataloader : vak.config.dataloader.DataLoaderConfig
        represents [DATALOADER] section of config.ini file
    train : vak.config.train.TrainConfig
        represents [TRAIN] section of config.ini file
    predict : vak.config.predict.PredictConfig
        represents [PREDICT] section of config.ini file.
    learncurve : vak.config.learncurve.LearncurveConfig
        represents [LEARNCURVE] section of config.ini file
    """
    spect = attr.ib(validator=instance_of(SpectConfig), default=SpectConfig())
    dataloader = attr.ib(validator=instance_of(DataLoaderConfig), default=DataLoaderConfig())

    prep = attr.ib(validator=optional(instance_of(PrepConfig)), default=None)
    train = attr.ib(validator=optional(instance_of(TrainConfig)), default=None)
    predict = attr.ib(validator=optional(instance_of(PredictConfig)), default=None)
    learncurve = attr.ib(validator=optional(instance_of(LearncurveConfig)), default=None)


SECTION_PARSERS = {
    'SPECTROGRAM': parse_spect_config,
    'DATAlOADER': parse_dataloader_config,
    'PREP': parse_prep_config,
    'TRAIN': parse_train_config,
    'LEARNCURVE': parse_learncurve_config,
    'PREDICT': parse_predict_config,

}


def from_path(config_path):
    """parse a config.ini file

    Parameters
    ----------
    config_path : str, Path
        path to config.ini file

    Returns
    -------
    config : vak.config.parse.Config
        instance of Config class, whose attributes correspond to
        sections in a config.ini file.
    """
    # check config_path is a file,
    # because if it doesn't ConfigParser will just return an "empty" instance w/out sections or options
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f'path not recognized as a file: {config_path}')

    try:
        config_obj = ConfigParser()
        config_obj.read(config_path)
    except (MissingSectionHeaderError, ParsingError, DuplicateOptionError, DuplicateSectionError):
        # try to add some context for users that do not spend their lives thinking about ConfigParser objects
        print(f"Error when opening the following config_path: {config_path}")
        raise
    except:
        # say something different if we can't add very good context
        print(f"Unexpected error when opening the following config_path: {config_path}")
        raise

    are_sections_valid(config_obj, config_path)

    if config_obj.has_section('TRAIN') and config_obj.has_section('LEARNCURVE'):
        raise ValueError(
            'a single config.ini file cannot contain both TRAIN and LEARNCURVE sections, '
            'because it is unclear which of those two sections to add paths to when running '
            'the "prep" command to prepare datasets'
        )

    if config_obj.has_section('TRAIN'):
        if config_obj.has_option('PREP', 'test_set_duration'):
            raise ValueError(
                "cannot define 'test_set_duration' option for PREP section when using with vak 'train' command, "
                "'test_set_duration' is not a valid option for the TRAIN section. "
                "Were you trying to use the 'learncurve' command instead?"
            )

    config_dict = {}
    for section_name, section_parser in SECTION_PARSERS.items():
        if config_obj.has_section(section_name):
            are_options_valid(config_obj, section_name, config_path)
            config_dict[section_name.lower()] = section_parser(config_obj, config_path)

    return Config(**config_dict)
