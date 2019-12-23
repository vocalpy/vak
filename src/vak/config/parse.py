import os
from configparser import ConfigParser
from configparser import MissingSectionHeaderError, ParsingError,\
    DuplicateOptionError, DuplicateSectionError

import attr
from attr.validators import instance_of, optional

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
    spect_params : vak.config.spectrogram.SpectConfig
        represents [SPECTROGRAM] section of config.ini file
    learncurve : vak.config.learncurve.LearncurveConfig
        represents [LEARNCURVE] section of config.ini file
    train : vak.config.train.TrainConfig
        represents [TRAIN] section of config.ini file
    predict : vak.config.predict.PredictConfig
        represents [PREDICT] section of config.ini file.
    networks : dict
        contains neural network configuration sections of config.ini file.
        These will vary depending on which networks the user specifies.
    """
    prep = attr.ib(validator=optional(instance_of(PrepConfig)), default=None)
    spect_params = attr.ib(validator=optional(instance_of(SpectConfig)), default=None)
    learncurve = attr.ib(validator=optional(instance_of(LearncurveConfig)), default=None)
    train = attr.ib(validator=optional(instance_of(TrainConfig)), default=None)
    predict = attr.ib(validator=optional(instance_of(PredictConfig)), default=None)
    networks = attr.ib(validator=optional(instance_of(dict)), default=None)


def parse_config(config_file):
    """parse a config.ini file

    Parameters
    ----------
    config_file : str
        path to config.ini file

    Returns
    -------
    config : vak.config.parse.Config
        instance of Config class, whose attributes correspond to
        sections in a config.ini file.
    """
    # check config_file exists,
    # because if it doesn't ConfigParser will just return an "empty" instance w/out sections or options
    if not os.path.isfile(config_file):
        raise FileNotFoundError('config file {} is not found'
                                .format(config_file))

    try:
        config_obj = ConfigParser()
        config_obj.read(config_file)
    except (MissingSectionHeaderError, ParsingError, DuplicateOptionError, DuplicateSectionError):
        # try to add some context for users that do not spend their lives thinking about ConfigParser objects
        print(f"Error when opening the following config_file: {config_file}")
        raise
    except:
        # say something different if we can't add very good context
        print(f"Unexpected error when opening the following config_file: {config_file}")
        raise

    are_sections_valid(config_obj, config_file)

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
    if config_obj.has_section('PREP'):
        config_dict['prep'] = parse_prep_config(config_obj, config_file)

    ### if **not** using spectrograms from .mat files ###
    if config_obj.has_section('SPECTROGRAM'):
        are_options_valid(config_obj, 'SPECTROGRAM', config_file)
        config_dict['spect_params'] = parse_spect_config(config_obj)

    networks = []
    if config_obj.has_section('LEARNCURVE'):
        are_options_valid(config_obj, 'LEARNCURVE', config_file)
        config_dict['learncurve'] = parse_learncurve_config(config_obj, config_file)
        networks += config_dict['learncurve'].networks

    if config_obj.has_section('TRAIN'):
        are_options_valid(config_obj, 'TRAIN', config_file)
        config_dict['train'] = parse_train_config(config_obj, config_file)
        networks += config_dict['train'].networks

    if config_obj.has_section('PREDICT'):
        are_options_valid(config_obj, 'PREDICT', config_file)
        config_dict['predict'] = parse_predict_config(config_obj)
        networks += config_dict['predict'].networks

    if networks:
        config_dict['networks'] = _get_nets_config(config_obj, networks)

    return Config(**config_dict)
