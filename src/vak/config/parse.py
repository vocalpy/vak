import os
from configparser import ConfigParser
from configparser import NoSectionError, MissingSectionHeaderError, ParsingError,\
    DuplicateOptionError, DuplicateSectionError
from collections import namedtuple

import attr
from attr.validators import instance_of, optional

from .data import parse_data_config, DataConfig
from .spectrogram import parse_spect_config, SpectConfig
from .train import parse_train_config, TrainConfig
from .output import parse_output_config, OutputConfig
from .predict import parse_predict_config, PredictConfig
from ..network import _load


@attr.s
class Config:
    """class to represent config.ini file

    Attributes
    ----------
    data : vak.config.data.DataConfig
        represents [DATA] section of config.ini file
    spect_params : vak.config.spectrogram.SpectConfig
        represents [SPECTROGRAM] section of config.ini file
    train : vak.config.train.TrainConfig
        represents [TRAIN] section of config.ini file
    predict : vak.config.predict.PredictConfig
        represenets [PREDICT] section of config.ini file.
    output : vak.config.output.OutputConfig
        represents [OUTPUT] section of config.ini file
    networks : dict
        represents neural network configuration sections of config.ini file.
        These will vary depending on which network user specifies.
    """
    data = attr.ib(validator=optional(instance_of(DataConfig)))
    spect_params = attr.ib(validator=optional(instance_of(SpectConfig)))
    train = attr.ib(validator=optional(instance_of(TrainConfig)))
    predict = attr.ib(validator=optional(instance_of(PredictConfig)))
    output = attr.ib(validator=optional(instance_of(OutputConfig)))
    networks = attr.ib()


def parse_config(config_file):
    """parse a config.ini file

    Parameters
    ----------
    config_file : str
        path to config.ini file

    Returns
    -------
    config_tuple : ConfigTuple
        instance of a ConfigTuple whose fields correspond to
        sections in the config.ini file.
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

    if config_obj.has_section('DATA'):
        data = parse_data_config(config_obj, config_file)
    else:
        data = None

    ### if **not** using spectrograms from .mat files ###
    if config_obj.has_section('SPECTROGRAM'):
        spect_params = parse_spect_config(config_obj)
    else:
        spect_params = None

    if config_obj.has_section('TRAIN'):
        train = parse_train_config(config_obj, config_file)
        networks = train.networks
    else:
        train = None

    if config_obj.has_section('PREDICT'):
        predict = parse_predict_config(config_obj)
        networks = predict.networks
    else:
        predict = None

    # load entry points within function, not at module level,
    # to avoid circular dependencies
    # (user would be unable to import networks in other packages
    # that subclass vak.network.AbstractVakNetwork
    # since the module in the other package would need to `import vak`)
    NETWORKS = _load()
    sections = config_obj.sections()
    # make tuple that will have network names as fields
    # and a config tuple for each network as the value assigned to the corresponding field
    NetworkTuple = namedtuple('NetworkTuple', [network for network in networks])
    networks_dict = {}
    for network in networks:
        if network.lower() not in [section.lower() for section in sections]:
            raise NoSectionError('No section found specifying parameters for network {}'
                                 .format(network))
        network_option_names = set(config_obj[network].keys())
        config_field_names = set(NETWORKS[network].Config._fields)
        # if some options in this network's section are not found in the Config tuple
        # that is a class attribute for that network, raise an error because we don't
        # know what to do with that option
        if not network_option_names.issubset(config_field_names):
            unknown_options = network_option_names - config_field_names
            raise ValueError('The following option(s) in section for network {} are '
                             'not found in the Config for that network: {}.\n'
                             'Valid options are: {}'
                             .format(network, unknown_options, config_field_names))

        options = {}
        # do type conversion using the networks' Config typed namedtuple
        # for the rest of the options
        for option, value in config_obj[network].items():
            option_type = NETWORKS[network].Config._field_types[option]
            try:
                options[option] = option_type(value)
            except ValueError:
                raise ValueError('Could not cast value {} for option {} in {} section '
                                 ' to specified type {}.'
                                 .format(value, option, network, option_type))
        networks_dict[network] = NETWORKS[network].Config(**options)
    networks = NetworkTuple(**networks_dict)

    if config_obj.has_section('OUTPUT'):
        output = parse_output_config(config_obj)
    else:
        output = None

    return Config(data,
                  spect_params,
                  train,
                  output,
                  networks,
                  predict)
