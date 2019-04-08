import os
from configparser import ConfigParser
from configparser import NoSectionError, MissingSectionHeaderError, ParsingError,\
    DuplicateOptionError, DuplicateSectionError

import attr
from attr.validators import instance_of, optional

from .data import parse_data_config, DataConfig
from .spectrogram import parse_spect_config, SpectConfig
from .train import parse_train_config, TrainConfig
from .output import parse_output_config, OutputConfig
from .predict import parse_predict_config, PredictConfig
from .. import network


def _get_nets_config(config_obj, networks):
    """helper function to get configuration for only networks that user specified
    in a specific section of config.ini file, e.g. in the TRAIN section

    Parameters
    ----------
    config_obj : configparser.ConfigParser
        instance of ConfigParser with config.ini file already read into it
        that has sections representing configurations for networks
    networks : list
        of str, i.e. names of networks specified by a section
        (such as TRAIN or PREDICT) that should each have corresponding sections
        specifying their configuration: hyperparameters such as learning
        rate, number of time steps, etc.

    Returns
    -------
    networks_dict : dict
        where each key is the name of a network and the corresponding value is
        another dict containing key-value pairs of configuration options and the
        value specified for that option
    """
    # load entry points within function, not at module level,
    # to avoid circular dependencies
    # (user would be unable to import networks in other packages
    # that subclass vak.network.AbstractVakNetwork
    # since the module in the other package would need to `import vak`)
    NETWORKS = network._load()
    sections = config_obj.sections()
    networks_dict = {}
    for network_name in networks:
        if network_name not in sections:
            raise NoSectionError('No section found specifying parameters for network {}'
                                 .format(network_name))
        network_option_names = set(config_obj[network_name].keys())
        config_field_names = set(NETWORKS[network_name].Config._fields)
        # if some options in this network's section are not found in the Config tuple
        # that is a class attribute for that network, raise an error because we don't
        # know what to do with that option
        if not network_option_names.issubset(config_field_names):
            unknown_options = network_option_names - config_field_names
            raise ValueError('The following option(s) in section for network {} are '
                             'not found in the Config for that network: {}.\n'
                             'Valid options are: {}'
                             .format(network_name, unknown_options, config_field_names))

        options = {}
        # do type conversion using the networks' Config typed namedtuple
        # for the rest of the options
        for option, value in config_obj[network_name].items():
            option_type = NETWORKS[network_name].Config._field_types[option]
            try:
                options[option] = option_type(value)
            except ValueError:
                raise ValueError('Could not cast value {} for option {} in {} section '
                                 ' to specified type {}.'
                                 .format(value, option, network_name, option_type))
        networks_dict[network_name] = NETWORKS[network_name].Config(**options)
        return networks_dict


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
        represents [PREDICT] section of config.ini file.
    output : vak.config.output.OutputConfig
        represents [OUTPUT] section of config.ini file
    networks : dict
        contains neural network configuration sections of config.ini file.
        These will vary depending on which networks the user specifies.
    """
    data = attr.ib(validator=optional(instance_of(DataConfig)), default=None)
    spect_params = attr.ib(validator=optional(instance_of(SpectConfig)), default=None)
    train = attr.ib(validator=optional(instance_of(TrainConfig)), default=None)
    predict = attr.ib(validator=optional(instance_of(PredictConfig)), default=None)
    output = attr.ib(validator=optional(instance_of(OutputConfig)), default=None)
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

    config_dict = {}
    if config_obj.has_section('DATA'):
        config_dict['data'] = parse_data_config(config_obj, config_file)

    ### if **not** using spectrograms from .mat files ###
    if config_obj.has_section('SPECTROGRAM'):
        config_dict['spect_params'] = parse_spect_config(config_obj)

    networks = []
    if config_obj.has_section('TRAIN'):
        config_dict['train'] = parse_train_config(config_obj, config_file)
        networks += config_dict['train'].networks

    if config_obj.has_section('PREDICT'):
        config_dict['predict'] = parse_predict_config(config_obj)
        networks += config_dict['predict'].networks

    if networks:
        config_dict['networks'] = _get_nets_config(config_obj, networks)

    if config_obj.has_section('OUTPUT'):
        config_dict['output'] = parse_output_config(config_obj)

    return Config(**config_dict)
