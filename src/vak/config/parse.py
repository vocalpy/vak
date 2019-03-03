import os
from configparser import ConfigParser, NoSectionError
from collections import namedtuple

from .data import parse_data_config
from .spectrogram import parse_spect_config
from .train import parse_train_config
from .output import parse_output_config
from .predict import parse_predict_config
from ..network import _load


ConfigTuple = namedtuple('ConfigTuple', ['data',
                                         'spect_params',
                                         'train',
                                         'output',
                                         'networks',
                                         'predict'])


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
    if not config_file.endswith('.ini'):
        raise ValueError('{} is not a valid config file, '
                         'must have .ini extension'.format(config_file))
    if not os.path.isfile(config_file):
        raise FileNotFoundError('config file {} is not found'
                                .format(config_file))
    config_obj = ConfigParser()
    config_obj.read(config_file)

    if config_obj.has_section('TRAIN') and config_obj.has_section('PREDICT'):
        raise ValueError('Please do not declare both TRAIN and PREDICT sections '
                         ' in one config.ini file: unclear which to use')

    if config_obj.has_section('DATA'):
        data = parse_data_config(config_obj, config_file)
    else:
        data = None

    ### if **not** using spectrograms from .mat files ###
    if data.mat_spect_files_path is None:
        # then user needs to specify spectrogram parameters
        if not config_obj.has_section('SPECTROGRAM'):
            raise ValueError('No annotation_path specified in config_file that '
                             'would point to annotated spectrograms, but no '
                             'parameters provided to generate spectrograms '
                             'either.')
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

        if data.freq_bins:
            # start options dict with freq_bins that we got out of data above
            # (this argument is required for all networks)
            options = {'freq_bins': data.freq_bins}
        else:
            # except if freq_bins doesn't exist yet, e.g. because we haven't run make_data
            options = {}

        # and then do type conversion using the networks Config typed namedtuple
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

    return ConfigTuple(data,
                       spect_params,
                       train,
                       output,
                       networks,
                       predict)
