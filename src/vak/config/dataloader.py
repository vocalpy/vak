import attr
from attr.validators import instance_of


@attr.s
class DataLoaderConfig:
    """represents options for DataLoaders specified in config.toml file

    Attributes
    ----------
    window_size : int
        size of windows taken from spectrograms, in number of time bins,
        shonw to neural networks
    """
    window_size = attr.ib(converter=int,
                          validator=instance_of(int),
                          default=88)


def parse_dataloader_config(config_toml, toml_path):
    """parse [DATALOADER] section of config.toml file

    Parameters
    ----------
    config_toml : dict
        containing configuration file in TOML format, already loaded by parse function
    toml_path : Path
        path to a configuration file in TOML format.
        Note that this is not actually used but the function has this parameter
        for consistency with other functions.
        **Removing it will cause ``vak.config.parse.from_toml`` to crash, since it
        tries to loop through all the section parser functions,
        nd pass them both arguments.**

    Returns
    -------
    dataloader_config : vak.config.dataloader.DataloaderConfig
        instance of Dataloader class
    """
    # return defaults if config doesn't have DATALOADER section
    dataloader_section = {}
    if 'DATALOADER' in config_toml:
        dataloader_section.update(
            config_toml['DATALOADER'].items()
        )
    return DataLoaderConfig(**dataloader_section)
