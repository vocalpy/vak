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


def parse_dataloader_config(config, config_path):
    # return defaults if config doesn't have SPECTROGRAM section
    dataloader_section = {}
    if 'DATALOADER' in config:
        dataloader_section.update(
            config['DATALOADER'].items()
        )
    return DataLoaderConfig(**dataloader_section)
