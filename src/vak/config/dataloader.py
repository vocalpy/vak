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

    window_size = attr.ib(converter=int, validator=instance_of(int), default=88)
