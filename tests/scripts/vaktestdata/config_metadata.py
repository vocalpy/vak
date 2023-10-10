import attrs


@attrs.define
class ConfigMetadata:
    """Dataclass that represents metadata
    about a configuration file

    Attributes
    ----------
    filename : str
        The name of the configuration file.
    model : str
        The name of the model in :mod:`vak`
        that the configuration file is used with.
    model_family : str
        The name of the model family
        for the model in the configuration file.
    config_type : str
        The type of config, one of
        {'train', 'eval', 'predict', 'learncurve'}.
    audio_format : str
        The format of the audio files.
    spect_format : str
        The format of the spectrogram files.
    spect_output_dir : str, optional
        The directory where spectrograms should be saved
        when generated for this configuration file.
        If not specified, then no spectrograms are generated.
        This attribute is used to avoid repeatedly
        generating the same set of spectrograms for multiple
        configs.
    data_dir : str, optional
        The directory that should be used as the `data_dir`
        option for this config.
        The option will be changed to this value in the generated
        config file.
        This attribute is used to avoid repeatedly
        generating the same set of spectrograms for multiple
        configs.
    use_dataset_from_config : str, optional
        The filename of another configuration file.
        The ``dataset_path`` option of that configuration file
        will be used for this configuration file.
        This option is used to avoid repeatedly
        generating the same dataset for multiple configs.
    use_results_from_config : str, optional
        The filename of another configuration file.
        The most recent results from ``results_path`` option
        of that configuration file
        will be used for this configuration file.
    """
    filename: str = attrs.field(converter=str)
    model: str = attrs.field(converter=str)
    model_family: str = attrs.field(converter=str)
    config_type: str = attrs.field(converter=str)
    audio_format: str = attrs.field(converter=attrs.converters.optional(str), default=None)
    spect_format: str = attrs.field(converter=attrs.converters.optional(str), default=None)
    annot_format: str = attrs.field(converter=attrs.converters.optional(str), default=None)
    spect_output_dir: str = attrs.field(converter=attrs.converters.optional(str), default=None)
    data_dir: str = attrs.field(converter=attrs.converters.optional(str), default=None)
    use_dataset_from_config: str = attrs.field(converter=attrs.converters.optional(str), default=None)
    use_result_from_config: str = attrs.field(converter=attrs.converters.optional(str), default=None)
