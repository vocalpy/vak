import attrs


@attrs.define
class ConfigMetadata:
    filename: str = attrs.field()
    model: str = attrs.field()
    config_type: str = attrs.field()
    audio_format: str = attrs.field()
    spect_format: str = attrs.field()
    annot_format: str = attrs.field()
    use_dataset_from_config = attrs.field(default=None)
    use_result_from_config = attrs.field(default=None)
