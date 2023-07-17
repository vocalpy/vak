import attrs


@attrs.define
class ConfigMetadata:
    config_path: str
    audio_format: str
    spect_format: str