
from __future__ import annotations

from attrs import define, field, validators


def is_valid_devices(instance, attribute, value):
    """check if ``devices`` is valid"""
    if isinstance(value, int):
        return
    elif isinstance(value, list):
        if not all(
            [isinstance(val, int) for val in value]
        ):
            types_in_list = set([type(val) for val in value])
            raise ValueError(
                "TrainerConfig attribute `devices` must be either an int or list of ints "
                f"but received list with the following types: {types_in_list}"
            )


@define
class TrainerConfig:
    """Class that represents ``trainer`` sub-table
    in a toml configuration file.

    Used to configure :class:`lightning.Trainer`.

    Attributes
    ----------

    """
    accelerator: str
    devices: int | list[int] = field(validator=validators.optional(is_valid_devices), default=None)