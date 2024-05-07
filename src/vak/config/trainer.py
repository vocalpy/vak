from __future__ import annotations

from attrs import asdict, define, field, validators

from .. import common


def is_valid_accelerator(instance, attribute, value):
    """Check if ``accelerator`` is valid"""
    if value == "auto":
        raise ValueError(
            "Using the 'auto' value for the `lightning.pytorch.Trainer` parameter `accelerator` currently "
            "breaks functionality for the command-line interface of `vak`. "
            "Please see this issue: https://github.com/vocalpy/vak/issues/691"
            "If you need to use the 'auto' mode of `lightning.pytorch.Trainer`, please use `vak` directly in a script."
        )
    elif value in ("cpu", "gpu", "tpu", "ipu"):
        return
    else:
        raise ValueError(
            f"Invalid value for 'accelerator' key in 'trainer' table of configuration file: {value}. "
            'Value must be one of: {"cpu", "gpu", "tpu", "ipu"}'
        )


def is_valid_devices(instance, attribute, value):
    """Check if ``devices`` is valid"""
    if not (
        (isinstance(value, int))
        or (
            isinstance(value, list)
            and all([isinstance(el, int) for el in value])
        )
    ):
        raise ValueError(
            "Invalid value for 'devices' key in 'trainer' table of configuration file: {value}"
        )


@define
class TrainerConfig:
    """Class that represents ``trainer`` sub-table
    in a toml configuration file.

    Used to configure :class:`lightning.pytorch.Trainer`.

    Attributes
    ----------
    accelerator : str
        Value for the `accelerator` argument to :class:`lightning.pytorch.Trainer`.
        Default is the return value of :func:`vak.common.accelerator.get_default`.
    devices: int, list of int
        Number of devices (int) or exact device(s) (list of int) to use.

    Notes
    -----
    Using the 'auto' value for the `lightning.pytorch.Trainer` parameter `accelerator` currently
    breaks functionality for the command-line interface of `vak`.
    Please see this issue: https://github.com/vocalpy/vak/issues/691
    If you need to use the 'auto' mode of `lightning.pytorch.Trainer`, please use `vak` directly in a script.

    Likewise, setting a value for the `lightning.pytorch.Trainer` parameter `devices` that is not either 1
    (meaning \"use a single GPU\") or a list with a single number (meaning \"use this exact GPU\") currently
    breaks functionality for the command-line interface of `vak`.
    Please see this issue: https://github.com/vocalpy/vak/issues/691
    If you need to use multiple GPUs, please use `vak` directly in a script.
    """

    accelerator: str = field(
        validator=is_valid_accelerator,
        default=common.accelerator.get_default(),
    )
    devices: int | list[int] = field(
        validator=validators.optional(is_valid_devices),
        # for devices, we need to look at accelerator in post-init to determine default
        default=None,
    )

    def __attrs_post_init__(self):
        # set default self.devices *before* we validate,
        # so that we don't throw error because of the default None
        # that we need to change here depending on the value of self.accelerator
        if self.devices is None:
            if self.accelerator == "cpu":
                # ~"use all available"
                self.devices = 1
            elif self.accelerator in ("gpu", "tpu", "ipu"):
                # we can only use a single device, assume there's only one
                self.devices = [0]

        if self.accelerator in ("gpu", "tpu", "ipu"):
            if not (
                (isinstance(self.devices, int) and self.devices == 1)
                or (
                    isinstance(self.devices, list)
                    and len(self.devices) == 1
                    and all([isinstance(el, int) for el in self.devices])
                )
            ):
                raise ValueError(
                    "Setting a value for the `lightning.pytorch.Trainer` parameter `devices` that is not either 1 "
                    '(meaning "use a single GPU") or a list with a single number '
                    '(meaning "use this exact GPU") currently '
                    "breaks functionality for the command-line interface of `vak`. "
                    "Please see this issue: https://github.com/vocalpy/vak/issues/691"
                    "If you need to use multiple GPUs, please use `vak` directly in a script."
                )
        elif self.accelerator == "cpu":
            if isinstance(self.devices, list):
                raise ValueError(
                    f"Value for `devices` cannot be a list when `accelerator` is `cpu`. Value was: {self.devices}\n"
                    "When `accelerator` is `cpu`, please set `devices` to 1 or -1 (which are equivalent)."
                )
            if self.devices < 1:
                raise ValueError(
                    "When value for 'accelerator' is 'cpu', value for `devices` "
                    f"should be an int > 0, but was: {self.devices}"
                )

    def asdict(self):
        """Convert this :class:`TrainerConfig` instance
        to a :class:`dict` that can be passed
        into functions that take a ``trainer_config`` argument,
        like :func:`vak.train` and :func:`vak.predict`.
        """
        return asdict(self)
