"""Functions to parse toml config files."""

from __future__ import annotations

import pathlib

import tomlkit
import tomlkit.exceptions


def _tomlkit_to_popo(d):
    """Convert tomlkit to "popo" (Plain-Old Python Objects)

    From https://github.com/python-poetry/tomlkit/issues/43#issuecomment-660415820

    We need this so we don't get a ``tomlkit.items._ConvertError`` when
    the `from_config_dict` classmethods try to add a class to a ``config_dict``,
    e.g. when :meth:`EvalConfig.from_config_dict` converts the ``spect_params``
    key-value pairs to a :class:`vak.config.SpectParamsConfig` instance
    and then assigns it to the ``spect_params`` key.
    We would get this error if we just return the result of :func:`tomlkit.load`,
    which is a `tomlkit.TOMLDocument` that tries to ensure that everything is valid toml.
    """
    try:
        result = getattr(d, "value")
    except AttributeError:
        result = d

    if isinstance(result, list):
        result = [_tomlkit_to_popo(x) for x in result]
    elif isinstance(result, dict):
        result = {
            _tomlkit_to_popo(key): _tomlkit_to_popo(val)
            for key, val in result.items()
        }
    elif isinstance(result, tomlkit.items.Integer):
        result = int(result)
    elif isinstance(result, tomlkit.items.Float):
        result = float(result)
    elif isinstance(result, tomlkit.items.String):
        result = str(result)
    elif isinstance(result, tomlkit.items.Bool):
        result = bool(result)

    return result


def _load_toml_from_path(toml_path: str | pathlib.Path) -> dict:
    """Load a toml file from a path, and return as a :class:`dict`.

    Notes
    -----
    Helper function to load toml config file,
    factored out to use in other modules when needed.
    Checks if ``toml_path`` exists before opening,
    and tries to give a clear message if an error occurs when loading.

    Note also this function checks that the loaded :class:`dict`
    has a single top-level key ``'vak'``,
    and that it returns the :class:`dict` one level down
    that is accessed with that key.
    This avoids the need to write ``['vak']`` everywhere in
    calling functions.
    However it also means you need to add back that key
    if you are *writing* a toml file.
    """
    toml_path = pathlib.Path(toml_path)
    if not toml_path.is_file():
        raise FileNotFoundError(f".toml config file not found: {toml_path}")

    try:
        with toml_path.open("r") as fp:
            config_dict: dict = tomlkit.load(fp)
    except tomlkit.exceptions.TOMLKitError as e:
        raise Exception(
            f"Error when parsing .toml config file: {toml_path}"
        ) from e

    if "vak" not in config_dict:
        raise ValueError(
            "Toml file does not contain a top-level table named `vak`. "
            "Please see example configuration files here:\n"
            "https://github.com/vocalpy/vak/tree/main/doc/toml"
        )

    # Next line, convert TOMLDocument returned by tomlkit.load to a dict.
    # We need this so we don't get a ``tomlkit.items._ConvertError`` when
    # the `from_config_dict` classmethods try to add a class to a ``config_dict``,
    # e.g. when :meth:`EvalConfig.from_config_dict` converts the ``spect_params``
    # key-value pairs to a :class:`vak.config.SpectParamsConfig` instance
    # and then assigns it to the ``spect_params`` key.
    # We would get this error if we just return the result of :func:`tomlkit.load`,
    # which is a `tomlkit.TOMLDocument` that tries to ensure that everything is valid toml.
    return _tomlkit_to_popo(config_dict)["vak"]
