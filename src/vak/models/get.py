"""Function that gets an instance of a model,
given its name and a configuration as a dict."""

from __future__ import annotations

import inspect
from typing import Callable

import lightning

from . import registry


def get(
    name: str,
    config: dict,
    input_shape: tuple[int, int, int],
    num_classes: int | None = None,
    labelmap: dict | None = None,
    post_tfm: Callable | None = None,
) -> lightning.LightningModule:
    """Get a model instance, given its name and
    a configuration as a :class:`dict`.

    Parameters
    ----------
    name : str
        Model name, must be one of vak.models.registry.MODEL_NAMES.
    config: dict
        Model configuration in a ``dict``,
        as loaded from a .toml file,
        and used by the model method ``from_config``.
    num_classes : int
        Number of classes model will be trained to classify.
    input_shape : tuple
        Of int values, sizes of dimensions,
        e.g. (channels, height, width).
        Batch size is not required for input shape.
    post_tfm : callable
        Post-processing transform that models applies during evaluation.
        Default is None, in which case the model defaults to using
        ``vak.transforms.frame_labels.ToLabels`` (that does not
        apply any post-processing clean-ups).
        To be valid, ``post_tfm`` must be either an instance of
        ``vak.transforms.frame_labels.PostProcess``.

    Returns
    -------
    model : lightning.LightningModule
        Instance of :class:`lightning.LightningModule`,
        one of the model familes.
    """
    # we do this dynamically so we always get all registered models
    try:
        model_factory = registry.MODEL_REGISTRY[name]
    except KeyError as e:
        raise ValueError(
            f"Invalid model name: '{name}'.\n"
            f"Valid model names are: {registry.MODEL_NAMES}"
        ) from e

    model_family = registry.MODEL_FAMILY_FROM_NAME[name]

    if model_family == "FrameClassificationModel":
        # still need to special case model logic here
        net_init_params = list(
            inspect.signature(
                model_factory.definition.network.__init__
            ).parameters.keys()
        )
        if ("num_input_channels" in net_init_params) and (
            "num_freqbins" in net_init_params
        ):
            num_input_channels = input_shape[-3]
            num_freqbins = input_shape[-2]
            config["network"].update(
                num_classes=num_classes,
                num_input_channels=num_input_channels,
                num_freqbins=num_freqbins,
            )
        else:
            raise ValueError(
                f"Detected that model with name '{name}' was family '{model_family}', but "
                f"unable to determine network init arguments for model. Currently all models "
                f"in this family must have networks with parameters ``num_input_channels`` and ``num_freqbins``"
            )
        model = model_factory.from_config(
            config=config, labelmap=labelmap, post_tfm=post_tfm
        )
    elif model_family == "ParametricUMAPModel":
        encoder_init_params = list(
            inspect.signature(
                model_factory.definition.network["encoder"].__init__
            ).parameters.keys()
        )
        if "input_shape" in encoder_init_params:
            if "encoder" in config["network"]:
                config["network"]["encoder"].update(input_shape=input_shape)
            else:
                config["network"]["encoder"] = dict(input_shape=input_shape)

        model = model_factory.from_config(config=config)
    else:
        raise ValueError(
            f"Value for ``model_family`` not recognized: {model_family}"
        )

    return model
