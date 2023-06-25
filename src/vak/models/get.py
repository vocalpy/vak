"""Function that gets an instance of a model,
given its name and a configuration as a dict."""
from __future__ import annotations
from typing import Callable

from ._api import MODEL_NAMES


def get(name: str,
        config: dict,
        num_classes: int,
        input_shape: tuple[int, int, int],
        labelmap: dict,
        post_tfm: Callable | None = None):
    """Get a model instance, given its name and
    a configuration as a :class:`dict`.

    Parameters
    ----------
    name : str
        Model name, must be one of vak.models.MODEL_NAMES.
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
    model : vak.models.Model
        Instance of a sub-class of the base Model class,
        e.g. a TweetyNet instance.
    """
    import vak.models

    if name == 'DAS':
        n_audio_channels = input_shape[-2]
        num_samples = input_shape[-1]
        config["network"].update(
            num_classes=num_classes,
            n_audio_channels=n_audio_channels,
            num_samples=num_samples
        )
    elif name in ('TweetyNet', 'TeenyTweetyNet'):
        config["network"].update(
            num_classes=num_classes,
            input_shape=input_shape,
        )
    else:
        raise ValueError(
            f"Invalid model name: '{name}'.\nValid model names are: {MODEL_NAMES}"
        )

    try:
        model_class = getattr(vak.models, name)
    except AttributeError as e:
        raise ValueError(
            f"Invalid model name: '{name}'.\nValid model names are: {MODEL_NAMES}"
        ) from e

    model = model_class.from_config(config=config, labelmap=labelmap, post_tfm=post_tfm)

    return model
