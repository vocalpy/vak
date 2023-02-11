"""Helper function to load models"""
from __future__ import annotations

from .tweetynet import TweetyNet
from .teenytweetynet import TeenyTweetyNet


# TODO: Replace constant with decorator that registers models, https://github.com/vocalpy/vak/issues/623
BUILTIN_MODELS = {
    'TweetyNet': TweetyNet,
    'TeenyTweetyNet': TeenyTweetyNet
}

MODEL_NAMES = list(BUILTIN_MODELS.keys())


def from_model_config_map(model_config_map: dict[str: dict],
                          # TODO: move num_classes / input_shape into model configs
                          num_classes: int,
                          input_shape: tuple[int, int, int],
                          labelmap: dict) -> dict:
    """Get models that are ready to train, given their names and configurations.

    Given a dictionary that maps model names to configurations,
    along with the number of classes they should be trained to discriminate and their input shape,
    return a dictionary that maps model names to instances of the model

    Parameters
    ----------
    model_config_map : dict
        where each key-value pair is model name : dict of config parameters
    num_classes : int
        number of classes model will be trained to classify
    input_shape : tuple, list
        e.g. (channels, height, width).
        Batch size is not required for input shape.
    post_tfm : callable
        Post-processing transform that models applies during evaluation.
        Default is None, in which case the model defaults to using
        ``vak.transforms.labeled_timebins.ToLabels`` (that does not
        apply any post-processing clean-ups).
        To be valid, ``post_tfm`` must be either an instance of
        ``vak.transforms.labeled_timebins.ToLabels`` or
        ``vak.transforms.labeled_timebins.ToLabelsWithPostprocessing``.

    Returns
    -------
    models_map : dict
        where keys are model names and values are instances of the models, ready for training
    """
    import vak.models

    models_map = {}
    for model_name, model_config in model_config_map.items():
        # pass section dict as kwargs to config parser function
        # TODO: move num_classes / input_shape into model configs
        # TODO: add labelmap to config dynamically if needed? outside this function
        model_config["network"].update(
            num_classes=num_classes,
            input_shape=input_shape,
        )

        try:
            model_class = getattr(vak.models, model_name)
        except AttributeError as e:
            raise ValueError(
                f"Invalid model name: '{model_name}'.\nValid model names are: {MODEL_NAMES}"
            ) from e

        model = model_class.from_config(config=model_config, labelmap=labelmap)
        models_map[model_name] = model

    return models_map
