"""module that contains helper function to load models

Models in separate packages should make themselves available to vak by including
'vak.models' in the entry_points of their setup.py file.

For example, if you had a package `grunet` containing a model
that was instantiated by a function `GRUnet`,
then that package would include the following in its setup.py file:

setup(
    ...
    entry_points={'vak.models': 'GRUnet = grunet:GRUnet'},
    ...
)

For more detail on entry points in Python, see:
https://packaging.python.org/guides/creating-and-discovering-plugins/#using-package-metadata
https://setuptools.readthedocs.io/en/latest/setuptools.html#dynamic-discovery-of-services-and-plugins
https://amir.rachum.com/blog/2017/07/28/python-entry-points/
"""
from .. import entry_points

MODELS_ENTRY_POINT = "vak.models"


def find():
    """find installed vak.models

    returns generator that yields model name and function for loading
    """
    for entrypoint in entry_points._iter(MODELS_ENTRY_POINT):
        yield entrypoint.name, entrypoint.load()


def from_model_config_map(model_config_map,
                          # TODO: move num_classes / input_shape into model configs
                          num_classes,
                          input_shape,
                          labelmap):
    """get models that are ready to train, given their names and configurations.

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
    MODELS = {model_name: model_builder for model_name, model_builder in find()}

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
            model = MODELS[model_name].from_config(config=model_config, labelmap=labelmap)
        except KeyError:
            model = MODELS[f"{model_name}Model"].from_config(
                config=model_config,
                labelmap=labelmap
            )
        models_map[model_name] = model
    return models_map
