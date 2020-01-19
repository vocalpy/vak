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
import attr


MODELS_ENTRY_POINT = 'vak.models'
MODEL_CONFIG_PARSERS_ENTRY_POINT = 'vak.model_config_parsers'


def iter_entry_points(group_name):
    try:
        import pkg_resources
    except (ImportError, IOError):
        return []

    return pkg_resources.iter_entry_points(group_name)


def find():
    for entrypoint in iter_entry_points(MODELS_ENTRY_POINT):
        yield entrypoint.name, entrypoint.load()


def find_config_parsers():
    for entrypoint in iter_entry_points(MODEL_CONFIG_PARSERS_ENTRY_POINT):
        yield entrypoint.name, entrypoint.load()


def from_model_config_map(model_config_map, num_classes, input_shape):
    """get models that are ready to train, given their names and configurations.

    Given a dictionary that maps model names to configurations,
    along with the number of classes they should be trained to discriminate and their input shape,
    return a dictionary that maps model names to instances of the model

    Parameters
    ----------
    model_config_map : dict
    num_classes : int
    input_shape : tuple, list

    Returns
    -------
    models_map : dict
        where keys are model names and values are instances of the models, ready for training
    """
    MODELS = {model_name: model_builder for model_name, model_builder in find()}
    MODEL_CONFIG_PARSERS = {model_name: model_config_parser
                            for model_name, model_config_parser in find_config_parsers()}

    models_map = {}
    for model_name, model_config_kwargs in model_config_map.items():
        # pass section dict as kwargs to config parser function
        model_config = MODEL_CONFIG_PARSERS[model_name](num_classes=num_classes,
                                                        input_shape=input_shape,
                                                        **model_config_kwargs)
        # now convert the config itself to a dict and pass as kwargs to the model
        if type(model_config) != dict:
            # if not a dict, assume an attrs-based class for config, and convert to dict
            model_config = attr.asdict(model_config)
        model = MODELS[model_name](**model_config)
        models_map[model_name] = model
    return models_map
