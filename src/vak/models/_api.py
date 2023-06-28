from typing import Any

from . import registry


def __getattr__(name: str) -> Any:
    """Module-level __getattr__ function that we use to dynamically determine models."""
    if name == 'MODEL_FAMILY_FROM_NAME':
        return {
            model_name: family_name
            for family_name, family_dict in registry.MODELS_BY_FAMILY_REGISTRY.items()
            for model_name, model_class in family_dict.items()
        }
    elif name == 'BUILTIN_MODELS':
        return {
            model_name: model_class
            for family_name, family_dict in registry.MODELS_BY_FAMILY_REGISTRY.items()
            for model_name, model_class in family_dict.items()
        }
    elif name == 'MODEL_NAMES':
        return list(
            {
                model_name: model_class
                for family_name, family_dict in registry.MODELS_BY_FAMILY_REGISTRY.items()
                for model_name, model_class in family_dict.items()
            }.keys()
        )
    else:
        raise AttributeError(
            f"Not an attribute of `vak.models._api`: {name}"
        )
