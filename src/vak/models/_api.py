from .das import DAS
from .tweetynet import TweetyNet
from .teenytweetynet import TeenyTweetyNet


# TODO: Replace constant with decorator that registers models, https://github.com/vocalpy/vak/issues/623
MODEL_FAMILY_NAME_CLASS_MAPS = {
    'frame classification':
        {
            'DAS': DAS,
            'TweetyNet': TweetyNet,
            'TeenyTweetyNet': TeenyTweetyNet
        }
}

# used by e.g. vak.train to figure out which train function to call
MODEL_FAMILY_FROM_NAME = {
    model_name: family_name
    for family_name, family_dict in MODEL_FAMILY_NAME_CLASS_MAPS.items()
    for model_name, model_class in family_dict.items()
}

BUILTIN_MODELS = {
    model_name: model_class
    for family_name, family_dict in MODEL_FAMILY_NAME_CLASS_MAPS.items()
    for model_name, model_class in family_dict.items()
}

MODEL_NAMES = list(BUILTIN_MODELS.keys())
