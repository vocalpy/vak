from .eval import eval
from .train import train
from .learncurve import learning_curve
from .predict import predict
from .prep import prep


def cli(command, config_file):
    """command-line interface

    Parameters
    ----------
    command : string
        One of {'prep', 'train', 'eval', 'predict', 'finetune', 'learncurve'}
    config_file : str, Path
        path to a config.toml file
    """
    if command == 'prep':
        prep(toml_path=config_file)

    elif command == 'train':
        train(toml_path=config_file)

    elif command == 'eval':
        eval(toml_path=config_file)

    elif command == 'predict':
        predict(toml_path=config_file)

    elif command == 'learncurve':
        learning_curve(toml_path=config_file)

    elif command == 'finetune':
        raise NotImplementedError

    else:
        raise ValueError(
            f'command not recognized: {command}'
        )
