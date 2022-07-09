def eval(toml_path):
    from .eval import eval
    eval(toml_path=toml_path)


def train(toml_path):
    from .train import train
    train(toml_path=toml_path)


def learncurve(toml_path):
    from .learncurve import learning_curve
    learning_curve(toml_path=toml_path)


def predict(toml_path):
    from .predict import predict
    predict(toml_path=toml_path)


def prep(toml_path):
    from .prep import prep
    prep(toml_path=toml_path)


COMMAND_FUNCTION_MAP = {
    "prep": prep,
    "train": train,
    "eval": eval,
    "predict": predict,
    "learncurve": learncurve,
}


CLI_COMMANDS = tuple(COMMAND_FUNCTION_MAP.keys())


def cli(command, config_file):
    """Execute the commands of the command-line interface.

    Parameters
    ----------
    command : string
        One of {'prep', 'train', 'eval', 'predict', 'learncurve'}
    config_file : str, Path
        path to a config.toml file
    """
    if command in COMMAND_FUNCTION_MAP:
        COMMAND_FUNCTION_MAP[command](toml_path=config_file)
    else:
        raise ValueError(f"command not recognized: {command}")
