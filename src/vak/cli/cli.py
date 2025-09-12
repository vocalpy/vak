"""Implements the vak command-line interface"""
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


def eval(args):
    from .eval import eval

    eval(toml_path=args.config_file)


def train(args):
    from .train import train

    train(toml_path=args.config_file)


def learncurve(args):
    from .learncurve import learning_curve

    learning_curve(toml_path=args.config_file)


def predict(args):
    from .predict import predict

    predict(toml_path=args.config_file)


def prep(args):
    from .prep import prep

    prep(toml_path=args.config_file)


@dataclass
class CLICommand:
    """Dataclass representing a cli command
    
    Attributes
    ----------
    name : str
        Name of the command, that gets added to the CLI as a sub-parser
    help : str
        Help for the command, that gets added to the CLI as a sub-parser
    func : Callable
        Function to call for command
    add_parser_args_func: Callable
        Function to call to add arguments to sub-parser representing command
    """
    name: str
    help: str
    func: Callable
    add_parser_args_func : Callable


def add_configfile_arg(
    cli_command,
    cli_command_parser
):
        cli_command_parser.add_argument(
            "configfile",
            type=Path,
            help="name of TOML configuration file to use \n"
            f"$ vak {cli_command.name} ./configs/config_rat01337.toml",
        )


CLI_COMMANDS = [
    CLICommand(
        name='prep',
        help='prepare a dataset',
        func=prep,
        add_parser_args_func=add_configfile_arg,
    ),
    CLICommand(
        name='train',
        help='train a model',
        func=train,
        add_parser_args_func=add_configfile_arg,
    ),
    CLICommand(
        name='eval',
        help='evaluate a trained model',
        func=eval,
        add_parser_args_func=add_configfile_arg,
    ),
    CLICommand(
        name='predict',
        help='generate predictions from trained model',
        func=predict,
        add_parser_args_func=add_configfile_arg,
    ),
    CLICommand(
        name='learncurve',
        help='run a learning curve',
        func=learncurve,
        add_parser_args_func=add_configfile_arg,
    ),
]


def get_parser():
    """returns ArgumentParser instance used by main()"""
    parser = argparse.ArgumentParser(
        prog="vak",
        description="vak command-line interface",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # create sub-parser
    sub_parsers = parser.add_subparsers(
        help='Commands for vak command-line interface',
        dest="command",
    )

    for cli_command in CLI_COMMANDS:
        cli_command_parser = sub_parsers.add_parser(
            cli_command.name,
            help=cli_command.help
        )
        cli_command.add_parser_args_func(
            cli_command,
            cli_command_parser
        )

    return parser


CLI_COMMAND_FUNCTION_MAP = {
    cli_command.name: cli_command.func 
    for cli_command in CLI_COMMANDS
}


def cli(args):
    """Execute the commands of the command-line interface.

    Parameters
    ----------
    command : string
        One of {'prep', 'train', 'eval', 'predict', 'learncurve'}
    config_file : str, Path
        path to a config.toml file
    """
    if args.command in CLI_COMMAND_FUNCTION_MAP:
        CLI_COMMAND_FUNCTION_MAP[args.command](args)
    else:
        raise ValueError(f"command not recognized: {args.command}")
