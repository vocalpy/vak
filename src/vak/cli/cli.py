"""Implements the vak command-line interface"""
import argparse
import pathlib
from dataclasses import dataclass
from typing import Callable


def eval(args):
    from .eval import eval

    eval(toml_path=args.configfile)


def train(args):
    from .train import train

    train(toml_path=args.configfile)


def learncurve(args):
    from .learncurve import learning_curve

    learning_curve(toml_path=args.configfile)


def predict(args):
    from .predict import predict

    predict(toml_path=args.configfile)


def prep(args):
    from .prep import prep

    prep(toml_path=args.configfile)


def configfile(args):
    from .. import config
    config.generate(
        kind=args.kind,
        add_prep=args.add_prep,
        dst=args.dst,
    )


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


def add_single_arg_configfile_to_command(
    cli_command,
    cli_command_parser
):
    """Most of the CLICommands call this function
    to add arguments to their sub-parser.
    It adds a single positional argument, `configfile`.
    Not to be confused with the *command* configfile,
    that adds different arguments
    """
    cli_command_parser.add_argument(
        "configfile",
        type=pathlib.Path,
        help="name of TOML configuration file to use \n"
        f"$ vak {cli_command.name} ./configs/config_rat01337.toml",
    )


KINDS_OF_CONFIG_FILES = [
    # FIXME: there's no way to have a stand-alone prep file right now
    # we need to add a `purpose` key-value pair to the file format
    # to make this possible
    # "prep",
    "train",
    "eval",
    "predict",
    "learncurve",
]


def add_args_to_configfile_command(
    cli_command,
    cli_command_parser
):
    """This is the function that gets called
    to add arguments to the sub-parser 
    for the configfile command
    """
    cli_command_parser.add_argument(
        "kind",
        type=str,
        choices=KINDS_OF_CONFIG_FILES,
        help="kind: the kind of TOML configuration file to generate"
    )
    cli_command_parser.add_argument(
        "--add-prep",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Adding this option will add a 'prep' table to the TOML configuration file. Default is False."
    )
    cli_command_parser.add_argument(
        "--dst",
        type=pathlib.Path,
        default=pathlib.Path.cwd(),
        help="Destination, where TOML configuration file should be generated. Default is current working directory."
    )
    # TODO: add this option
    # cli_command_parser.add_argument(
    #     "--from",
    #     type=pathlib.Path,
    #     help="Path to another configuration file that this file should be generated from\n"
    # )


CLI_COMMANDS = [
    CLICommand(
        name='prep',
        help='prepare a dataset',
        func=prep,
        add_parser_args_func=add_single_arg_configfile_to_command,
    ),
    CLICommand(
        name='train',
        help='train a model',
        func=train,
        add_parser_args_func=add_single_arg_configfile_to_command,
    ),
    CLICommand(
        name='eval',
        help='evaluate a trained model',
        func=eval,
        add_parser_args_func=add_single_arg_configfile_to_command,
    ),
    CLICommand(
        name='predict',
        help='generate predictions from trained model',
        func=predict,
        add_parser_args_func=add_single_arg_configfile_to_command,
    ),
    CLICommand(
        name='learncurve',
        help='run a learning curve',
        func=learncurve,
        add_parser_args_func=add_single_arg_configfile_to_command,
    ),
    CLICommand(
        name='configfile',
        help='generate a TOML configuration file for vak',
        func=configfile,
        add_parser_args_func=add_args_to_configfile_command,
    ),
]


def get_parser():
    """returns ArgumentParser instance used by main()"""
    parser = argparse.ArgumentParser(
        prog="vak",
        description="Vak command-line interface",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # create sub-parser
    sub_parsers = parser.add_subparsers(
        title="Command",
        description="Commands for the vak command-line interface",
        dest="command",
        required=True,
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


def cli(args: argparse.Namespace):
    """Execute the commands of the command-line interface.

    Parameters
    ----------
    args : argparse.Namespace
        Result of calling :meth:`ArgumentParser.parse_args` 
        on the :class:`ArgumentParser` instance returned by 
        :func:`vak.cli.cli.get_parser`.
    """
    if args.command in CLI_COMMAND_FUNCTION_MAP:
        CLI_COMMAND_FUNCTION_MAP[args.command](args)
    else:
        raise ValueError(f"command not recognized: {args.command}")
