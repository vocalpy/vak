"""
Invokes __main__ when the module is run as a script.
Example: python -m vak --help
"""
import argparse
from pathlib import Path

from .cli import cli


def get_parser():
    """returns ArgumentParser instance used by main()"""
    parser = argparse.ArgumentParser(
        prog='vak',
        description="vak command-line interface",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "command",
        type=str,
        metavar="command",
        choices=cli.CLI_COMMANDS,
        help="Command to run, valid options are:\n"
        f"{cli.CLI_COMMANDS}\n"
        "$ vak train ./configs/config_2018-12-17.toml",
    )
    parser.add_argument(
        "configfile",
        type=Path,
        help="name of config.toml file to use \n"
        "$ vak train ./configs/config_2018-12-17.toml",
    )
    return parser


def main(args=None):
    """Main function called when run as script or through command-line interface

    called when package is run with `python -m vak` or
    alternatively just calling `vak` at the command line (because this
    function is installed under just `vak` as a console script)

    ``args`` is used for unit testing only
    """
    if args is None:
        parser = get_parser()
        args = parser.parse_args()
    cli.cli(command=args.command, config_file=args.configfile)


if __name__ == "__main__":
    main()
