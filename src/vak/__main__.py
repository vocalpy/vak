"""
Invokes __main__ when the module is run as a script.
Example: python -m vak --help
"""
from .cli import cli


def main(args=None):
    """Main function called when run as script or through command-line interface

    called when package is run with `python -m vak` or
    alternatively just calling `vak` at the command line (because this
    function is installed under just `vak` as a console script)

    ``args`` is used for unit testing only
    """
    if args is None:
        parser = cli.get_parser()
        args = parser.parse_args()
    cli.cli(command=args.command, config_file=args.configfile)


if __name__ == "__main__":
    main()
