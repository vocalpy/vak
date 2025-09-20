"""
Invokes __main__ when the module is run as a script.
Example: python -m vak --help
"""
import sys

from .cli import cli


def main(args=None):
    """Main function called when run as script or through command-line interface

    called when package is run with `python -m vak` or
    alternatively just calling `vak` at the command line (because this
    function is installed under just `vak` as a console script)

    ``args`` is used for unit testing only
    """
    parser = cli.get_parser()

    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)

    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)
    cli.cli(args)


if __name__ == "__main__":
    main()
