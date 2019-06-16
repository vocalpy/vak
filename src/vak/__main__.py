"""
Invokes __main__ when the module is run as a script.
Example: python -m vak --help
"""
import argparse

from .cli import cli


def get_parser():
    """returns ArgumentParser instance used by main()"""
    CHOICES = [
        'prep',
        'train',
        'predict',
        'finetune',
        'learncurve',
    ]

    parser = argparse.ArgumentParser(description='vak command-line interface',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('command', type=str, metavar='command',
                        choices=CHOICES,
                        help="Command to run, valid options are:\n"
                             f"{CHOICES}\n"
                             "$ vak train ./configs/config_2018-12-17.ini")
    parser.add_argument('configfile', type=str,
                        help='name of config.ini file to use \n'
                             '$ vak train ./configs/config_2018-12-17.ini')
    return parser


def main():
    """main function, called when package is run with `python -m vak` or
    alternatively just calling `vak` at the command line (because this
    function is installed under just `vak` as a console script entry point,
    see setup.py)
    """
    parser = get_parser()
    args = parser.parse_args()
    cli(command=args.command,
        config_file=args.configfile)


if __name__ == "__main__":
    main()
