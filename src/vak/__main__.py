"""
Invokes __main__ when the module is run as a script.
Example: python -m vak --help
"""
import argparse
import os
from glob import glob

from .cli import cli


def get_parser():
    """returns ArgumentParser instance used by main()"""
    CHOICES = [
        'prep',
        'train',
        'predict',
        'finetune',
        'learncurve',
        'summary',
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
    parser.add_argument('-d', '--dataset', type=str,
                        help='Create a dataset from a list of files '
                             'in a .txt file, by passing in the name of the .txt file.\n'
                             '$ vak --dataset ./audio_files_list.txt')
    parser.add_argument('-g', '--glob', type=str,
                        help='string to use with glob function '
                             'to search for config files fitting some pattern.\n'
                             '$ vak --glob ./config_finches*.ini')
    parser.add_argument('-t', '--txt', type=str,
                        help='name of .txt file containing list of '
                             'config files to run\n'
                             '$ vak --text ./list_of_config_filenames.txt')
    return parser


def main():
    """main function, called when package is run with `python -m vak` or
    alternatively just calling `vak` at the command line (because this
    function is installed under just `vak` as a console script entry point,
    see setup.py)
    """
    parser = get_parser()
    args = parser.parse_args()
    # logic for handling glob + txt flags, affects how conifg

    if args.glob:
        config_files = glob(args.glob)
    elif args.txt:
        with open(args.txt, 'r') as config_list_file:
            config_files = config_list_file.readlines()
    elif args.configfile:
        config_files = [args.configfile]  # single-item list

    if args.glob or args.txt:
        config_files_cleaned = []
        for config_file in config_files:
            config_file = config_file.rstrip()
            config_file = config_file.lstrip()
            if os.path.isfile(config_file):
                config_files_cleaned.append(config_file)
            else:
                if args.txt:
                    txt_dir = os.path.dirname(args.txt)
                    txt_dir_config = os.path.join(txt_dir,
                                                  config_file)
                    if os.path.isfile(txt_dir_config):
                        config_files_cleaned.append(txt_dir_config)
                    else:
                        raise FileNotFoundError("Can't find config file: {}"
                                                .format(txt_dir_config))
                else:  # if --glob was used instead of --txt
                    raise FileNotFoundError("Can't find config file: {}"
                                            .format(config_file))
        config_files = config_files_cleaned

    cli(command=args.command,
        config_files=config_files)


if __name__ == "__main__":
    main()
