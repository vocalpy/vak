"""
Invokes __main__ when the module is run as a script.
Example: python -m tweetynet --help
The same function is run by the script `tweetynet-cli` which is installed on the
path by pip, so `$ tweetynet-cli --help` would have the same effect (i.e., no need
to type the python -m)
"""

import argparse
import os
from glob import glob

import tweetynet

parser = argparse.ArgumentParser(description='main script',
                                 formatter_class=argparse.RawTextHelpFormatter,)
parser.add_argument('-c', '--config', type=str,
                    help='run learning curve experiment with a single config'
                    '.ini file, by passing the name of that file.\n'
                            '$ cnn-bilstm --config ./config_bird1.ini')
parser.add_argument('-g', '--glob', type=str,
                    help='string to use with glob function '
                         'to search for config files fitting some pattern.\n'
                         '$ cnn-bilstm --glob ./config_finches*.ini')
parser.add_argument('-p', '--predict', type=str,
                    help='predict segments and labels for song, using a trained '
                         'model specified in a single config.ini file\n'
                         '$ cnn-bilstm --predict ./predict_bird1.ini')
parser.add_argument('-s', '--summary', type=str,
                    help='runs function that summarizes results from generating'
                         'a learning curve, using a single config.ini file\n'
                         '$ cnn-bilstm --summary ./config_bird1.ini')
parser.add_argument('-t', '--txt', type=str,
                    help='name of .txt file containing list of '
                         'config files to run\n'
                         '$ cnn-bilstm --text ./list_of_config_filenames.txt')
args = parser.parse_args()


def main():
    if sum([bool(arg)
            for arg in [args.config, args.glob, args.predict, args.summary,
                        args.txt,]]) != 1:
        parser.error("Please specify exactly one of the following flags: "
                     "--glob, --txt, --config, or --predict.\n"
                     "Run 'cnn-bilstm --help' for an explanation of each.")

    if args.glob:
        config_files = glob(args.glob)
    elif args.txt:
        with open(args.txt, 'r') as config_list_file:
            config_files = config_list_file.readlines()
    elif args.config:
        config_files = [args.config]  # single-item list


    if args.glob or args.txt or args.config:
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

        for config_file in config_files:
            tweetynet.make_data(config_file)
            tweetynet.train(config_file)
            tweetynet.learn_curve(config_file)
    elif args.predict:
        tweetynet.cli.predict(args.predict)
    elif args.summary:
        tweetynet.cli.summary(args.summary)


if __name__ == "__main__":
    main()
