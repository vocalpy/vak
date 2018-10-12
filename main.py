import argparse
from glob import glob
import os

import cnn_bilstm

parser = argparse.ArgumentParser(description='main script that generates '
                                             'learning curves')
parser.add_argument('-g', '--glob', type=str,
                    help='string to use with glob function '
                         'to search for config files')
parser.add_argument('-t', '--txt', type=str,
                    help='name of .txt file containing list of '
                         'config files to run')
args = parser.parse_args()

if __name__ == '__main__':
    if not args.glob and not args.txt:
        parser.error('Either --glob or --txt required.')

    if args.glob and args.txt:
        parser.error('Cannot use --glob and --txt together.')

    if args.glob:
        config_files = glob(args.glob_string)
    elif args.txt:
        with open(args.txt, 'r') as config_list_file:
            config_files = config_list_file.readlines()

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
        cnn_bilstm.make_data(config_file)
        cnn_bilstm.train(config_file)
        cnn_bilstm.learn_curve(config_file)
