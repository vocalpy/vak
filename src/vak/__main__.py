"""
Invokes __main__ when the module is run as a script.
Example: python -m vak --help
The same function is run by the script `vak-cli` which is installed on the
path by pip, so `$ vak-cli --help` would have the same effect (i.e., no need
to type the python -m)
"""

import argparse
import os
from glob import glob

import vak

parser = argparse.ArgumentParser(description='main script',
                                 formatter_class=argparse.RawTextHelpFormatter,)
parser.add_argument('-c', '--config', type=str,
                    help='run learning curve experiment with a single config'
                         '.ini file, by passing the name of that file.\n'
                         '$ vak-cli --config ./config_bird1.ini')
parser.add_argument('-d', '--dataset', type=str,
                    help='Create a dataset from a list of files '
                         'in a .txt file, by passing in the name of the .txt file.\n'
                         '$ vak-cli --dataset ./audio_files_list.txt')
parser.add_argument('-g', '--glob', type=str,
                    help='string to use with glob function '
                         'to search for config files fitting some pattern.\n'
                         '$ vak-cli --glob ./config_finches*.ini')
parser.add_argument('-p', '--predict', type=str,
                    help='predict segments and labels for song, using a trained '
                         'model specified in a single config.ini file\n'
                         '$ vak-cli --predict ./predict_bird1.ini')
parser.add_argument('-s', '--summary', type=str,
                    help='runs function that summarizes results from generating'
                         'a learning curve, using a single config.ini file\n'
                         '$ vak-cli --summary ./config_bird1.ini')
parser.add_argument('-t', '--txt', type=str,
                    help='name of .txt file containing list of '
                         'config files to run\n'
                         '$ vak-cli --text ./list_of_config_filenames.txt')
args = parser.parse_args()


def main():
    if sum([bool(arg)
            for arg in [args.config, args.glob, args.predict, args.summary,
                        args.txt,]]) != 1:
        parser.error("Please specify exactly one of the following flags: "
                     "--glob, --txt, --config, or --predict.\n"
                     "Run 'vak-cli --help' for an explanation of each.")

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
            config = vak.config.parse.parse_config(config_file)
            vak.cli.make_data(labelset=config.data.labelset,
                                   all_labels_are_int=config.data.all_labels_are_int,
                                   data_dir=config.data.data_dir,
                                   total_train_set_dur=config.data.total_train_set_dur,
                                   val_dur=config.data.val_dur,
                                   test_dur=config.data.test_dur,
                                   config_file=config_file,
                                   silent_gap_label=config.data.silent_gap_label,
                                   skip_files_with_labels_not_in_labelset=config.data.skip_files_with_labels_not_in_labelset,
                                   output_dir=config.data.output_dir,
                                   mat_spect_files_path=config.data.mat_spect_files_path,
                                   mat_spects_annotation_file=config.data.mat_spects_annotation_file,
                                   spect_params=config.spect_params)

            # get config again because make_data changed train_data_dict_path and val_data_dict_path
            config = vak.config.parse.parse_config(config_file)
            vak.cli.learncurve(train_data_dict_path=config.train.train_data_dict_path,
                                    val_data_dict_path=config.train.val_data_dict_path,
                                    spect_params=config.spect_params,
                                    total_train_set_duration=config.data.total_train_set_dur,
                                    train_set_durs=config.train.train_set_durs,
                                    num_replicates=config.train.num_replicates,
                                    num_epochs=config.train.num_epochs,
                                    config_file=config_file,
                                    networks=config.networks,
                                    val_error_step=config.train.val_error_step,
                                    checkpoint_step=config.train.checkpoint_step,
                                    patience=config.train.patience,
                                    save_only_single_checkpoint_file=config.train.save_only_single_checkpoint_file,
                                    normalize_spectrograms=config.train.normalize_spectrograms,
                                    use_train_subsets_from_previous_run=config.train.use_train_subsets_from_previous_run,
                                    previous_run_path=config.train.previous_run_path,
                                    root_results_dir=config.output.root_results_dir)

            # get config again because learncurve changed results_dir_made_by_main_script
            config = vak.config.parse.parse_config(config_file)
            vak.cli.summary(results_dirname=config.output.results_dirname,
                                 train_data_dict_path=config.train.train_data_dict_path,
                                 networks=config.networks,
                                 train_set_durs=config.train.train_set_durs,
                                 num_replicates=config.train.num_replicates,
                                 labelset=config.data.labelset,
                                 test_data_dict_path=config.train.test_data_dict_path,
                                 normalize_spectrograms=config.train.normalize_spectrograms)
    elif args.predict:
        vak.cli.predict(args.predict)
    elif args.summary:
        vak.cli.summary(args.summary)


if __name__ == "__main__":
    main()
