"""This script gets run by continuous integration 
(in ./github/workflows/ci-{os}.yml files)
so that all the paths are correct on the virtual machines
"""
import argparse
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
PROJ_ROOT = HERE / ".." / ".."
PROJ_ROOT_ABS = PROJ_ROOT.resolve()  # <- used to fix paths!!!
GENERATED_TEST_DATA = PROJ_ROOT / "tests" / "data_for_tests" / "generated"


def main(current_proj_dirname, new_proj_root, strict=False):
    """loads csv files created by `prep` and changes the parent of paths so it's
    the local file system, instead of what's on my laptop.
    To get tests to run on CI without FileNotFound errors"""
    if not current_proj_dirname.endswith('/'):
        current_proj_dirname = current_proj_dirname + '/'  # so split works correctly below
    new_proj_root = Path(new_proj_root)

    prep_csvs = sorted(GENERATED_TEST_DATA.glob("**/*prep*csv"))
    for prep_csv in prep_csvs:
        vak_df = pd.read_csv(prep_csv)
        for path_column_name in ("spect_path", "audio_path", "annot_path"):
            paths = vak_df[path_column_name].values.tolist()
            paths = [str(path) for path in paths]
            new_column = []
            for path in paths:
                if path == "nan":
                    continue
                if current_proj_dirname not in str(path):
                    raise ValueError(
                        f'`current_proj_dirname` not found in path.'
                        f'`current_proj_dirname`={current_proj_dirname}.'
                        f'`path`={path}'
                    )
                before, aft = path.split(current_proj_dirname)
                new_path = new_proj_root.joinpath(aft)
                if strict:
                    if not new_path.exists():
                        raise FileNotFoundError(
                            f"--strict flag was set and new path does not exist:\n{path}"
                        )
                new_column.append(str(new_path))
            vak_df[path_column_name] = pd.Series(new_column)
        vak_df.to_csv(prep_csv)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--current-proj-dirname', default='vak/',
        help='Name of current project directory, the name in paths that will be replaced in vak prep .csv files'
    )
    parser.add_argument(
        '--new-proj-root', default=PROJ_ROOT_ABS,
        help=('Path to new root, that will replace --current-proj-dirname '
              'and everything before ti in paths in vak prep .csv files'),
    )
    parser.add_argument(
        '--strict', action='store_true',
        help=("If this flag is set, throw an error if the fixed paths don't exist"),
    )
    return parser


parser = get_parser()
args = parser.parse_args()

main(args.current_proj_dirname, args.new_proj_root, args.strict)
