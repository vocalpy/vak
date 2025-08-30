"""This script gets run by continuous integration 
(in ./github/workflows/ci-{os}.yml files)
so that all the paths are correct on the virtual machines
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
PROJ_ROOT = HERE / ".." / ".."
PROJ_ROOT_ABS = PROJ_ROOT.resolve()  # <- used to fix paths!!!
GENERATED_TEST_DATA = PROJ_ROOT / "tests" / "data_for_tests" / "generated"


def main():
    """loads csv files created by `prep` and changes the parent of paths so it's
    the local file system, instead of what's on my laptop.
    To get tests to run on CI without FileNotFound errors"""
    prep_csvs = sorted(GENERATED_TEST_DATA.glob("**/*prep*csv"))
    for prep_csv in prep_csvs:
        vak_df = pd.read_csv(prep_csv)
        for path_column_name in ("spect_path", "audio_path", "annot_path"):
            paths = vak_df[path_column_name].values.tolist()
            paths = [str(path) for path in paths]
            new_column = []
            for path_str in paths:
                if path_str == "nan":
                    new_column.append(path_str)
                    continue
                tests_root_ind = path_str.find('tests/data_for_tests')
                if (tests_root_ind == -1
                    and path_column_name == 'audio_path'
                    and 'spect_mat_annot_yarden' in str(prep_csv)):
                    # prep somehow gives root to audio -- from annotation?; we don't need these to exist though
                    new_column.append(path_str)
                    continue
                new_path_str = path_str[tests_root_ind:]  # get rid of parent directories
                new_path = PROJ_ROOT_ABS / new_path_str
                if not new_path.exists():
                    raise FileNotFoundError(
                        f"New path does not exist:\n{new_path}"
                    )
                new_column.append(str(new_path))
            vak_df[path_column_name] = new_column
        vak_df.to_csv(prep_csv)


main()
