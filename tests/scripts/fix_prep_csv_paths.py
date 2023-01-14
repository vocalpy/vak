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
            for path in paths:
                if path == "nan":
                    continue
                try:
                    before, aft = path.split("vak/")
                except ValueError:  # used different name for directory locally
                    before, aft = path.split("vak-vocalpy/")
                new_path = PROJ_ROOT_ABS.joinpath(aft)
                new_column.append(str(new_path))
            vak_df[path_column_name] = pd.Series(new_column)
        vak_df.to_csv(prep_csv)


if __name__ == "__main__":
    main()
