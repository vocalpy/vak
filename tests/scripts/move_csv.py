from argparse import ArgumentParser
from pathlib import Path, PosixPath, WindowsPath

import numpy as np
import pandas as pd


def main(
    csv, os, audio_path_parent=None, spect_path_parent=None, annot_path_parent=None
):
    """ "move" the paths in a .csv file saved by `vak prep`,
    by keeping the filename but changing the parent path

    Parameters
    ----------
    csv : str
        prep csv filename
    os : str
        operating system to move to.
        one of {'windows', 'unix'}.
        Determines separators used for path.
    audio_path_parent : str
        new parent for paths in audio_path column
    spect_path_parent : str
        new parent for paths in spect_path column
    annot_path_parent : str
        new parent for paths in annot_path column
    """
    csv = Path(csv)
    if not csv.exists():
        raise FileNotFoundError(f"csv not found: {csv}")

    if os == "windows":
        ToPath = WindowsPath
    elif os == "unix":
        ToPath = PosixPath
    else:
        raise ValueError(f"invalid value for os: {os}")

    vak_df = pd.read_csv(csv)

    for path_column, new_parent in zip(
        ("audio_path", "spect_path", "annot_path"),
        (audio_path_parent, spect_path_parent, annot_path_parent),
    ):
        if new_parent is None:
            continue

        new_parent = ToPath(new_parent)
        if not new_parent.exists():
            raise NotADirectoryError(
                f"new parent not recognized as a directory: {new_parent}"
            )
        new_parent = new_parent.expanduser()

        paths = vak_df[path_column].values
        new_path_col = []
        for path in paths:
            old_path = Path(path)
            new_path = str(new_parent / old_path.name)
            new_path_col.append(new_path)
        vak_df[path_column] = np.array(new_path_col)

    vak_df.to_csv(csv, index=False)


TO_CHOICES = ("windows", "unix")


def get_parser():
    parser = ArgumentParser()
    parser.add_argument("csv", type=str, help='csv file that should be "moved"')
    parser.add_argument(
        "os",
        choices=TO_CHOICES,
        help=f"which OS to move paths in .csv to: {TO_CHOICES}",
    )
    parser.add_argument("--audio_path_parent", type=str)
    parser.add_argument("--spect_path_parent", type=str)
    parser.add_argument("--annot_path_parent", type=str)
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(
        csv=args.csv,
        os=args.os,
        audio_path_parent=args.audio_path_parent,
        spect_path_parent=args.spect_path_parent,
        annot_path_parent=args.annot_path_parent,
    )
