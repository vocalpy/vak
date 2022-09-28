"""script to download sample data for vak autoannotate tutorial

Adapted from
https://github.com/NickleDave/bfsongrepo/blob/main/src/scripts/download_dataset.py
"""
from __future__ import annotations
import argparse
import pathlib
import shutil
import sys
import time
from typing import Union
import urllib.request
import warnings


DATA_TO_DOWNLOAD = {
    "gy6or6": {
        "sober.repo1.gy6or6.032212.wav.csv.tar.gz": {
            "MD5": "8c88b46ba87f9784d3690cc8ee4bf2f4",
            "download": "https://figshare.com/ndownloader/files/37509160"
        },
        "sober.repo1.gy6or6.032312.wav.csv.tar.gz": {
            "MD5": "063ba4d50d1b94009b4b00f0a941d098",
            "download": "https://figshare.com/ndownloader/files/37509172"
        }
    }
}


def reporthook(count: int, block_size: int, total_size: int) -> None:
    """hook for urlretrieve that gives us a simple progress report
    https://blog.shichao.io/2012/10/04/progress_speed_indicator_for_urlretrieve_in_python.html
    """
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                    (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()


def download_dataset(download_urls_by_bird_ID: dict,
                     bfsongrepo_dir: pathlib.Path) -> None:
    """download the dataset, given a dict of download urls"""
    tar_dir = bfsongrepo_dir / "tars"
    tar_dir.mkdir()
    # top-level keys are bird ID: bl26lb16, gr41rd51, ...
    for bird_id, tars_dict in download_urls_by_bird_ID.items():
        print(
            f'Downloading .tar files for bird: {bird_id}'
        )
        # bird ID -> dict where keys are .tar.gz filenames mapping to download url + MD5 hash
        for tar_name, url_md5_dict in tars_dict.items():
            print(
                f'Downloading tar: {tar_name}'
            )
            download_url = url_md5_dict['download']
            filename = tar_dir / tar_name
            urllib.request.urlretrieve(download_url, filename, reporthook)
            print('\n')


def extract_tars(bfsongrepo_dir: pathlib.Path) -> None:
    tar_dir = bfsongrepo_dir / "tars"  # made by download_dataset function
    tars = sorted(tar_dir.glob('*.tar.gz'))
    for tar_path in tars:
        print(
            f"\nunpacking: {tar_path}"
        )

        shutil.unpack_archive(
            filename=tar_path,
            extract_dir=bfsongrepo_dir,
            format="gztar"
        )


def main(dst: Union[str, pathlib.Path]) -> None:
    """main function that downloads and extracts entire dataset"""
    dst = pathlib.Path(dst).expanduser().resolve()
    if not dst.is_dir():
        raise NotADirectoryError(
            f"Value for 'dst' argument not recognized as a directory: {dst}"
        )
    bfsongrepo_dir = dst / 'bfsongrepo'
    if bfsongrepo_dir.exists():
        warnings.warn(
            f"Directory already exists: {bfsongrepo_dir}\n"
            "Will download and write over any existing files. Press Ctrl-C to stop."
        )

    try:
        bfsongrepo_dir.mkdir(exist_ok=True)
    except PermissionError as e:
        raise PermissionError(
            f"Unable to create directory in 'dst': {dst}\n"
            "Please try running with 'sudo' on Unix systems or as Administrator on Windows systems.\n"
            "If that fails, please download files for tutorial manually from the 'download' links in tutorial page."
        ) from e

    print(
        f'Downloading Bengalese Finch Song Repository to: {bfsongrepo_dir}'
    )

    download_dataset(DATA_TO_DOWNLOAD, bfsongrepo_dir)
    extract_tars(bfsongrepo_dir)


def get_parser() -> argparse.ArgumentParser:
    """get ArgumentParser used to parse command-line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dst',
        default='.',
        help=("Destination where dataset should be downloaded. "
              "Default is '.', i.e., current working directory "
              "from which this script is run.'")
    )
    return parser


parser = get_parser()
args = parser.parse_args()
main(dst=args.dst)
