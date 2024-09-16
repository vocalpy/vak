"""Helper functions used with CMACBench dataset."""
from __future__ import annotations

import json
import pathlib

from attrs import define


@define
class SampleIDVectorPaths:
    train: pathlib.Path
    val: pathlib.Path
    test: pathlib.Path


@define
class IndsInSampleVectorPaths:
    train: pathlib.Path
    val: pathlib.Path
    test: pathlib.Path


def _get_metadata_item(metadata: dict, key: str, metadata_path: pathlib.Path):
        try:
            value = metadata[key]
        except KeyError as e:
            raise KeyError(
                f"Metadata not found in `json_path`: {key}.\n"
                f"`metadata_path`: {metadata_path}\n"
                f"Contents of `metadata`:\n{metadata}"
            ) from e
        return value


def _path_absolute_or_relative_to_dataset_path(
        path: str | pathlib.Path, dataset_path: pathlib.Path, path_key: str
) -> pathlib.Path:
    path = pathlib.Path(path)
    dataset_path = pathlib.Path(dataset_path)
    if path.resolve().exists():
        return path.resolve()
    path_relative_to_dataset = dataset_path / path
    if not path_relative_to_dataset.exists():
        raise FileNotFoundError(
            f"Did not find `{path_key}` using either absolute path ({path.resolve()})"
            f"or relative to `dataset_path` ({path_relative_to_dataset})"
        )
    return path_relative_to_dataset


@define
class Metadata:
    """Class that represents metadata about CMACbench datasets, 
    loaded from a json file.
    
    Attributes
    ----------
    splits_csv_path : pathlib.Path
        Path to a csv file that assigns dataset files to splits.
    sample_id_vector_paths : SampleIDVectorPaths
        Dataclass whose attributes are paths to numpy array files 
        containing vectors, one file for each split in the dataset.
        Each element in a vector indicates which sample in a dataset the frame belongs to.
        A sample is one (input, target) pair.
    inds_in_sample_vector_paths : IndsInSampleVectorPaths
        Dataclass whose attributes are paths to numpy array files 
        containing vectors, one file for each split in the dataset.
        Each element in a vector indicates which sample in a dataset the frame belongs to.
        A sample is one (input, target) pair.
    frame_dur : float
        Duration of a frame in the dataset, in seconds.
    labelmap_json_path : pathlib.Path
        Path to a json file containing the labelmap for this dataset.
    raw : dict
        The raw metadata loaded from a json file.
    """

    splits_csv_path: pathlib.Path
    sample_id_vector_paths: SampleIDVectorPaths
    inds_in_sample_vector_paths: IndsInSampleVectorPaths
    frame_dur: float
    labelmap_json_path: pathlib.Path
    raw: dict

    @classmethod
    def from_paths(cls, metadata_path, dataset_path):
        """Classmethod that loads metadata from path to `"metadata.json"` file
        and path to CMACBench dataset root.
        """

        metadata_path = _path_absolute_or_relative_to_dataset_path(metadata_path, dataset_path, "metadata_path")
        with metadata_path.open("r") as fp:
            metadata = json.load(fp)

        dataset_path = pathlib.Path(dataset_path)
        if not dataset_path.exists() or not dataset_path.is_dir():
            raise NotADirectoryError(
                f"`dataset_path` not found or not a directory: {dataset_path}"
            )

        splits_csv_path = _get_metadata_item(metadata, "splits_csv_path", metadata_path)
        splits_csv_path = _path_absolute_or_relative_to_dataset_path(splits_csv_path, dataset_path, "splits_csv_path")

        sample_id_vector_paths = _get_metadata_item(metadata, "sample_id_vec_path", metadata_path)
        sample_id_vector_paths = {
            split: _path_absolute_or_relative_to_dataset_path(
                path, dataset_path, f"`sample_id_vector_path` for split '{split}'"
            )
            for split, path in sample_id_vector_paths.items()
        }
        sample_id_vector_paths = SampleIDVectorPaths(**sample_id_vector_paths)

        inds_in_sample_vector_paths = _get_metadata_item(metadata, "inds_in_sample_vec_path", metadata_path)
        inds_in_sample_vector_paths = {
            split: _path_absolute_or_relative_to_dataset_path(
                path, dataset_path, f"`inds_in_sample_vec_path` for split '{split}'"
            )
            for split, path in inds_in_sample_vector_paths.items()
        }
        inds_in_sample_vector_paths = IndsInSampleVectorPaths(
            **inds_in_sample_vector_paths
        )

        frame_dur = _get_metadata_item(metadata, "frame_dur", metadata_path)
        frame_dur = float(frame_dur)

        labelmap_json_path = _get_metadata_item(metadata, "labelmap_json_path", metadata_path)
        labelmap_json_path = _path_absolute_or_relative_to_dataset_path(labelmap_json_path, dataset_path, "labelmap_json_path")

        return cls(
            splits_csv_path,
            sample_id_vector_paths,
            inds_in_sample_vector_paths,
            frame_dur,
            labelmap_json_path,
            raw=metadata
        )
