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


@define
class SplitsMetadata:
    """Class that represents metadata about dataset splits
    in the BioSoundSegBench dataset, loaded from a json file"""

    splits_csv_path: pathlib.Path
    sample_id_vector_paths: SampleIDVectorPaths
    inds_in_sample_vector_paths: IndsInSampleVectorPaths

    @classmethod
    def from_paths(cls, json_path, dataset_path):
        json_path = pathlib.Path(json_path)
        with json_path.open("r") as fp:
            splits_json = json.load(fp)

        dataset_path = pathlib.Path(dataset_path)
        if not dataset_path.exists() or not dataset_path.is_dir():
            raise NotADirectoryError(
                f"`dataset_path` not found or not a directory: {dataset_path}"
            )

        splits_csv_path = pathlib.Path(
            dataset_path / splits_json["splits_csv_path"]
        )
        if not splits_csv_path.exists():
            raise FileNotFoundError(
                f"`splits_csv_path` not found: {splits_csv_path}"
            )

        sample_id_vector_paths = {
            split: dataset_path / path
            for split, path in splits_json["sample_id_vec_path"].items()
        }
        for split, vec_path in sample_id_vector_paths.items():
            if not vec_path.exists():
                raise FileNotFoundError(
                    f"`sample_id_vector_path` for split '{split}' not found: {vec_path}"
                )
        sample_id_vector_paths = SampleIDVectorPaths(**sample_id_vector_paths)

        inds_in_sample_vector_paths = {
            split: dataset_path / path
            for split, path in splits_json["inds_in_sample_vec_path"].items()
        }
        for split, vec_path in inds_in_sample_vector_paths.items():
            if not vec_path.exists():
                raise FileNotFoundError(
                    f"`inds_in_sample_vec_path` for split '{split}' not found: {vec_path}"
                )
        inds_in_sample_vector_paths = IndsInSampleVectorPaths(
            **inds_in_sample_vector_paths
        )

        return cls(
            splits_csv_path,
            sample_id_vector_paths,
            inds_in_sample_vector_paths,
        )


@define
class TrainingReplicateMetadata:
    """Class representing metadata for a
    pre-defined training replicate
    in the BioSoundSegBench dataset.
    """

    biosound_group: str
    id: str | None
    frame_dur: float
    unit: str
    data_source: str | None
    train_dur: float
    replicate_num: int


def metadata_from_splits_json_path(
    splits_json_path: pathlib.Path, datset_path: pathlib.Path
) -> TrainingReplicateMetadata:
    name = splits_json_path.name
    try:
        (
            biosound_group,
            unit,
            id_,
            frame_dur_1st_half,
            frame_dur_2nd_half,
            data_source,
            train_dur_1st_half,
            train_dur_2nd_half,
            replicate_num,
            _,
            _,
        ) = name.split(".")
    # Human-Speech doesn't have ID or data source in filename
    # so it will raise a ValueError
    except ValueError:
        name = splits_json_path.name
        (
            biosound_group,
            unit,
            frame_dur_1st_half,
            frame_dur_2nd_half,
            train_dur_1st_half,
            train_dur_2nd_half,
            replicate_num,
            _,
            _,
        ) = name.split(".")
        id_ = None
        data_source = None
    if id_ is not None:
        id_ = id_.split("-")[-1]
    frame_dur = float(
        frame_dur_1st_half.split("-")[-1]
        + "."
        + frame_dur_2nd_half.split("-")[0]
    )
    train_dur = float(
        train_dur_1st_half.split("-")[-1]
        + "."
        + train_dur_2nd_half.split("-")[0]
    )
    replicate_num = int(replicate_num.split("-")[-1])
    return TrainingReplicateMetadata(
        biosound_group,
        id_,
        frame_dur,
        unit,
        data_source,
        train_dur,
        replicate_num,
    )