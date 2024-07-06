"""Helper function that gets instances of classes representing datasets built into :mod:`vak`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Mapping

from .. import common

Dataset = Mapping

if TYPE_CHECKING:
    from ..transforms import FramesStandardizer


def get(
    dataset_config: dict,
    split: Literal["predict", "test", "train", "val"],
    frames_standardizer: FramesStandardizer | None = None,
) -> Dataset:
    """Get an instance of a dataset class from :mod:`vak.datasets`.

    Parameters
    ----------
    dataset_config: dict
        Dataset configuration in a :class:`dict`.
        Can be obtained by calling :meth:`vak.config.DatasetConfig.asdict`.
    split : str
        Name of split to use.
        One of {"predict", "test", "train", "val"}.

    Returns
    -------
    dataset : class
        An instance of a class from :mod:`vak.datasets`,
        e.g. :class:`vak.datasets.BioSoundSegBench`.
    """
    from . import DATASETS  # avoid circular import

    if "name" not in dataset_config:
        raise KeyError(
            "A name is required to get a dataset, but "
            "`vak.datasets.get` received a `dataset_config` "
            f'without a "name":\n{dataset_config}'
        )
    if split not in common.constants.VALID_SPLITS:
        raise ValueError(
            f"Invalid value for `split`: {split}.\n"
            f"Valid splits are: {common.constants.VALID_SPLITS}"
        )

    dataset_name = dataset_config["name"]
    try:
        dataset_class = DATASETS[dataset_name]
    except KeyError as e:
        raise ValueError(
            f"Invalid dataset name: {dataset_name}\n."
            f"Built-in dataset names are: {DATASETS.keys()}"
        ) from e
    if frames_standardizer is not None:
        dataset_config["params"]["frames_standardizer"] = frames_standardizer
    dataset = dataset_class(
        dataset_path=dataset_config["path"],
        splits_path=dataset_config["splits_path"],
        split=split,
        **dataset_config["params"],
    )
    return dataset
