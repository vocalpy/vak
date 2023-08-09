"""High-level function that evaluates trained models."""
from __future__ import annotations

import logging
import pathlib

from .frame_classification import eval_frame_classification_model
from .parametric_umap import eval_parametric_umap_model
from .. import (
    models,
)
from ..common import validators


logger = logging.getLogger(__name__)


def eval(
    model_name: str,
    model_config: dict,
    dataset_path: str | pathlib.Path,
    checkpoint_path: str | pathlib.Path,
    output_dir: str | pathlib.Path,
    num_workers: int,
    labelmap_path: str | pathlib.Path | None = None,
    batch_size: int | None = None,
    transform_params: dict | None = None,
    dataset_params: dict | None = None,
    split: str = "test",
    spect_scaler_path: str | pathlib.Path = None,
    post_tfm_kwargs: dict | None = None,
    device: str | None = None
) -> None:
    """Evaluate a trained model.

    Parameters
    ----------
    model_name : str
        Model name, must be one of vak.models.registry.MODEL_NAMES.
    model_config : dict
        Model configuration in a ``dict``,
        as loaded from a .toml file,
        and used by the model method ``from_config``.
    dataset_path : str, pathlib.Path
        Path to dataset, e.g., a csv file generated by running ``vak prep``.
    checkpoint_path : str, pathlib.Path
        path to directory with checkpoint files saved by Torch, to reload model
    output_dir : str, pathlib.Path
        Path to location where .csv files with evaluation metrics should be saved.
    num_workers : int
        Number of processes to use for parallel loading of data.
        Argument to torch.DataLoader. Default is 2.
    labelmap_path : str, pathlib.Path, optional
        Path to 'labelmap.json' file.
        Optional, default is None.
    batch_size : int, optional.
        Number of samples per batch fed into model.
        Optional, default is None.
    transform_params: dict, optional
        Parameters for data transform.
        Passed as keyword arguments.
        Optional, default is None.
    dataset_params: dict, optional
        Parameters for dataset.
        Passed as keyword arguments.
        Optional, default is None.
    split : str
        split of dataset on which model should be evaluated.
        One of {'train', 'val', 'test'}. Default is 'test'.
    spect_scaler_path : str, pathlib.Path
        path to a saved SpectScaler object used to normalize spectrograms.
        If spectrograms were normalized and this is not provided, will give
        incorrect results.
        Default is None.
    post_tfm_kwargs : dict
        Keyword arguments to post-processing transform.
        If None, then no additional clean-up is applied
        when transforming labeled timebins to segments,
        the default behavior. The transform used is
        ``vak.transforms.frame_labels.PostProcess`.
        Valid keyword argument names are 'majority_vote'
        and 'min_segment_dur', and should be appropriate
        values for those arguments: Boolean for ``majority_vote``,
        a float value for ``min_segment_dur``.
        See the docstring of the transform for more details on
        these arguments and how they work.
    device : str
        Device on which to work with model + data.
        Defaults to 'cuda' if torch.cuda.is_available is True.

    Notes
    -----
    Note that unlike ``core.predict``, this function
    can modify ``labelmap`` so that metrics like edit distance
    are correctly computed, by converting any string labels
    in ``labelmap`` with multiple characters
    to (mock) single-character labels,
    with ``vak.labels.multi_char_labels_to_single_char``.
    """
    # ---- pre-conditions ----------------------------------------------------------------------------------------------
    for path, path_name in zip(
            (checkpoint_path, labelmap_path, spect_scaler_path),
            ('checkpoint_path', 'labelmap_path', 'spect_scaler_path'),
    ):
        if path is not None:  # because `spect_scaler_path` is optional
            if not validators.is_a_file(path):
                raise FileNotFoundError(
                    f"value for ``{path_name}`` not recognized as a file: {path}"
                )

    dataset_path = pathlib.Path(dataset_path)
    if not dataset_path.exists() or not dataset_path.is_dir():
        raise NotADirectoryError(
            f"`dataset_path` not found or not recognized as a directory: {dataset_path}"
        )

    try:
        model_family = models.registry.MODEL_FAMILY_FROM_NAME[model_name]
    except KeyError as e:
        raise ValueError(
            f"No model family found for the model name specified: {model_name}"
        ) from e
    if model_family == "FrameClassificationModel":
        eval_frame_classification_model(
            model_name=model_name,
            model_config=model_config,
            dataset_path=dataset_path,
            checkpoint_path=checkpoint_path,
            labelmap_path=labelmap_path,
            output_dir=output_dir,
            num_workers=num_workers,
            transform_params=transform_params,
            dataset_params=dataset_params,
            split=split,
            spect_scaler_path=spect_scaler_path,
            device=device,
            post_tfm_kwargs=post_tfm_kwargs,
        )
    elif model_family == "ParametricUMAPModel":
        eval_parametric_umap_model(
            model_name=model_name,
            model_config=model_config,
            dataset_path=dataset_path,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            transform_params=transform_params,
            dataset_params=dataset_params,
            split=split,
            spect_scaler_path=spect_scaler_path,
            device=device,
        )
    else:
        raise ValueError(
            f"Model family not recognized: {model_family}"
        )
