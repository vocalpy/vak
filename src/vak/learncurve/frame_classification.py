"""Function that generates results for a learning curve for frame classification models."""
from __future__ import annotations

import logging
import pathlib

import pandas as pd

from .dirname import replicate_dirname, train_dur_dirname
from ..eval.frame_classification import eval_frame_classification_model
from ..train.frame_classification import train_frame_classification_model
from .. import (
    common,
    datasets,
)
from ..common.converters import expanded_user_path
from ..common.paths import generate_results_dir_name_as_path


logger = logging.getLogger(__name__)


def learning_curve_for_frame_classification_model(
    model_name: str,
    model_config: dict,
    dataset_path: str | pathlib.Path,
    window_size: int,
    batch_size: int,
    num_epochs: int,
    num_workers: int,
    root_results_dir: str | pathlib.Path | None = None,
    results_path: str | pathlib.Path = None,
    post_tfm_kwargs: dict | None =None,
    normalize_spectrograms: bool = True,
    shuffle: bool = True,
    val_step: int | None = None,
    ckpt_step: int | None = None,
    patience: int | None = None,
    device: str | None = None,
) -> None:
    """Generate results for a learning curve,
    where model performance is measured as a
    function of dataset duration in seconds.

    Trains a class of model with a range of dataset durations,
    and then evaluates each trained model
    with a test set that is held constant
    (unlike the training sets).
    Results are saved in a new directory within ``root_results_dir``.

    Parameters
    ----------
    model_name : str
        Model name, must be one of vak.models.MODEL_NAMES.
    model_config : dict
        Model configuration in a ``dict``,
        as loaded from a .toml file,
        and used by the model method ``from_config``.
    dataset_path : str
        path to where dataset was saved as a csv.
    window_size : int
        size of windows taken from spectrograms, in number of time bins,
        shown to neural networks
    batch_size : int
        number of samples per batch presented to models during training.
    num_epochs : int
        number of training epochs. One epoch = one iteration through the entire
        training set.
    num_workers : int
        Number of processes to use for parallel loading of data.
        Argument to torch.DataLoader.
    root_results_dir : str, pathlib.Path
        Root directory in which a new directory will be created where results will be saved.
    results_path : str, pathlib.Path
        Directory where results will be saved. If specified, this parameter overrides root_results_dir.
    previous_run_path : str, Path
        path to directory containing dataset .csv files
        that represent subsets of training set, created by
        a previous run of ``vak.core.learncurve.learning_curve``.
        Typically directory will have a name like ``results_{timestamp}``
        and the actual .csv splits will be in sub-directories with names
        corresponding to the training set duration
    post_tfm_kwargs : dict
        Keyword arguments to post-processing transform.
        If None, then no additional clean-up is applied
        when transforming labeled timebins to segments,
        the default behavior. The transform used is
        ``vak.transforms.frame_labels.ToSegmentsWithPostProcessing`.
        Valid keyword argument names are 'majority_vote'
        and 'min_segment_dur', and should be appropriate
        values for those arguments: Boolean for ``majority_vote``,
        a float value for ``min_segment_dur``.
        See the docstring of the transform for more details on
        these arguments and how they work.
    device : str
        Device on which to work with model + data.
        Default is None. If None, then a device will be selected with vak.device.get_default.
        That function defaults to 'cuda' if torch.cuda.is_available is True.
    shuffle: bool
        if True, shuffle training data before each epoch. Default is True.
    normalize_spectrograms : bool
        if True, use spect.utils.data.SpectScaler to normalize the spectrograms.
        Normalization is done by subtracting off the mean for each frequency bin
        of the training set and then dividing by the std for that frequency bin.
        This same normalization is then applied to validation + test data.
    val_step : int
        Step on which to estimate accuracy using validation set.
        If val_step is n, then validation is carried out every time
        the global step / n is a whole number, i.e., when val_step modulo the global step is 0.
        Default is None, in which case no validation is done.
    ckpt_step : int
        Step on which to save to checkpoint file.
        If ckpt_step is n, then a checkpoint is saved every time
        the global step / n is a whole number, i.e., when ckpt_step modulo the global step is 0.
        Default is None, in which case checkpoint is only saved at the last epoch.
    patience : int
        number of validation steps to wait without performance on the
        validation set improving before stopping the training.
        Default is None, in which case training only stops after the specified number of epochs.
    """
    # ---------------- pre-conditions ----------------------------------------------------------------------------------
    dataset_path = expanded_user_path(dataset_path)
    if not dataset_path.exists() or not dataset_path.is_dir():
        raise NotADirectoryError(
            f"`dataset_path` not found or not recognized as a directory: {dataset_path}"
        )

    logger.info(
        f"Loading dataset from path: {dataset_path}",
    )
    metadata = datasets.frame_classification.Metadata.from_dataset_path(dataset_path)
    dataset_csv_path = dataset_path / metadata.dataset_csv_filename
    dataset_df = pd.read_csv(dataset_csv_path)

    if val_step and not dataset_df["split"].str.contains("val").any():
        raise ValueError(
            f"val_step set to {val_step} but dataset does not contain a validation set; "
            f"please run `vak prep` with a config.toml file that specifies a duration for the validation set."
        )

    # ---- set up directory to save output -----------------------------------------------------------------------------
    if results_path:
        results_path = expanded_user_path(results_path)
        if not results_path.is_dir():
            raise NotADirectoryError(
                f"results_path not recognized as a directory: {results_path}"
            )
    else:
        results_path = generate_results_dir_name_as_path(root_results_dir)
        results_path.mkdir()

    logger.info(f"Saving results to: {results_path}")

    frame_dur = metadata.frame_dur
    logger.info(
        f"Duration of a frame in dataset, in seconds: {frame_dur}",
    )

    # ---- get training set subsets ------------------------------------------------------------------------------------
    dataset_df = dataset_df[
        (dataset_df.train_dur.notna()) & (dataset_df.replicate_num.notna())
    ]
    train_durs = sorted(dataset_df['train_dur'].unique())
    replicate_nums = [int(replicate_num)
                      for replicate_num in sorted(dataset_df['replicate_num'].unique())]
    to_do = []
    for train_dur in train_durs:
        for replicate_num in replicate_nums:
            to_do.append((train_dur, replicate_num))

    # ---- main loop that creates "learning curve" ---------------------------------------------------------------------
    logger.info(f"Starting training for learning curve.")
    for train_dur, replicate_num in to_do:
        logger.info(
            f"Training model with training set of size: {train_dur}s, replicate number {replicate_num}.",
        )
        results_path_this_train_dur = results_path / train_dur_dirname(train_dur)
        if not results_path_this_train_dur.exists():
            results_path_this_train_dur.mkdir()

        results_path_this_replicate = results_path_this_train_dur / replicate_dirname(replicate_num)
        results_path_this_replicate.mkdir()

        logger.info(
            f"Saving results to: {results_path_this_replicate}",
        )

        # `split` lets us use correct subset ("split") of training set for this duration / replicate
        split = common.learncurve.get_train_dur_replicate_split_name(
            train_dur, replicate_num
        )

        train_frame_classification_model(
            model_name,
            model_config,
            dataset_path,
            window_size,
            batch_size,
            num_epochs,
            num_workers,
            results_path=results_path_this_replicate,
            normalize_spectrograms=normalize_spectrograms,
            shuffle=shuffle,
            val_step=val_step,
            ckpt_step=ckpt_step,
            patience=patience,
            device=device,
            split=split,
        )

        logger.info(
            f"Evaluating model from replicate {replicate_num} "
        )
        results_model_root = (
            results_path_this_replicate.joinpath(model_name)
        )
        ckpt_root = results_model_root.joinpath("checkpoints")
        ckpt_paths = sorted(ckpt_root.glob("*.pt"))
        if any(["max-val-acc" in str(ckpt_path) for ckpt_path in ckpt_paths]):
            ckpt_paths = [
                ckpt_path
                for ckpt_path in ckpt_paths
                if "max-val-acc" in str(ckpt_path)
            ]
            if len(ckpt_paths) != 1:
                raise ValueError(
                    f"did not find a single max-val-acc checkpoint path, instead found:\n{ckpt_paths}"
                )
            ckpt_path = ckpt_paths[0]
        else:
            if len(ckpt_paths) != 1:
                raise ValueError(
                    f"did not find a single checkpoint path, instead found:\n{ckpt_paths}"
                )
            ckpt_path = ckpt_paths[0]
        logger.info(
            f"Using checkpoint: {ckpt_path}"
        )
        labelmap_path = results_path_this_replicate.joinpath(
            "labelmap.json"
        )
        logger.info(
            f"Using labelmap: {labelmap_path}"
        )
        if normalize_spectrograms:
            spect_scaler_path = (
                results_path_this_replicate.joinpath(
                    "StandardizeSpect"
                )
            )
            logger.info(
                f"Using spect scaler to normalize: {spect_scaler_path}",
            )
        else:
            spect_scaler_path = None

        eval_frame_classification_model(
            model_name,
            model_config,
            dataset_path,
            checkpoint_path=ckpt_path,
            labelmap_path=labelmap_path,
            output_dir=results_path_this_replicate,
            window_size=window_size,
            num_workers=num_workers,
            split="test",
            spect_scaler_path=spect_scaler_path,
            post_tfm_kwargs=post_tfm_kwargs,
            device=device,
        )

    # ---- make a csv for analysis -------------------------------------------------------------------------------------
    # use one of the eval csvs just to get columns, that we use below to re-order
    eval_csv_paths = sorted(results_path.glob("**/eval*.csv"))
    eval_df_0 = pd.read_csv(eval_csv_paths[0])
    eval_columns = eval_df_0.columns.tolist()

    eval_dfs = []
    for train_dur, replicate_num in to_do:
        results_path_this_train_dur = results_path / train_dur_dirname(train_dur)
        results_path_this_replicate = results_path_this_train_dur / replicate_dirname(replicate_num)
        eval_csv_path = sorted(results_path_this_replicate.glob('eval*.csv'))
        if not len(eval_csv_path) == 1:
            raise ValueError(
                "Did not find exactly one eval results csv file in replicate directory after running learncurve. "
                f"Directory is: {results_path_this_replicate}."
                f"Result of globbing directory for eval csv file: {eval_csv_path}"
            )
        eval_csv_path = eval_csv_path[0]
        eval_df = pd.read_csv(eval_csv_path)
        eval_df["train_dur"] = train_dur
        eval_df["replicate_num"] = replicate_num
        eval_dfs.append(eval_df)

    all_eval_df = pd.concat(eval_dfs)
    all_eval_columns = ["train_dur", "replicate_num", *eval_columns]
    all_eval_df = all_eval_df[all_eval_columns]
    all_eval_df.sort_values(by=["train_dur", "replicate_num"])
    learncurve_csv_path = results_path.joinpath("learning_curve.csv")
    all_eval_df.to_csv(
        learncurve_csv_path, index=False
    )  # index=False to avoid adding "Unnamed: 0" column
