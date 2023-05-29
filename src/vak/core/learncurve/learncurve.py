from __future__ import annotations

import logging
import pathlib
import re

import numpy as np
import pandas as pd

from ..eval import eval
from ..prep.prep_helper import validate_and_get_timebin_dur
from ..train import train
from ... import (
    datasets,
)
from ...converters import expanded_user_path
from ...paths import generate_results_dir_name_as_path

logger = logging.getLogger(__name__)


def train_dur_dirname(train_dur: int) -> str:
    """Returns name of directory for all replicates
    trained with a training set of a specified duration,
    ``f"train_dur_{train_dur}s"``.
    """
    return f"train_dur_{train_dur}s"


def replicate_dirname(replicate_num: int) -> str:
    """Returns name of directory for a replicate,
    ``f"replicate_{replicate_num}``.
    """
    return f"replicate_{replicate_num}"


def learning_curve(
    model_name: str,
    model_config: dict,
    dataset_path: str | pathlib.Path,
    labelset: set,
    window_size: int,
    batch_size: int,
    num_epochs: int,
    num_workers: int,
    root_results_dir: str | pathlib.Path | None = None,
    results_path: str | pathlib.Path = None,
    post_tfm_kwargs: dict | None =None,
    spect_key:str = "s",
    timebins_key: str = "t",
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
    train_set_durs : list
        of int, durations in seconds of subsets taken from training data
        to create a learning curve, e.g. [5, -10, 15, 20].
    num_replicates : int
        number of times to replicate training for each training set duration
        to better estimate metrics for a training set of that size.
        Each replicate uses a different randomly drawn subset of the training
        data (but of the same duration).
    dataset_path : str
        path to where dataset was saved as a csv.
    labelset : set
        of str or int, the set of labels that correspond to annotated segments
        that a network should learn to segment and classify. Note that if there
        are segments that are not annotated, e.g. silent gaps between songbird
        syllables, then `vak` will assign a dummy label to those segments
        -- you don't have to give them a label here.
    window_size : int
        size of windows taken from spectrograms, in number of time bins,
        shonw to neural networks
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
        ``vak.transforms.labeled_timebins.ToSegmentsWithPostProcessing`.
        Valid keyword argument names are 'majority_vote'
        and 'min_segment_dur', and should be appropriate
        values for those arguments: Boolean for ``majority_vote``,
        a float value for ``min_segment_dur``.
        See the docstring of the transform for more details on
        these arguments and how they work.
    spect_key : str
        key for accessing spectrogram in files. Default is 's'.
    timebins_key : str
        key for accessing vector of time bins in files. Default is 't'.
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
    metadata = datasets.metadata.Metadata.from_dataset_path(dataset_path)
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

    timebin_dur = validate_and_get_timebin_dur(dataset_df)
    logger.info(
        f"Size of each timebin in spectrogram, in seconds: {timebin_dur}",
    )

    # ---- get training set subsets ------------------------------------------------------------------------------------

    dataset_learncurve_dir = dataset_path / 'learncurve'
    splits_path = dataset_learncurve_dir / 'learncurve-splits-metadata.csv'
    logger.info(
        f"Loading learncurve splits from: {splits_path}",
    )
    splits_df = pd.read_csv(splits_path)

    # ---- main loop that creates "learning curve" ---------------------------------------------------------------------
    logger.info(f"Starting training for learning curve.")
    # We iterate over the dataframe here, instead of e.g. filtering by 'train_dur' / 'replicate_num',
    # because we want to train models using exactly the splits were generated by `prep` and saved in the csv.
    # Technically this is inefficient, but we are not doing real data processing here, just using pandas
    # to iterate over a tabular data structure in our main loop. Slightly less annoying then, say, parsing json
    for splits_df_row in splits_df.itertuples():
        train_dur, replicate_num = splits_df_row.train_dur, splits_df_row.replicate_num
        logger.info(
            f"Training replicates for training set of size: {train_dur}s",
        )
        results_path_this_train_dur = results_path / train_dur_dirname(train_dur)
        if not results_path_this_train_dur.exists():
            results_path_this_train_dur.mkdir()

        results_path_this_replicate = results_path_this_train_dur / replicate_dirname(replicate_num)
        results_path_this_replicate.mkdir()

        split_csv_path = dataset_learncurve_dir / splits_df_row.split_csv_filename
        logger.info(
            f"Training replicate {replicate_num} "
            f"using dataset from .csv file: {split_csv_path}",
        )
        logger.info(
            f"Saving results to: {results_path_this_replicate}",
        )

        window_dataset_kwargs = {}
        for window_dataset_kwarg in [
            "source_ids",
            "source_inds",
            "window_inds",
        ]:
            vec_filename = getattr(splits_df_row, f'{window_dataset_kwarg}_npy_filename')
            window_dataset_kwargs[window_dataset_kwarg] = np.load(
                dataset_learncurve_dir / vec_filename
                )

        train(
            model_name,
            model_config,
            dataset_path,
            window_size,
            batch_size,
            num_epochs,
            num_workers,
            dataset_csv_path=split_csv_path,
            labelset=labelset,
            results_path=results_path_this_replicate,
            spect_key=spect_key,
            timebins_key=timebins_key,
            normalize_spectrograms=normalize_spectrograms,
            shuffle=shuffle,
            val_step=val_step,
            ckpt_step=ckpt_step,
            patience=patience,
            device=device,
            **window_dataset_kwargs,
        )

        logger.info(
            f"Evaluating model from replicate {replicate_num} "
            f"using dataset from .csv file: {split_csv_path}",
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

        eval(
            model_name,
            model_config,
            results_path_this_replicate,
            checkpoint_path=ckpt_path,
            labelmap_path=labelmap_path,
            output_dir=results_path_this_replicate,
            window_size=window_size,
            num_workers=num_workers,
            split="test",
            spect_scaler_path=spect_scaler_path,
            post_tfm_kwargs=post_tfm_kwargs,
            spect_key=spect_key,
            timebins_key=timebins_key,
            device=device,
        )

    # ---- make a csv for analysis -------------------------------------------------------------------------------------
    reg_exp_num = re.compile(
        r"[-+]?\d*\.\d+|\d+"
    )  # to extract train set dur and replicate num from paths

    eval_csv_paths = sorted(results_path.glob("**/eval*.csv"))
    eval_df_0 = pd.read_csv(eval_csv_paths[0])  # use to just get columns
    eval_columns = eval_df_0.columns.tolist()  # will use below to re-order
    eval_dfs = []
    for eval_csv_path in eval_csv_paths:
        train_set_dur = reg_exp_num.findall(eval_csv_path.parents[1].name)
        if len(train_set_dur) != 1:
            raise ValueError(
                f"unable to determine training set duration from .csv path: {train_set_dur}"
            )
        else:
            train_set_dur = float(train_set_dur[0])
        replicate_num = reg_exp_num.findall(eval_csv_path.parents[0].name)
        if len(replicate_num) != 1:
            raise ValueError(
                f"unable to determine replicate number from .csv path: {train_set_dur}"
            )
        else:
            replicate_num = int(replicate_num[0])
        eval_df = pd.read_csv(eval_csv_path)
        eval_df["train_set_dur"] = train_set_dur
        eval_df["replicate_num"] = replicate_num
        eval_dfs.append(eval_df)
    all_eval_df = pd.concat(eval_dfs)
    all_eval_columns = ["train_set_dur", "replicate_num", *eval_columns]
    all_eval_df = all_eval_df[all_eval_columns]
    all_eval_df.sort_values(by=["train_set_dur", "replicate_num"])
    learncurve_csv_path = results_path.joinpath("learning_curve.csv")
    all_eval_df.to_csv(
        learncurve_csv_path, index=False
    )  # index=False to avoid adding "Unnamed: 0" column
