from collections import OrderedDict
from datetime import datetime
import json
import logging

import joblib
import pandas as pd
import torch.utils.data

from .. import (
    files,
    models,
    timebins,
    transforms,
    validators
)
from ..datasets.vocal_dataset import VocalDataset
from ..labels import multi_char_labels_to_single_char


logger = logging.getLogger(__name__)


def eval(
    csv_path,
    model_config_map,
    checkpoint_path,
    labelmap_path,
    output_dir,
    window_size,
    num_workers,
    split="test",
    spect_scaler_path=None,
    post_tfm_kwargs=None,
    spect_key="s",
    timebins_key="t",
    device=None,
):
    """Evaluate a trained model.

    Parameters
    ----------
    csv_path : str, pathlib.Path
        path to where dataset was saved as a csv.
    model_config_map : dict
        where each key-value pair is model name : dict of config parameters
    checkpoint_path : str, pathlib.Path
        path to directory with checkpoint files saved by Torch, to reload model
    output_dir : str, pathlib.Path
        Path to location where .csv files with evaluation metrics should be saved.
    window_size : int
        size of windows taken from spectrograms, in number of time bins,
        shown to neural networks
    labelmap_path : str, pathlib.Path
        path to 'labelmap.json' file.
    models : list
        of model names. e.g., 'models = TweetyNet, GRUNet, ConvNet'
    batch_size : int
        number of samples per batch presented to models during training.
    num_workers : int
        Number of processes to use for parallel loading of data.
        Argument to torch.DataLoader. Default is 2.
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
            (checkpoint_path, csv_path, labelmap_path, spect_scaler_path),
            ('checkpoint_path', 'csv_path', 'labelmap_path', 'spect_scaler_path'),
    ):
        if path is not None:  # because `spect_scaler_path` is optional
            if not validators.is_a_file(path):
                raise FileNotFoundError(
                    f"value for ``{path_name}`` not recognized as a file: {path}"
                )
    if not validators.is_a_directory(output_dir):
        raise NotADirectoryError(
            f'value for ``output_dir`` not recognized as a directory: {output_dir}'
        )

    # ---- get time for .csv file --------------------------------------------------------------------------------------
    timenow = datetime.now().strftime("%y%m%d_%H%M%S")

    # ---------------- load data for evaluation ------------------------------------------------------------------------
    if spect_scaler_path:
        logger.info(f"loading spect scaler from path: {spect_scaler_path}")
        spect_standardizer = joblib.load(spect_scaler_path)
    else:
        logger.info(f"not using a spect scaler")
        spect_standardizer = None

    logger.info(f"loading labelmap from path: {labelmap_path}")
    with labelmap_path.open("r") as f:
        labelmap = json.load(f)

    # replace any multiple character labels in mapping
    # with dummy single-character labels
    # so that we do not affect edit distance computation
    # see https://github.com/NickleDave/vak/issues/373
    labelmap_keys = [lbl for lbl in labelmap.keys() if lbl != 'unlabeled']
    if any([len(label) > 1 for label in labelmap_keys]):  # only re-map if necessary
        # (to minimize chance of knock-on bugs)
        labelmap = multi_char_labels_to_single_char(labelmap)

    item_transform = transforms.get_defaults(
        "eval",
        spect_standardizer,
        window_size=window_size,
        return_padding_mask=True,
    )
    logger.info(f"creating dataset for evaluation from: {csv_path}")
    val_dataset = VocalDataset.from_csv(
        csv_path=csv_path,
        split=split,
        labelmap=labelmap,
        spect_key=spect_key,
        timebins_key=timebins_key,
        item_transform=item_transform,
    )
    val_data = torch.utils.data.DataLoader(
        dataset=val_dataset,
        shuffle=False,
        # batch size 1 because each spectrogram reshaped into a batch of windows
        batch_size=1,
        num_workers=num_workers,
    )

    # ---------------- do the actual evaluating ------------------------------------------------------------------------
    input_shape = val_dataset.shape
    # if dataset returns spectrogram reshaped into windows,
    # throw out the window dimension; just want to tell network (channels, height, width) shape
    if len(input_shape) == 4:
        input_shape = input_shape[1:]

    if post_tfm_kwargs:
        dataset_df = pd.read_csv(csv_path)
        # we use the timebins vector from the first spect path to get timebin dur.
        # this is less careful than calling io.dataframe.validate_and_get_timebin_dur
        # but it's also much faster, and we can assume dataframe was validated when it was made
        spect_dict = files.spect.load(dataset_df['spect_path'].values[0])
        timebin_dur = timebins.timebin_dur_from_vec(spect_dict[timebins_key])

        post_tfm = transforms.labeled_timebins.PostProcess(
            timebin_dur=timebin_dur,
            **post_tfm_kwargs,
        )
    else:
        post_tfm = None

    models_map = models.from_model_config_map(
        model_config_map, num_classes=len(labelmap), input_shape=input_shape, post_tfm=post_tfm
    )

    for model_name, model in models_map.items():
        logger.info(f"running evaluation for model: {model_name}")
        model.load(checkpoint_path, device=device)
        metric_vals = model.evaluate(eval_data=val_data, device=device)
        logger.info(f"Finished evaluating. Logging computing metrics.")
        for metric_name, metric_val in metric_vals.items():
            if metric_name.startswith('avg_'):
                logger.info(
                    f'{metric_name}: {metric_val:0.5f}'
                )
        # create a "DataFrame" with just one row which we will save as a csv;
        # the idea is to be able to concatenate csvs from multiple runs of eval
        row = OrderedDict(
            [
                ("model_name", model_name),
                ("checkpoint_path", checkpoint_path),
                ("labelmap_path", labelmap_path),
                ("spect_scaler_path", spect_scaler_path),
                ("csv_path", csv_path),
            ]
        )
        # order metrics by name to be extra sure they will be consistent across runs
        row.update(
            sorted([(k, v) for k, v in metric_vals.items() if k.startswith("avg_")])
        )

        # pass index into dataframe, needed when using all scalar values (a single row)
        # throw away index below when saving to avoid extra column
        eval_df = pd.DataFrame(row, index=[0])
        eval_csv_path = output_dir.joinpath(f"eval_{model_name}_{timenow}.csv")
        logger.info(f"saving csv with evaluation metrics at: {eval_csv_path}")
        eval_df.to_csv(
            eval_csv_path, index=False
        )  # index is False to avoid having "Unnamed: 0" column when loading
