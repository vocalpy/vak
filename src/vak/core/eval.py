from collections import OrderedDict
from datetime import datetime
import json

import joblib
import pandas as pd
import torch.utils.data

from .. import models
from .. import transforms
from ..datasets.vocal_dataset import VocalDataset
from ..logging import log_or_print


def eval(csv_path,
         model_config_map,
         checkpoint_path,
         labelmap_path,
         output_dir,
         window_size,
         num_workers,
         split='test',
         spect_scaler_path=None,
         spect_key='s',
         timebins_key='t',
         device=None,
         logger=None):
    """evaluate a trained model

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
    spect_key : str
        key for accessing spectrogram in files. Default is 's'.
    timebins_key : str
        key for accessing vector of time bins in files. Default is 't'.
    device : str
        Device on which to work with model + data.
        Defaults to 'cuda' if torch.cuda.is_available is True.

    Other Parameters
    ----------------
    logger : logging.Logger
        instance created by vak.logging.get_logger. Default is None.

    Returns
    -------
    None
    """
    # ---- get time for .csv file --------------------------------------------------------------------------
    timenow = datetime.now().strftime('%y%m%d_%H%M%S')

    # ---------------- load data for evaluation ------------------------------------------------------------------------
    if spect_scaler_path:
        log_or_print(
            f'loading spect scaler from path: {spect_scaler_path}',
            logger=logger, level='info'
        )
        spect_standardizer = joblib.load(spect_scaler_path)
    else:
        log_or_print(
            f'not using a spect scaler',
            logger=logger, level='info',
        )
        spect_standardizer = None

    logger.info(
        f'loading labelmap from path: {spect_scaler_path}'
    )
    with labelmap_path.open('r') as f:
        labelmap = json.load(f)

    item_transform = transforms.get_defaults('eval',
                                             spect_standardizer,
                                             window_size=window_size,
                                             return_padding_mask=True,
                                             )
    log_or_print(
        f'creating dataset for evaluation from: {csv_path}',
        logger=logger, level='info',
    )
    val_dataset = VocalDataset.from_csv(csv_path=csv_path,
                                        split=split,
                                        labelmap=labelmap,
                                        spect_key=spect_key,
                                        timebins_key=timebins_key,
                                        item_transform=item_transform,
                                        )
    val_data = torch.utils.data.DataLoader(dataset=val_dataset,
                                           shuffle=False,
                                           # batch size 1 because each spectrogram reshaped into a batch of windows
                                           batch_size=1,
                                           num_workers=num_workers)

    # ---------------- do the actual evaluating ------------------------------------------------------------------------
    input_shape = val_dataset.shape
    # if dataset returns spectrogram reshaped into windows,
    # throw out the window dimension; just want to tell network (channels, height, width) shape
    if len(input_shape) == 4:
        input_shape = input_shape[1:]

    models_map = models.from_model_config_map(
        model_config_map,
        num_classes=len(labelmap),
        input_shape=input_shape
    )

    for model_name, model in models_map.items():
        logger.info(
            f'running evaluation for model: {model_name}'
        )
        model.load(checkpoint_path)
        metric_vals = model.evaluate(eval_data=val_data,
                                     device=device)
        # create a "DataFrame" with just one row which we will save as a csv;
        # the idea is to be able to concatenate csvs from multiple runs of eval
        row = OrderedDict(
            [
                ('model_name', model_name),
                ('checkpoint_path', checkpoint_path),
                ('labelmap_path', labelmap_path),
                ('spect_scaler_path', spect_scaler_path),
                ('csv_path', csv_path),
            ]
        )
        # order metrics by name to be extra sure they will be consistent across runs
        row.update(
            sorted([(k, v) for k, v in metric_vals.items() if k.startswith('avg_')])
        )

        # pass index into dataframe, needed when using all scalar values (a single row)
        # throw away index below when saving to avoid extra column
        eval_df = pd.DataFrame(row, index=[0])
        eval_csv_path = output_dir.joinpath(
            f'eval_{model_name}_{timenow}.csv'
        )
        logger.info(
            f'saving csv with evaluation metrics at: {eval_csv_path}'
        )
        eval_df.to_csv(eval_csv_path, index=False)  # index is False to avoid having "Unnamed: 0" column when loading
