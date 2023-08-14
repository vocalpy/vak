"""Function that generates new inferences from trained models in the frame classification family."""
from __future__ import annotations

import logging
import os
import pathlib

import pytorch_lightning as lightning
import torch.utils.data

from .. import datasets, models, transforms
from ..common import validators
from ..common.device import get_default as get_default_device
from ..datasets.parametric_umap import ParametricUMAPDataset

logger = logging.getLogger(__name__)


def predict_with_parametric_umap_model(
    model_name: str,
    model_config: dict,
    dataset_path,
    checkpoint_path,
    num_workers=2,
    transform_params: dict | None = None,
    dataset_params: dict | None = None,
    timebins_key="t",
    device=None,
    output_dir=None,
):
    """Make predictions on a dataset with a trained model.

     Parameters
     ----------
    model_name : str
        Model name, must be one of vak.models.registry.MODEL_NAMES.
    model_config : dict
        Model configuration in a ``dict``,
        as loaded from a .toml file,
        and used by the model method ``from_config``.
     dataset_path : str
         Path to dataset, e.g., a csv file generated by running ``vak prep``.
     checkpoint_path : str
         path to directory with checkpoint files saved by Torch, to reload model
     num_workers : int
         Number of processes to use for parallel loading of data.
         Argument to torch.DataLoader. Default is 2.
    transform_params: dict, optional
        Parameters for data transform.
        Passed as keyword arguments.
        Optional, default is None.
    dataset_params: dict, optional
        Parameters for dataset.
        Passed as keyword arguments.
        Optional, default is None.
     timebins_key : str
         key for accessing vector of time bins in files. Default is 't'.
     device : str
         Device on which to work with model + data.
         Defaults to 'cuda' if torch.cuda.is_available is True.
     annot_csv_filename : str
         name of .csv file containing predicted annotations.
         Default is None, in which case the name of the dataset .csv
         is used, with '.annot.csv' appended to it.
     output_dir : str, Path
         path to location where .csv containing predicted annotation
         should be saved. Defaults to current working directory.
    """
    for path, path_name in zip(
        (checkpoint_path,),
        ("checkpoint_path",),
    ):
        if path is not None:
            if not validators.is_a_file(path):
                raise FileNotFoundError(
                    f"value for ``{path_name}`` not recognized as a file: {path}"
                )

    dataset_path = pathlib.Path(dataset_path)
    if not dataset_path.exists() or not dataset_path.is_dir():
        raise NotADirectoryError(
            f"`dataset_path` not found or not recognized as a directory: {dataset_path}"
        )
    logger.info(
        f"Loading metadata from dataset path: {dataset_path}",
    )
    metadata = datasets.frame_classification.Metadata.from_dataset_path(
        dataset_path
    )

    if output_dir is None:
        output_dir = pathlib.Path(os.getcwd())
    else:
        output_dir = pathlib.Path(output_dir)

    if not output_dir.is_dir():
        raise NotADirectoryError(
            f"value specified for output_dir is not recognized as a directory: {output_dir}"
        )

    if device is None:
        device = get_default_device()

    # ---------------- load data for prediction ------------------------------------------------------------------------
    if transform_params is None:
        transform_params = {}
    if "padding" not in transform_params and model_name == "ConvEncoderUMAP":
        padding = models.convencoder_umap.get_default_padding(metadata.shape)
        transform_params["padding"] = padding

    item_transform = transforms.defaults.get_default_transform(
        model_name, "predict", transform_params
    )

    dataset_csv_path = dataset_path / metadata.dataset_csv_filename
    logger.info(
        f"loading dataset to predict from csv path: {dataset_csv_path}"
    )

    if dataset_params is None:
        dataset_params = {}
    pred_dataset = ParametricUMAPDataset.from_dataset_path(
        dataset_path=dataset_path,
        split="predict",
        transform=item_transform,
        **dataset_params,
    )

    pred_loader = torch.utils.data.DataLoader(
        dataset=pred_dataset,
        shuffle=False,
        # batch size 1 because each spectrogram reshaped into a batch of windows
        batch_size=1,
        num_workers=num_workers,
    )

    # ---------------- do the actual predicting + converting to annotations --------------------------------------------
    input_shape = pred_dataset.shape
    # if dataset returns spectrogram reshaped into windows,
    # throw out the window dimension; just want to tell network (channels, height, width) shape
    if len(input_shape) == 4:
        input_shape = input_shape[1:]
    logger.info(
        f"Shape of input to networks used for predictions: {input_shape}"
    )

    logger.info(f"instantiating model from config:/n{model_name}")

    model = models.get(
        model_name,
        model_config,
        input_shape=input_shape,
    )

    # ---------------- do the actual predicting --------------------------------------------------------------------
    logger.info(
        f"loading checkpoint for {model_name} from path: {checkpoint_path}"
    )
    model.load_state_dict_from_path(checkpoint_path)

    if device == "cuda":
        accelerator = "gpu"
    else:
        accelerator = None
    trainer_logger = lightning.loggers.TensorBoardLogger(save_dir=output_dir)
    trainer = lightning.Trainer(accelerator=accelerator, logger=trainer_logger)

    logger.info(f"running predict method of {model_name}")
    results = trainer.predict(model, pred_loader)  # noqa : F841

    # eval_df = pd.DataFrame(row, index=[0])
    # eval_csv_path = output_dir.joinpath(f"eval_{model_name}_{timenow}.csv")
    # logger.info(f"saving csv with evaluation metrics at: {eval_csv_path}")
    # eval_df.to_csv(
    #     eval_csv_path, index=False
    # )  # index is False to avoid having "Unnamed: 0" column when loading
