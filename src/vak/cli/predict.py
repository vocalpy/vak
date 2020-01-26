import os


def predict(predict_vds_path,
            train_vds_path,
            checkpoint_path,
            networks,
            spect_scaler_path=None,
            ):
    """make predictions with one trained model

    Parameters
    ----------
    predict_vds_path : str
        path to saved Dataset that contains data for which annotations
        should be predicted.
    train_vds_path : str
        path to Dataset that represents training data.
        To fetch labelmap used during training, to map labels used
        in annotation to a series of consecutive integers that become
        outputs of the neural network. Used here to convert
        back to labels used in annotation.
    checkpoint_path : str
        path to directory with checkpoint files saved by Tensorflow, to reload model
    networks : namedtuple
        where each field is the Config tuple for a neural network and the name
        of that field is the name of the class that represents the network.
    spect_scaler_path : str
        path to a saved SpectScaler object used to normalize spectrograms.
        If spectrograms were normalized and this is not provided, will give
        incorrect results.

    Returns
    -------
    None
    """
    if not os.path.isdir(checkpoint_path):
        raise FileNotFoundError('directory {}, specified as '
                                'checkpoint_path, is not found.'
                                .format(checkpoint_path))

    if spect_scaler_path:
        if not os.path.isfile(spect_scaler_path):
            raise FileNotFoundError(
                f'file for spect_scaler not found at path:\n{spect_scaler_path}'
            )

    if type(predict_vds_path) is str:
        predict_vds_path = [predict_vds_path]
    elif type(predict_vds_path) is list:
        pass
    else:
        raise TypeError(
            'predict_vds_path should be a string path or list '
            f'of string paths, but type was {type(predict_vds_path)}'
        )

    train_vds = Dataset.load(json_fname=train_vds_path)
    if train_vds.are_spects_loaded() is False:
        train_vds = train_vds.load_spects()
    labelmap = train_vds.labelmap
    del train_vds
