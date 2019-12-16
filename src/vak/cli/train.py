from configparser import ConfigParser
import os
import shutil
from datetime import datetime

from .. import core


def train(models,
          csv_path,
          labelset,
          num_epochs,
          config_file,
          batch_size,
          train_unlabeled=True,
          shuffle=True,
          val_error_step=None,
          checkpoint_step=None,
          patience=None,
          save_only_single_checkpoint_file=True,
          normalize_spectrograms=False,
          spect_key='s',
          timebins_key='t',
          root_results_dir=None
          ):
    """train models using training set specified in config.ini file.
    Function called by command-line interface.

    Parameters
    ----------
    models : dict
        where each key is the name of a neural network and the corresponding
        value is the configuration for that network (in a namedtuple or a dict)
    csv_path : str
        path to csv file that represents dataset, created by vak.cli.prep.
    labelset : list
        of str or int, set of labels for syllables. The names for the classes
        that the network is trained to predict.
    num_epochs : int
        number of training epochs. One epoch = one iteration through the entire
        training set.
    config_file : str
        path to config.ini file. Used to rewrite file with options determined by
        this function and needed for other functions (e.g. cli.summary)
    batch_size : int
        number of samples per batch presented to models during training.
    train_unlabeled : bool
        if True, add a class to labelset for unlabeled segments in annotation, and
        train network to predict those unlabeled segments.
    shuffle : bool
        if True, shuffle samples in training set every epoch. Default is True.
    val_error_step : int
        step/epoch at which to estimate accuracy using validation set.
        Default is None, in which case no validation is done.
    checkpoint_step : int
        step/epoch at which to save to checkpoint file.
        Default is None, in which case checkpoint is only saved at the last epoch.
    patience : int
        number of epochs to wait without the error dropping before stopping the
        training. Default is None, in which case training continues for num_epochs
    save_only_single_checkpoint_file : bool
        if True, save only one checkpoint file instead of separate files every time
        we save. Default is True.
    normalize_spectrograms : bool
        if True, use spect.utils.data.SpectScaler to normalize the spectrograms.
        Normalization is done by subtracting off the mean for each frequency bin
        of the training set and then dividing by the std for that frequency bin.
        This same normalization is then applied to validation + test data.
    spect_key : str
        key for accessing spectrogram in files. Default is 's'.
        Used when fitting a SpectScaler to normalize spectrograms.
    timebins_key : str
        key for accessing vector of time bins in files. Default is 't'.
    root_results_dir : str
        path in which to create results directory that will containing files from training

    Returns
    -------
    None

    Saves results in root_results_dir and adds some options to config_file.
    """
    # ---- set up directory to save output -----------------------------------------------------------------------------
    timenow = datetime.now().strftime('%y%m%d_%H%M%S')
    if root_results_dir:
        results_dirname = os.path.join(root_results_dir,
                                       'results_' + timenow)
    else:
        results_dirname = os.path.join('.', 'results_' + timenow)
    os.makedirs(results_dirname)
    # copy config file into results dir now that we've made the dir
    shutil.copy(config_file, results_dirname)

    core.train(models=models,
               csv_path=csv_path,
               labelset=labelset,
               num_epochs=num_epochs,
               batch_size=batch_size,
               shuffle=shuffle,
               train_unlabeled=train_unlabeled,
               val_error_step=val_error_step,
               checkpoint_step=checkpoint_step,
               patience=patience,
               save_only_single_checkpoint_file=save_only_single_checkpoint_file,
               normalize_spectrograms=normalize_spectrograms,
               spect_key=spect_key,
               timebins_key=timebins_key,
               root_results_dir=None)

    # lastly rewrite config file,
    # so that paths where results were saved are automatically in config
    config = ConfigParser()
    config.read(config_file)
    config.set(section='TRAIN',
               option='results_dir_made_by_main_script',
               value=results_dirname)
    with open(config_file, 'w') as config_file_rewrite:
        config.write(config_file_rewrite)
