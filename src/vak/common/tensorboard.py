"""functions dealing with ``tensorboard``"""
from pathlib import Path

import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def get_summary_writer(log_dir, filename_suffix):
    """get an instance of ``tensorboard.SummaryWriter``,
    to use with a vak.Model during training

    Parameters
    ----------
    log_dir : str
        directory where event file will be written
    filename_suffix : str
        suffix added to events file name

    Returns
    -------
    summary_writer : torch.utils.tensorboard.SummaryWriter

    Examples
    --------
    >>> summary_writer = vak.summary_writer.get_summary_writer(log_dir='./experiments')
    >>> tweety_net_model.summary_writer = summary_writer  # set attribute equal to instance we just made
    >>> tweety_net_model.train()  # now events during training will be logged with that summary writer
    """
    return SummaryWriter(log_dir=log_dir, filename_suffix=filename_suffix)


DEFAULT_SIZE_GUIDANCE = {
    "compressedHistograms": 1,
    "images": 1,
    "scalars": 0,  # 0 means load all
    "histograms": 1,
}


def events2df(events_path, size_guidance=None, drop_wall_time=True):
    """convert ``tensorboard`` "events" log file to pandas DataFrame

    events files are created by SummaryWriter from PyTorch or Tensorflow.

    Parameters
    ----------
    events_path : str, Path
        path to either a log directory or a specific events file
        saved by a SummaryWriter in a log directory.
        By default, ``vak`` saves logs in a directory with the model name
        inside a ``results`` directory generated at the start of training.
    size_guidance: dict
        Argument passed to the ``EventAccumlator`` class from
        ``tensorboard`` that is used to load the events file.
        Information on how much data the EventAccumulator should
        store in memory. Dict that maps a `tagType` string
        to an integer representing the number of items to keep per tag
        for items of that `tagType`. If the size is 0,
        all events are stored. Default is None, in which case
        ``vak.tensorboard.DEFAULT_SIZE_GUIDANCE`` is used.
        For more information see
        https://github.com/tensorflow/tensorboard/blob/master/tensorboard/backend/event_processing/event_accumulator.py
    drop_wall_time : bool
        if True, drop wall times logged in events file. Default is True.

    Returns
    -------
    df : pandas.Dataframe
        with index 'step' and all Scalars from the events file

    Examples
    --------
    >>> events_path = 'tweetynet/results_210322_103904/train_dur_6s/replicate_2/TweetyNet/'
    >>> events_df = vak.tensorboard.events2df(events_path)
    >>> events_df
              loss/train  avg_acc/val  avg_levenshtein/val  avg_segment_error_rate/val  avg_loss/val
    step
    0       2.479142          NaN                  NaN                         NaN           NaN
    1       2.458833          NaN                  NaN                         NaN           NaN
    2       2.441571          NaN                  NaN                         NaN           NaN
    3       2.402737          NaN                  NaN                         NaN           NaN
    4       2.404369          NaN                  NaN                         NaN           NaN
    ...          ...          ...                  ...                         ...           ...
    996     0.171681          NaN                  NaN                         NaN           NaN
    997     0.100202          NaN                  NaN                         NaN           NaN
    998     0.073055          NaN                  NaN                         NaN           NaN
    999     0.031479          NaN                  NaN                         NaN           NaN
    1000         NaN     0.902475                 42.0                    0.880533      0.310385

    [1001 rows x 5 columns]
    """
    if isinstance(events_path, Path):
        events_path = str(events_path)

    if size_guidance is None:
        size_guidance = DEFAULT_SIZE_GUIDANCE

    ea = EventAccumulator(path=events_path, size_guidance=size_guidance)
    ea.Reload()  # load all data written so far

    scalar_tags = ea.Tags()["scalars"]  # list of tags for values written to scalar
    # make a dataframe for each tag, which we will then concatenate using 'step' as the index
    # so that pandas will fill in with NaNs for any scalars that were not measured on every step
    dfs = {}
    for scalar_tag in scalar_tags:
        dfs[scalar_tag] = pd.DataFrame(
            ea.Scalars(scalar_tag), columns=["wall_time", "step", scalar_tag]
        )
        dfs[scalar_tag] = dfs[scalar_tag].set_index("step")
        if drop_wall_time:
            dfs[scalar_tag].drop("wall_time", axis=1, inplace=True)
    return pd.concat([v for k, v in dfs.items()], axis=1)
