"""utility functions dealing with tensorboard SummaryWriter"""
from torch.utils.tensorboard import SummaryWriter


def get_summary_writer(log_dir, filename_suffix):
    """get an instance of a SummaryWriter,
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
    return SummaryWriter(
        log_dir=log_dir,
        filename_suffix=filename_suffix
    )
