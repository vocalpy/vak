"""Helper functions that return names of directories
generated during a run of a learning curve."""


def train_dur_dirname(train_dur: float) -> str:
    """Returns name of directory for all replicates
    trained with a training set of a specified duration,
    ``f"train_dur_{train_dur}s"``.
    """
    return f"train_dur_{float(train_dur)}s"


def replicate_dirname(replicate_num: int) -> str:
    """Returns name of directory for a replicate,
    ``f"replicate_{replicate_num}``.
    """
    return f"replicate_{int(replicate_num)}"
