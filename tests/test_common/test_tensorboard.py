import pandas as pd
from torch.utils.tensorboard import SummaryWriter

import vak.common.tensorboard


FILENAME_SUFFIX = "FakeModelName"


def test_summary_writer(tmp_path):
    writer = vak.common.tensorboard.get_summary_writer(
        log_dir=str(tmp_path), filename_suffix=FILENAME_SUFFIX
    )
    assert isinstance(writer, SummaryWriter)


def test_events2df(events_path):
    df = vak.common.tensorboard.events2df(events_path)

    assert isinstance(df, pd.DataFrame)
    assert df.index.name == "step"
