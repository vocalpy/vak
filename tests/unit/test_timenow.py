import time
from datetime import datetime

import vak.constants
import vak.timenow


def test_timenow():
    before_timestamp = time.mktime(datetime.now().timetuple())
    timenow_str = vak.timenow.get_timenow_as_str()
    assert len(timenow_str) == len(vak.constants.STRFTIME_TIMESTAMP)
    timenow_str_as_timestamp = time.mktime(
        datetime.strptime(timenow_str, vak.constants.STRFTIME_TIMESTAMP).timetuple()
    )
    after_timestamp = time.mktime(datetime.now().timetuple())
    assert before_timestamp <= timenow_str_as_timestamp <= after_timestamp
