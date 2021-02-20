from datetime import datetime
import os
import time

import pytest

import vak.constants
import vak.paths


def results_path_matches_expected(results_path,
                                  root_results_dir,
                                  before_timestamp,
                                  after_timestamp):
    assert results_path.parents == root_results_dir
    assert results_path.name.startswith(vak.constants.RESULTS_DIR_PREFIX)
    results_dirname_parts = results_path.name.split(sep='_')
    assert len(results_dirname_parts) == 2
    timenow_str = results_dirname_parts[1]
    assert len(timenow_str) == vak.constants.STRFTIME_TIMESTAMP
    timenow_str_as_timestamp = time.mktime(
        datetime.strptime(timenow_str, vak.constants.STRFTIME_TIMESTAMP).timetuple()
    )
    assert before_timestamp <= timenow_str_as_timestamp <= after_timestamp


def test_generate_results_dir_name_as_path(tmp_path):
    before_timestamp = time.mktime(datetime.now())
    results_path = vak.paths.generate_results_dir_name_as_path(root_results_dir=tmp_path)
    after_timestamp = time.mktime(datetime.now())

    assert results_path_matches_expected(results_path,
                                         root_results_dir=tmp_path,
                                         before_timestamp=before_timestamp,
                                         after_timestamp=after_timestamp)


def test_generate_results_dir_name_as_path_no_root_results_dir(tmp_path):
    before_timestamp = time.mktime(datetime.now())
    os.chdir(tmp_path)
    results_path = vak.paths.generate_results_dir_name_as_path(root_results_dir=None)
    after_timestamp = time.mktime(datetime.now())

    assert results_path_matches_expected(results_path,
                                         root_results_dir=tmp_path,
                                         before_timestamp=before_timestamp,
                                         after_timestamp=after_timestamp)


def test_generate_results_dir_name_as_path_nonexistent_root_raises():
    with pytest.raises(NotADirectoryError):
        vak.paths.generate_results_dir_name_as_path(root_results_dir='/obviously/not/an/existent/directory')
