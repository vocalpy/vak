from __future__ import annotations

import torch
import pytest

import vak.metrics.boundary_detection.functional

from .conftest import FIND_HITS_TEST_CASES, PRECISION_RECALL_FSCORE_RVAL_TEST_CASES


@pytest.mark.parametrize(
    'test_case',
    FIND_HITS_TEST_CASES,
)
def test_find_hits(test_case):
    (preds,
     target,
     tolerance,
     decimals,
     expected_hits_target,
     expected_hits_preds,
     expected_diffs,
     expected_num_hits) = (
        test_case.preds,
        test_case.target,
        test_case.tolerance,
        test_case.decimals,
        test_case.expected_hits_target,
        test_case.expected_hits_preds,
        test_case.expected_diffs,
        test_case.expected_num_hits,
    )

    if tolerance is None and decimals is None:
        # then we are testing with the default tolerance and decimals
        num_hits, hits_target, hits_preds, diffs = vak.metrics.boundary_detection.functional.find_hits(
            preds, target,
        )
    elif tolerance is None and decimals is not None:
        # we are testing with default `tolerance` but not decimals
        num_hits, hits_target, hits_preds, diffs = vak.metrics.boundary_detection.functional.find_hits(
            preds, target, decimals=decimals
        )
    elif tolerance is not None and decimals is None:
        # we are testing with default `decimals` but not tolerance
        num_hits, hits_target, hits_preds, diffs = vak.metrics.boundary_detection.functional.find_hits(
            preds, target, tolerance=tolerance,
        )
    else:
        # we are not testing with any defaults
        num_hits, hits_target, hits_preds, diffs = vak.metrics.boundary_detection.functional.find_hits(
            preds, target, tolerance=tolerance, decimals=decimals
        )

    assert torch.equal(num_hits, expected_num_hits)
    assert torch.equal(hits_target, expected_hits_target)
    assert torch.equal(hits_preds, expected_hits_preds)
    torch.testing.assert_close(diffs, expected_diffs)


OK_BOUNDARY_TIMES = torch.tensor([0.1, 1.0, 2.0, 3.0])


@pytest.mark.parametrize(
    'preds, target, expected_exception',
    [
        # preds is zero-D
        (
            torch.tensor(1.0),
            OK_BOUNDARY_TIMES,
            ValueError,
        ),
        # target is zero-D
        (
            OK_BOUNDARY_TIMES,
            torch.tensor(1.0),
            ValueError,
        ),
        # preds is two-D
        (
            torch.tensor([[1.0, 2.0, 3.0]]),
            OK_BOUNDARY_TIMES,
            ValueError,
        ),
        # target is two-D
        (
            OK_BOUNDARY_TIMES,
            torch.tensor([[1.0, 2.0, 3.0]]),
            ValueError,
        ),
        # preds is three-D
        (
            torch.tensor([[[1.0, 2.0, 3.0]]]),
            OK_BOUNDARY_TIMES,
            ValueError,
        ),
        # target is three-D
        (
            OK_BOUNDARY_TIMES,
            torch.tensor([[[1.0, 2.0, 3.0]]]),
            ValueError,
        ),
        # preds not floating point
        # preds is zero-D
        (
            torch.tensor([0, 1, 2, 3]).long(),
            OK_BOUNDARY_TIMES,
            TypeError,
        ),
        # target not floating point
        (
            OK_BOUNDARY_TIMES,
            torch.tensor([0, 1, 2, 3]).long(),
            TypeError,
        ),
        # preds has negative values (and no ignore val)
        (
            torch.tensor([-1.0, 0.0, 1.0, 2.0, 3.0]),
            OK_BOUNDARY_TIMES,
            ValueError,
        ),
        # target has negative values (and no ignore val)
        (
            OK_BOUNDARY_TIMES,
            torch.tensor([-1.0, 0.0, 1.0, 2.0, 3.0]),
            ValueError,
        ),
        # preds not strictly increasing
            (
            torch.tensor([1.0, 2.0, 0.0, 3.0]),
            OK_BOUNDARY_TIMES,
            ValueError,
        ),
        # target not strictly increasing
        (
            OK_BOUNDARY_TIMES,
            torch.tensor([1.0, 2.0, 0.0, 3.0]),
            ValueError,
        )
    ]
)
def test_find_hits_raises(
    preds, target, expected_exception
):
    with pytest.raises(expected_exception):
        vak.metrics.boundary_detection.functional.find_hits(
                preds, target,
            )


@pytest.mark.parametrize(
    'test_case',
    PRECISION_RECALL_FSCORE_RVAL_TEST_CASES,
)
@pytest.mark.parametrize(
    'metrics',
    [
        "precision",
        "recall",
        "fscore",
        "rval",
        ["precision", "recall"],
        ["precision", "fscore"],
        ["precision", "rval"],
        ["recall", "fscore"],
        ["recall", "rval"],
        ["fscore", "rval"],
        ["precision", "recall", "fscore"],
        ["precision", "recall", "rval"],
        ["recall", "fscore", "rval"],
        ["precision", "recall", "fscore", "rval"],
    ]
)
@pytest.mark.parametrize(
    'reduce_fx',
    [
        "mean",
        None,
    ]
)
@pytest.mark.parametrize(
    'return_metric_data',
    [
        True,
        False,
    ]
)
def test_precision_recall_fscore_rval(
    test_case, metrics, reduce_fx, return_metric_data
):
    (
        preds,
        target,
        tolerance,
        decimals,
        ignore_val,
    ) = (
        test_case.preds,
        test_case.target,
        test_case.tolerance,
        test_case.decimals,
        test_case.ignore_val,
    )

    if return_metric_data:
        (
            expected_hits_target,
            expected_hits_preds,
            expected_diffs,
            expected_num_hits
        ) = (
            test_case.expected_hits_target,
            test_case.expected_hits_preds,
            test_case.expected_diffs,
            test_case.expected_num_hits,
        )

    if tolerance is None and decimals is None:
        # then we are testing with the default `tolerance` and `decimals`
        out = vak.metrics.boundary_detection.functional.precision_recall_fscore_rval(
            preds, target, metrics, ignore_val=ignore_val, reduce_fx=reduce_fx, return_metric_data=return_metric_data
        )
    elif tolerance is None and decimals is not None:
        # we are testing with default `tolerance` but not default `decimals``
        out = vak.metrics.boundary_detection.functional.precision_recall_fscore_rval(
            preds, target, metrics, 
            decimals=decimals, 
            ignore_val=ignore_val, reduce_fx=reduce_fx, return_metric_data=return_metric_data
        )
    elif tolerance is not None and decimals is None:
        # we are testing with default `decimals` but not default `tolerance`
        out = vak.metrics.boundary_detection.functional.precision_recall_fscore_rval(
            preds, target, metrics, 
            tolerance=tolerance, 
            ignore_val=ignore_val, reduce_fx=reduce_fx, return_metric_data=return_metric_data
        )
    else:
        # we are not testing with any defaults
        out = vak.metrics.boundary_detection.functional.precision_recall_fscore_rval(
            preds, target, metrics, 
            tolerance=tolerance, decimals=decimals, 
            ignore_val=ignore_val, reduce_fx=reduce_fx, return_metric_data=return_metric_data
        )

    if return_metric_data:
        assert len(out) == 2
        metrics_dict, ir_metric_data_list = out
    else:
        metrics_dict = out
        ir_metric_data_list = None

    assert isinstance(metrics_dict, dict)

    if isinstance(metrics, str):
        metrics = [metrics]  # so we can iterate over it
    for metric_name in metrics:
        assert metric_name in metrics_dict
        metric_val = metrics_dict[metric_name]
        expected_val = getattr(test_case, f"expected_{metric_name}")
        if reduce_fx == "mean":
            torch.testing.assert_close(metric_val, expected_val.mean(), atol=1e-3, rtol=1e-5)
        else:
            torch.testing.assert_close(metric_val, expected_val, atol=1e-3, rtol=1e-5)

    if ir_metric_data_list is not None:
        assert isinstance(ir_metric_data_list, list)
        assert all(
            [isinstance(element, vak.metrics.boundary_detection.functional.IRMetricData) 
             for element in ir_metric_data_list]
        )
        for ind, ir_metric_data in enumerate(ir_metric_data_list):
            assert torch.equal(ir_metric_data.num_hits, expected_num_hits[ind])
            assert torch.equal(ir_metric_data.hits_target, expected_hits_target[ind])
            assert torch.equal(ir_metric_data.hits_preds, expected_hits_preds[ind])
            torch.testing.assert_close(ir_metric_data.diffs, expected_diffs[ind])


OK_BOUNDARY_TIMES_2D = torch.tensor(
    [[0.1, 1.0, 2.0, 3.0], [0.1, 1.0, 2.0, 3.0]]
)


@pytest.mark.parametrize(
    'preds, target, metrics, ignore_val, reduce_fx, return_metric_data, expected_exception',
    [
        # preds is zero-D
        (
            torch.tensor(1.0),
            OK_BOUNDARY_TIMES_2D,
            "precision",
            None,
            "mean",
            False,
            ValueError,
        ),
        # target is zero-D
        (
            OK_BOUNDARY_TIMES_2D,
            torch.tensor(1.0),
            "precision",
            None,
            "mean",
            False,
            ValueError,
        ),
        # preds is one-D but target is two-D
        (
            torch.tensor([1.0, 2.0, 3.0]),
            OK_BOUNDARY_TIMES_2D,
            "precision",
            None,
            "mean",
            False,
            ValueError,
        ),
        # preds is two-D but target is one-D
        (
            OK_BOUNDARY_TIMES_2D,
            torch.tensor([1.0, 2.0, 3.0]),
            "precision",
            None,
            "mean",
            False,
            ValueError,
        ),
        # preds is three-D
        (
            torch.tensor([[[1.0, 2.0, 3.0]]]),
            OK_BOUNDARY_TIMES_2D,
            "precision",
            None,
            "mean",
            False,
            ValueError,
        ),
        # target is three-D
        (
            OK_BOUNDARY_TIMES_2D,
            torch.tensor([[[1.0, 2.0, 3.0]]]),
            "precision",
            None,
            "mean",
            False,
            ValueError,
        ),
        # both preds and target are 2-D but have different size batch dims
        (
            OK_BOUNDARY_TIMES_2D,
            torch.tensor([[1.0, 2.0, 3.0]]),
            "precision",
            None,
            "mean",
            False,
            ValueError,
        ),
        (
            torch.tensor([[1.0, 2.0, 3.0]]),
            OK_BOUNDARY_TIMES_2D,
            "precision",
            None,
            "mean",
            False,
            ValueError,
        ),
        # invalid string metric name
        (
            OK_BOUNDARY_TIMES_2D,
            OK_BOUNDARY_TIMES_2D,
            "accuracy",
            -100,
            "mean",
            False,
            ValueError,
        ),
        # invalid type name in list of metrics
        (
            OK_BOUNDARY_TIMES_2D,
            OK_BOUNDARY_TIMES_2D,
            ["precision", 1],
            -100,
            "mean",
            False,
            TypeError,
        ),
        # invalid metric name in list
        (
            OK_BOUNDARY_TIMES_2D,
            OK_BOUNDARY_TIMES_2D,
            ["accuracy"],
            -100,
            "mean",
            False,
            ValueError,
        ),
        # invalid metric name in list
        (
            OK_BOUNDARY_TIMES_2D,
            OK_BOUNDARY_TIMES_2D,
            ["precision", "accuracy"],
            -100,
            "mean",
            False,
            ValueError,
        ),
        # positive ignore_val
        (
            OK_BOUNDARY_TIMES_2D,
            OK_BOUNDARY_TIMES_2D,
            ["precision", "recall"],
            100.0,
            "mean",
            False,
            ValueError,
        ),
        # invalid reduce_fx
        (
            OK_BOUNDARY_TIMES_2D,
            OK_BOUNDARY_TIMES_2D,
            ["precision", "recall"],
            -100.0,
            "sum",
            False,
            ValueError,
        ),
        # invalid return_ir_metric_data type
        (
            OK_BOUNDARY_TIMES_2D,
            OK_BOUNDARY_TIMES_2D,
            ["precision", "recall"],
            -100.0,
            "mean",
            1.5,
            TypeError,
        ),
        #  the rest of the test cases we expect to actually get thrown by `find_hits`
        # when it is called by `precision_recall_fscore_rval`
        # preds not floating point
        # preds is zero-D
        (
            torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]]).long(),
            OK_BOUNDARY_TIMES_2D,
            "precision",
            None,
            "mean",
            False,
            TypeError,
        ),
        # target not floating point
        (
            OK_BOUNDARY_TIMES_2D,
            torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]]).long(),
            "precision",
            None,
            "mean",
            False,
            TypeError,
        ),
        # preds has negative values (and no ignore val)
        (
            torch.tensor([[-1.0, 0.0, 1.0, 2.0, 3.0], [-1.0, 0.0, 1.0, 2.0, 3.0]]),
            OK_BOUNDARY_TIMES_2D,
            "precision",
            None,
            "mean",
            False,
            ValueError,
        ),
        # target has negative values (and no ignore val)
        (
            OK_BOUNDARY_TIMES_2D,
            torch.tensor([[-1.0, 0.0, 1.0, 2.0, 3.0], [-1.0, 0.0, 1.0, 2.0, 3.0]]),
            "precision",
            None,
            "mean",
            False,
            ValueError,
        ),
        # preds not strictly increasing
            (
            torch.tensor([[1.0, 2.0, 0.0, 3.0], [1.0, 2.0, 0.0, 3.0]]),
            OK_BOUNDARY_TIMES_2D,
            "precision",
            None,
            "mean",
            False,
            ValueError,
        ),
        # target not strictly increasing
        (
            OK_BOUNDARY_TIMES_2D,
            torch.tensor([[1.0, 2.0, 0.0, 3.0], [1.0, 2.0, 0.0, 3.0]]),
            "precision",
            None,
            "mean",
            False,
            ValueError,
        ),
    ]
)
def test_precision_recall_fscore_rval_raises(
    preds, target, metrics, ignore_val, reduce_fx, return_metric_data,
    expected_exception
):
    with pytest.raises(expected_exception):
        vak.metrics.boundary_detection.functional.precision_recall_fscore_rval(
            preds, target, metrics, ignore_val=ignore_val, reduce_fx=reduce_fx, 
            return_metric_data=return_metric_data
        )
