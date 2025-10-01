import pytest
import torch

import vak.metrics.boundary_detection

from .conftest import PRECISION_RECALL_FSCORE_RVAL_TEST_CASES


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
def test_PrecisionRecallFscoreRval(
    test_case, metrics, reduce_fx
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

    if tolerance is None and decimals is None:
        # then we are testing with the default `tolerance` and `decimals`
        instance = vak.metrics.boundary_detection.PrecisionRecallFScoreRVal(
            metrics=metrics, ignore_val=ignore_val, reduce_fx=reduce_fx
        )
    elif tolerance is None and decimals is not None:
        # we are testing with default `tolerance` but not default `decimals``
        instance = vak.metrics.boundary_detection.PrecisionRecallFScoreRVal(
            metrics=metrics, decimals=decimals, ignore_val=ignore_val, reduce_fx=reduce_fx
        )
    elif tolerance is not None and decimals is None:
        # we are testing with default `decimals` but not default `tolerance`
        instance = vak.metrics.boundary_detection.PrecisionRecallFScoreRVal(
            metrics=metrics, tolerance=tolerance, ignore_val=ignore_val, reduce_fx=reduce_fx,
        )
    else:
        # we are not testing with any defaults
        instance = vak.metrics.boundary_detection.PrecisionRecallFScoreRVal(
            metrics=metrics, tolerance=tolerance, decimals=decimals, ignore_val=ignore_val, reduce_fx=reduce_fx
        )
    out = instance(preds, target)
    assert isinstance(out, dict)

    if isinstance(metrics, str):
        metrics = [metrics]  # so we can iterate over it
    for metric_name in metrics:
        assert metric_name in out
        metric_val = out[metric_name]
        expected_val = getattr(test_case, f"expected_{metric_name}")
        if reduce_fx == "mean":
            torch.testing.assert_close(metric_val, expected_val.mean(), atol=1e-3, rtol=1e-5)
        else:
            torch.testing.assert_close(metric_val, expected_val, atol=1e-3, rtol=1e-5)
