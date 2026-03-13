"""Metrics for segmentation adapted from information retrieval."""
from __future__ import annotations

from collections import defaultdict
from typing import List, Literal, Mapping, Tuple

import attr
import torch

from vak.common import validators


def find_hits(
    preds: torch.FloatTensor,
    target: torch.FloatTensor,
    tolerance: float = 0.01,
    decimals: int = 3,
) -> tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.FloatTensor]:
    r"""Find hits in tensors of boundary times.

    Helper function used to compute information retrieval metrics,
    called by
    :func:`~vak.metrics.segmentation.functional.precision_recall_fscore_rval`.

    A boundary time in ``preds``, :math:`t_h`, 
    is considered a hit if its falls within 
    an interval around any boundary time :math:`t_0` 
    in ``target``, plus or minus ``tolerance``

    :math:`t_0 - \Delta t < t < t_0 + \Delta t`

    Only one hit is allowed for each boundary time in ``target``.
    If the condition :math:`|ref_i - hyp_j| < tolerance`
    is true for multiple times :math:`hyp_j` in ``preds``,
    then the hit is assigned to the time with the 
    smallest difference from :math:`ref_i`.

    Both ``preds`` and ``target`` must be 1-dimensional
    tensors of non-negative, strictly increasing values.

    Parameters
    ----------
    preds : torch.FloatTensor
        Boundaries, e.g., onsets or offsets of segments,
        as computed by some method.
    targets : torch.FloatTensor
        Ground truth boundaries that the hypothesized
        boundaries ``preds`` are compared to.
    tolerance : float
        Tolerance, in seconds.
        Default is 0.01, i.e., 10 milliseconds.
        Elements in ``preds`` are considered
        a true positive if they are within a time interval
        around any target boundary :math:`t_0`
        in ``target`` plus or minus
        the ``tolerance``, i.e.,
        if a hypothesized boundary :math:`t_h`
        is within the interval
        :math:`t_0 - \Delta t < t < t_0 + \Delta t`.
        See notes for more detail.
    decimals: int
        The number of decimal places to round both
        ``preds`` and ``target`` to, using
        :func:`torch.round`. This mitigates inflated
        error rates due to floating point error.
        Default is 3, which
        assumes that the values are in seconds
        and should be rounded to milliseconds.

    Returns
    -------
    num_hits : torch.LongTensor
        The number of hits, scalar integer value.
    hits_target : torch.LongTensor
        The indices of hits in ``targets``.
    hits_preds : torch.LongTensor
        The indices of hits in ``preds``.
    diffs : torch.FloatTensor
        Absolute differences :math:`|hit^{target}_i - hit^{preds}_i|`,
        i.e., ``torch.abs(targets[hits_target] - preds[hits_preds])``.
    """
     # ---- pre-conditions ----
    for boundaries_times, name in zip(
        (preds, target),
        ("preds", "target"),
    ):
        validators.is_1d_tensor(boundaries_times, name)
        if not torch.is_floating_point(boundaries_times):
            raise TypeError(
                f"Dtype of boundaries_times {name}must be floating point but was: {boundaries_times.dtype}"
            )
        validators.is_non_negative(boundaries_times, name)
        validators.is_strictly_increasing(boundaries_times, name)

    validators.have_same_dtype(preds, target, "preds", "target")
    validators.have_same_ndim(preds, target, "preds", "target")

    if tolerance < 0.0:
        raise ValueError(
            f"``tolerance`` must be a non-negative number but was: {tolerance}"
        )

    if not isinstance(tolerance, float):
        raise TypeError(
            f"Tolerance must be float but type was: {type(tolerance)}"
        )

    if not isinstance(decimals, int):
        raise ValueError(
            f"``decimals`` must be an integer but was: {decimals}"
        )

    if decimals < 0:
        raise ValueError(
            f"``decimals`` must be a non-negative number but was: {decimals}"
        )

    # we assume float values are in units of seconds and round to ``decimals``,
    # the default is 3 to indicate "milliseconds"
    target = torch.round(target, decimals=decimals)
    preds = torch.round(preds, decimals=decimals)

    # next line: adding a dim to `target` so that we get broadcasting
    # is equivalent to what `numpy.subtract.outer` does, see
    # https://stackoverflow.com/questions/52780559/outer-sum-etc-in-pytorch
    diffs = torch.abs(target[:, None] - preds)
    in_window = diffs <= tolerance
    hits_target, hits_preds = torch.where(in_window)

    # now force there to be only one hit in preds for each hit in target;
    # we do this by choosing the hit that has the smallest absolute difference
    diffs_in_window = diffs[hits_target, hits_preds]
    hits_diffs = sorted(
        zip(hits_target, hits_preds, diffs_in_window), key=lambda x: x[2]
    )
    hits_target_out = []
    hits_preds_out = []
    diffs_out = []
    for hit_target, hit_preds, diff in hits_diffs:
        if hit_target not in hits_target_out and hit_preds not in hits_preds_out:
            hits_target_out.append(hit_target)
            hits_preds_out.append(hit_preds)
            diffs_out.append(diff)
    hits_target_out = torch.tensor(hits_target_out)
    sort_inds = torch.argsort(hits_target_out)
    hits_target_out = hits_target_out[sort_inds]
    hits_preds_out = torch.tensor(hits_preds_out)[sort_inds]
    diffs_out = torch.tensor(diffs_out)[sort_inds]
    num_hits = torch.tensor(hits_target_out.numel()).long()
    return num_hits, hits_target_out, hits_preds_out, diffs_out


@attr.define
class IRMetricData:
    """Class representing data used to compute
    an information retrieval metric.

    This class contains data
    needed to compute metrics like precision and recall
    for estimated boundary times
    compared to target boundary times times.

    The class attributes are the variables
    returned by
    :func:`vak.metrics.boundary_detection.functional.find_hits`.
    Instances of this class are returned 
    along with the value of the computed metrics.

    The values can be useful when computing
    additional statistics,
    e.g., the classes of segments that had higher
    or lower precision or recall,
    or the distribution of
    differences between target times
    and estimated times for some class of events.

    Attributes
    ----------
    num_hits : torch.LongTensor
        The number of hits.
    hits_target : torch.LongTensor
        The indices of hits in ``target``.
    hits_preds : torch.LongTensor
        The indices of hits in ``preds``.
    diffs : torch.LongTensor
        Absolute differences :math:`|hit^{ref}_i - hit^{hyp}_i|`,
        i.e., ``torch.abs(target[hits_target] - preds[hits_preds])``.
    """
    num_hits: torch.LongTensor
    hits_target: torch.LongTensor
    hits_preds: torch.LongTensor
    diffs: torch.LongTensor


# type alias
BoundaryDetectionIRMetric = Literal["precision", "recall", "fscore", "rval"]

# for validation of args
BOUNDARY_DETECTION_IR_METRICS = (
    "precision", "recall", "fscore", "rval"
)


def precision_recall_fscore_rval(
    preds: torch.FloatTensor,
    target: torch.FloatTensor,
    metrics: BoundaryDetectionIRMetric | List[BoundaryDetectionIRMetric],
    tolerance: float = 0.01,
    decimals: int = 3,
    ignore_val: float | None = None,
    reduce_fx: Literal["mean"] | None = "mean",
    return_metric_data: bool = False,
) -> Mapping[BoundaryDetectionIRMetric, torch.Tensor] | Tuple[Mapping[BoundaryDetectionIRMetric, torch.Tensor], List[IRMetricData]]:
    r"""Compute information retrieval metrics for boundary detection: 
    precision, recall, F-score, and R-value.

    The metrics are computed by comparing a predicted tensor of 
    boundary times ``preds`` with  
    a ground truth tensor of boundaries times ``target``.

    Boundary times in ``preds`` are considered
    a hit, i.e., correctly detected, 
    if they are within a time interval
    around any boundary time :math:`t_0`
    in ``target`` plus or minus
    the ``tolerance``, i.e.,
    if a hypothesized boundary time :math:`t_h`
    is within the interval
    :math:`t_0 - \Delta t < t < t_0 + \Delta t`.
    See notes for more detail.

    Parameters
    ----------
    preds : torch.FloatTensor
        Batch of predicted boundary times, 
        e.g., onsets or offsets of segments.
    target : torch.FloatTensor
        Batch of ground truth boundary times that hypothesized
        boundary times ``preds`` are compared to.
    metrics : string or list of strings
        The name(s) of the metric to compute.
        Either a string or a list of strings. 
        Valid names are: ``{"precision", "recall", "fscore", "rval"}``.
    tolerance : float
        Tolerance, in seconds.
        Default is 0.01, i.e., 10 milliseconds.
    decimals: int
        The number of decimal places to round both
        ``preds`` and ``target`` to, using
        :func:`torch.round`. This mitigates inflated
        error rates due to floating point error.
        Default is 3, which
        assumes that the values are in seconds
        and should be rounded to milliseconds.
    ignore_val : float, optional
        Padding value to ignore in ``preds`` and ``target``.
        If given, must be a negative number 
        (boundary times cannot be negative).
        Default is ``vak.common.constants.DEFAULT_BOUNDARY_TIMES_PADVAL``.
    reduce_fx : string or None
        How to reduce the computed metric across the batch.
        Default is ``'mean'`` (currently the only 
        implemented reduction). If ``None`` then no 
        reduction is performed, and the returned 
        ``metrics`` :class:`dict` will map 
        metric names to a tensor of the same size 
        as the batch.

    Returns
    -------
    metrics : dict
        A :class:`dict` mapping the names in 
        ``metrics`` to (tensor) values.
    metric_data : IRMetricData
        Instance of :class:`IRMetricData`
        with the number of hits, the indices of hits 
        in both ``preds`` and ``target``,
        and the absolute difference between times
        in ``preds`` and ``target``for the hits.

    Notes
    -----
    Precision is defined as the number of true positives (:math:`T_p`)
    over the number of true positives
    plus the number of false positives (:math:`F_p`).

    :math:`P = \\frac{T_p}{T_p+F_p}`.

    Recall (:math:`R`) is defined as the number of true positives (:math:`T_p`)
    over the number of true positives plus the number of false negatives
    (:math:`F_n`).

    :math:`R = \\frac{T_p}{T_p + F_n}`

    The F-score can be interpreted as a harmonic mean of the precision and
    recall. It ranges between 0.0 and 1.0, and 
    the relative contribution of precision and recall to the F-score are
    equal. The formula for the F-score is:

    ``f_score = 2 * (precision * recall) / (precision + recall)``

    The R-value [5]_ is a metric developed for speech 
    segmentation, specifically to penalize algorithms 
    that over-segment (which results in inflated recall values). 
    We adapt our implementation from [6]_, 
    as used in [7]_.

    .. math:
    
       OS = ((P / R) - 1)
       r_1 = \sqrt{(1 - P) + (OS)^2}
       r_2 = \frac{-OS + P - 1}{\sqrt{2}}
       R = 1 - \frac{\text{abs}(r_1) + \text{abs}(r_2)}{2}

    All these metrics require computing the number of true positives,
    and both F-score and R-value require computing both precision and recall.
    To avoid duplicate computation, we provide a single function 
    that computes them all.

    Precision and recall are computed using hits found with
    :func:`~vak.metrics.boundary_detection.functional.find_hits`.
    See docstring of that function for more detail on how hits are computed.

    Both ``preds`` and ``target`` must be 
    tensors of non-negative, strictly increasing values.

    The addition of a tolerance parameter is based on [1]_.
    This is also sometimes known as a "collar" [2]_ or "forgiveness collar" [3]_.
    The value for the tolerance can be determined by visual inspection
    of the distribution; see for example [4]_.

    References
    ----------

    .. [1] Kemp, T., Schmidt, M., Whypphal, M., & Waibel, A. (2000, June).
       Strategies for automatic segmentation of audio data.
       In 2000 ieee international conference on acoustics, speech, and signal processing.
       proceedings (cat. no. 00ch37100) (Vol. 3, pp. 1423-1426). IEEE.

    .. [2] Jordán, P. G., & Giménez, A. O. (2023).
       Advances in Binary and Multiclass Sound Segmentation with Deep Learning Techniques.

    .. [3] NIST. (2009). The 2009 (RT-09) Rich Transcription Meeting Recognition Evaluation Plan.
       <https://web.archive.org/web/20100606041157if_/http://www.itl.nist.gov/iad/mig/thyps/rt/2009/docs/rt09-meeting-eval-plan-v2.pdf>

    .. [4] Du, P., & Troyer, T. W. (2006).
       A segmentation algorithm for zebra finch song at the note level.
       Neurocomputing, 69(10-12), 1375-1379.
    
    .. [5] Räsänen, Okko Johannes, Unto Kalervo Laine, and Toomas Altosaar. 
           "An improved speech segmentation quality measure: the r-value." 
           Interspeech. 2009.

    .. [6] https://github.com/felixkreuk/UnsupSeg

    .. [7] Felix Kreuk, Joseph Keshet, and Yossi Adi.
           "Self-Supervised Contrastive Learning for Unsupervised Phoneme Segmentation."
           Interspeech, 2020.
    """
    # ---- pre-conditions ----
    for boundary_times, name in zip(
        (preds, target),
        ("preds", "target")
    ):
        # other pre-conditions on boundary times are checked in `find_hits`
        validators.is_1d_or_2d_tensor(boundary_times, name)

    validators.have_same_ndim(preds, target, "preds", "target")

    if preds.ndim == 2:
        if preds.shape[0] != target.shape[0]:
            raise ValueError(
                "`preds` and `target` must be the same size in the first (batch) dimension, "
                f"but preds.shape[0]={preds.shape[0]} != target.shape[0]={target.shape[0]}"
            )

    if isinstance(metrics, str):
        if metrics not in BOUNDARY_DETECTION_IR_METRICS:
            raise ValueError(
                f"Invalid value for `metrics`: {metrics}\n"
                "`metrics` must be a string or list of strings, any of the following:"
                f"{BOUNDARY_DETECTION_IR_METRICS}"
            )
    elif isinstance(metrics, list):
        if not all(
            [isinstance(metric_name, str) for metric_name in metrics]
        ):
            invalid_types = [
                type(metric_name) 
                for metric_name in metrics
                if not isinstance(metric_name, str)
            ]
            raise TypeError(
                f"Invalid types in list of `metrics`: {invalid_types}\n"
                "`metrics` must be a string or list of strings, any of the following:"
                f"{BOUNDARY_DETECTION_IR_METRICS}"
            )
        if not all(
            [metric_name in BOUNDARY_DETECTION_IR_METRICS for metric_name in metrics]
        ):
            invalid_names = [
                metric_name
                for metric_name in metrics
                if metric_name not in BOUNDARY_DETECTION_IR_METRICS
            ]
            raise ValueError(
                f"Invalid values for `metrics`: {invalid_names}\n"
                "`metrics` must be a string or list of strings, any of the following:"
                f"{BOUNDARY_DETECTION_IR_METRICS}"
            )
    else:
        TypeError(
            f"Invalid type for `metrics`: {metric}\n"
            "`metrics` must be a string or list of strings, any of the following:"
            f"{BOUNDARY_DETECTION_IR_METRICS}"            
        )
    if reduce_fx is not None:
        if reduce_fx != "mean":
            raise ValueError(
                f"`reduce_fx` must be either \"mean\" or None, but was: {reduce_fx}"
            )
    if not isinstance(return_metric_data, bool):
        raise TypeError(
            f"`return_metric_data` must be True or False (bool) but type was: {type(return_metric_data)}"
        )
    if ignore_val is not None:
        if ignore_val >= 0.0:
            raise ValueError(
                f"`ignore_val` must be negative (to avoid clash with valid boundary times) but was: {ignore_val}"
            )

    # ---- actually compute metrics ----
    if isinstance(metrics, str):
        metrics = [metrics]  # so we can iterate over list in all cases

    if (preds.ndim == 1 and target.ndim == 1):
        preds_batch = [preds]
        target_batch = [target]
    elif (preds.ndim == 2 and target.ndim == 2):
        preds_batch = torch.unbind(preds)
        target_batch = torch.unbind(target)

    metrics_out = defaultdict(list)
    if return_metric_data:
        ir_metric_data = []
    for item_preds, item_target in zip(
        preds_batch, target_batch
    ):
        if ignore_val is not None:
            item_preds = item_preds[item_preds != ignore_val]
            item_target = item_target[item_target != ignore_val]

        # edge case: if both target and preds have a length of zero, we have a score of 1.0
        # but no hits. This is to avoid punishing the correct preds that there are no boundaries.
        # See https://github.com/vocalpy/vocalpy/issues/170
        if len(item_target) == 0 and len(item_preds) == 0:
            if return_metric_data:
                ir_metric_data.append(
                    IRMetricData(
                        num_hits=torch.tensor(0).long(),
                        hits_target=torch.tensor([]).long(),
                        hits_preds=torch.tensor([]).long(),
                        diffs=torch.tensor([]).float(),
                    )
                )
            for metric_name in metrics:
                metrics_out[metric_name].append(
                    torch.tensor(1.0, dtype=torch.float32)
                )
            continue

        # If we have no boundary times in just one of the tensors, we get a score of 0.
        if len(item_target) == 0 or len(item_preds) == 0:
            if return_metric_data:
                ir_metric_data.append(
                        IRMetricData(
                            num_hits=torch.tensor(0).long(),
                            hits_target=torch.tensor([]).long(),
                            hits_preds=torch.tensor([]).long(),
                            diffs=torch.tensor([]).float(),
                        )
                    )
            for metric_name in metrics:
                metrics_out[metric_name].append(
                    torch.tensor(0.0, dtype=torch.float32),
                )
            continue

        # we passed edge cases; so, actually compute hits and metrics
        num_hits, hits_target, hits_preds, diffs = find_hits(
            item_preds, item_target, tolerance, decimals
        )
        if return_metric_data:
            ir_metric_data.append(
                IRMetricData(num_hits, hits_target, hits_preds, diffs)
            )

        for metric_name in metrics:
            if metric_name == "precision":
                metric_val = torch.FloatTensor(num_hits / item_preds.numel())
            elif metric_name == "recall":
                metric_val = torch.FloatTensor(num_hits / item_target.numel())
            elif metric_name in ("fscore", "rval"):
                precision_ = torch.FloatTensor(num_hits / item_preds.numel())
                recall_ = torch.FloatTensor(num_hits / item_target.numel())
                if metric_name == "fscore":
                    if torch.isclose(precision_, torch.tensor(0.0)) and torch.isclose(recall_, torch.tensor(0.0)):
                        # avoids divide-by-zero that would give NaN
                        metric_val = torch.tensor(0).float()
                    else:
                        metric_val = 2 * (precision_ * recall_) / (precision_ + recall_)
                elif metric_name == "rval":
                    # Implementation adapted from Felix Kreuk's UnSupSeg
                    # https://github.com/felixkreuk/UnsupSeg/blob/master/utils.py#L89C1-L92C49
                    os = recall_ / (precision_ + torch.finfo(torch.float32).eps) - 1
                    r1 = torch.sqrt((1 - recall_) ** 2 + os ** 2)
                    r2 = (-os + recall_ - 1) / (torch.sqrt(torch.tensor(2.0)))
                    metric_val = 1 - (torch.abs(r1) + torch.abs(r2)) / 2
            metrics_out[metric_name].append(metric_val)
    
    metrics_out = {
        metric_name: torch.tensor(metric_vals)
        for metric_name, metric_vals in metrics_out.items()
    }
    if reduce_fx == "mean":
        metrics_out = {
            metric_name: metric_vals.mean()
            for metric_name, metric_vals in metrics_out.items()
        }

    if return_metric_data:
        return metrics_out, ir_metric_data
    else:
        return metrics_out
