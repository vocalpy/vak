from __future__ import annotations
from typing import List, Literal

import torch

from .functional import precision_recall_fscore_rval, BOUNDARY_DETECTION_IR_METRICS, BoundaryDetectionIRMetric


class PrecisionRecallFScoreRVal:
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
    def __init__(
        self,
        metrics: BoundaryDetectionIRMetric | List[BoundaryDetectionIRMetric],
        tolerance: float = 0.01,
        decimals: int = 3,
        ignore_val: float | None = None,
        reduce_fx: Literal["mean"] | None = "mean",
    ):
        
        super().__init__()

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

        if isinstance(metrics, str):
            metrics = [metrics]  # so we can iterate over list in all cases

        self.metrics = metrics
        self.reduce_fx = reduce_fx
        self.tolerance = tolerance
        self.decimals = decimals
        self.ignore_val = ignore_val

    def __call__(
        self, preds: torch.FloatTensor, target: torch.FloatTensor
    ) -> dict[str, torch.FloatTensor]:
        return precision_recall_fscore_rval(
            preds, target, self.metrics, self.tolerance, self.decimals, self.ignore_val, self.reduce_fx, return_metric_data=False,
        )
