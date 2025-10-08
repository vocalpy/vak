"""Metrics for segmentation adapted from information retrieval."""

from __future__ import annotations

import attr
import torch

from . import validators


def find_hits(
    preds: torch.Tensor,
    target: torch.Tensor,
    tolerance: float = 0.0,
    decimals: int = 3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Find hits in tensors of event times.

    This is a helper function used to compute information retrieval metrics.
    Specifically, this function is called by
    :func:`~vocalpy.metrics.segmentation.ir.precision_recall_fscore`.

    An element in ``preds``, is considered a hit
    if its value :math:`t_h` falls within an interval around
    any value in ``target``, :math:`t_0`, plus or minus ``tolerance``

    :math:`t_0 - \Delta t < t < t_0 + \Delta t`

    This function only allows there to be zero or one hit
    for each element in ``target``, but not more than one.
    If the condition :math:`|ref_i - hyp_j| < tolerance`
    is true for multiple values :math:`hyp_j` in ``preds``,
    then the value with the smallest difference from :math:`ref_i`
    is considered a hit.

    Both ``preds`` and ``target`` must be 1-dimensional
    tensors of non-negative, strictly increasing values.
    If you have two tensors ``onsets`` and ``offsets``,
    you can concatenate those into a single valid tensor
    of boundary times using :func:`concat_starts_and_stops`
    that you can then pass to this function.

    Parameters
    ----------
    preds : torch.Tensor
        Boundaries, e.g., onsets or offsets of segments,
        as computed by some method.
    targets : torch.Tensor
        Ground truth boundaries that the hypothesized
        boundaries ``preds`` are compared to.
    tolerance : float or int
        Tolerance, in seconds.
        Elements in ``preds`` are considered
        a true positive if they are within a time interval
        around any target boundary :math:`t_0`
        in ``target`` plus or minus
        the ``tolerance``, i.e.,
        if a hypothesized boundary :math:`t_h`
        is within the interval
        :math:`t_0 - \Delta t < t < t_0 + \Delta t`.
        Default is None,
        in which case it is set to ``0``
        (either float or int, depending on the
        dtype of ``preds`` and ``target``).
        See notes for more detail.
    decimals: int
        The number of decimal places to round both
        ``preds`` and ``target`` to, using
        :func:`numpy.round`. This mitigates inflated
        error rates due to floating point error.
        Rounding is only applied
        if both ``preds`` and ``target``
        are floating point values. To avoid rounding,
        e.g. to compute strict precision and recall,
        pass in the value ``False``. Default is 3, which
        assumes that the values are in seconds
        and should be rounded to milliseconds.

    Returns
    -------
    hits_ref : torch.Tensor
        The indices of hits in ``targets``.
    hits_hyp : torch.Tensor
        The indices of hits in ``preds``.
    diffs : torch.Tensor
        Absolute differences :math:`|hit^{ref}_i - hit^{hyp}_i|`,
        i.e., ``torch.abs(targets[hits_ref] - preds[hits_hyp])``.
    """
    validators.is_valid_boundaries_tensor(preds)  # 1-d, non-negative, strictly increasing
    validators.is_valid_boundaries_tensor(target)
    validators.have_same_dtype(preds, target)

    if tolerance < 0.0:
        raise ValueError(
            f"``tolerance`` must be a non-negative number but was: {tolerance}"
        )

    if not isinstance(tolerance, float):
        raise TypeError(
            "If ``preds`` and ``target`` are floating, tolerance must be a float also, "
            f"but type was: {type(tolerance)}"
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
    return hits_target_out, hits_preds_out, diffs_out


@attr.define
class IRMetricData:
    """Class representing data used to compute
    an information retrieval metric.

    This class contains data
    needed to compute metrics like precision and recall
    for estimated event times
    compared to target event times.

    The class attributes are the variables
    returned by
    :func:`vocalpy.metrics.segmentation.find_hits`.
    Instances of this class are returned by
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
    hits_ref : torch.Tensor
        The indices of hits in ``target``.
    hits_hyp : torch.Tensor
        The indices of hits in ``preds``.
    diffs : torch.Tensor
        Absolute differences :math:`|hit^{ref}_i - hit^{hyp}_i|`,
        i.e., ``torch.abs(target[hits_ref] - preds[hits_hyp])``.
    """

    hits_ref: torch.Tensor
    hits_hyp: torch.Tensor
    diffs: torch.Tensor


def precision_recall_fscore(
    preds: torch.Tensor,
    target: torch.Tensor,
    metric: str,
    tolerance: float | int | None = None,
    decimals: int | bool | None = None,
) -> tuple[float, int, IRMetricData]:
    r"""Helper function that computes precision, recall, and the F-score.

    Since all these metrics require computing the number of true positives,
    and F-score is a combination of precision and recall,
    we rely on this helper function to compute them.
    You can compute each directly without needing the ``metric`` argument
    that this function requires by calling the appropriate function:
    :func:`~vocalpy.metrics.segmentation.ir.precision`,
    :func:`~vocalpy.metrics.segmentation.ir.recall`, and
    :func:`~vocalpy.metrics.segmentation.ir.fscore`.
    See docstrings of those functions for definitions of the metrics
    in terms of segmentation algorithms.

    Precision, recall, and F-score are computed using hits found with
    :func:`vocalpy.metrics.segmentation._ir_helper.find_hits`.
    See docstring of that function for more detail on how hits are computed.

    Both ``preds`` and ``target`` must be 1-dimensional
    tensors of non-negative, strictly increasing values.
    If you have two tensors ``onsets`` and ``offsets``,
    you can concatenate those into a single valid tensor
    of boundary times using :func:`concat_starts_and_stops`
    that you can then pass to this function.

    Parameters
    ----------
    preds : torch.Tensor
        Boundaries, e.g., onsets or offsets of segments,
        as computed by some method.
    target : torch.Tensor
        Ground truth boundaries that the hypothesized
        boundaries ``preds`` are compared to.
    metric : str
        The name of the metric to compute.
        One of: ``{"precision", "recall", "fscore"}``.
    tolerance : float or int
        Tolerance, in seconds.
        Elements in ``preds`` are considered
        a true positive if they are within a time interval
        around any target boundary :math:`t_0`
        in ``target`` plus or minus
        the ``tolerance``, i.e.,
        if a hypothesized boundary :math:`t_h`
        is within the interval
        :math:`t_0 - \Delta t < t < t_0 + \Delta t`.
        Default is None,
        in which case it is set to ``0``
        (either float or int, depending on the
        dtype of ``preds`` and ``target``).
        See notes for more detail.
    decimals: int
        The number of decimal places to round both
        ``preds`` and ``target`` to, using
        :func:`numpy.round`. This mitigates inflated
        error rates due to floating point error.
        Rounding is only applied
        if both ``preds`` and ``target``
        are floating point values. To avoid rounding,
        e.g. to compute strict precision and recall,
        pass in the value ``False``. Default is 3, which
        assumes that the values are in seconds
        and should be rounded to milliseconds.

    Returns
    -------
    metric_value : float
        Value for ``metric``.
    n_tp : int
        The number of true positives.
    metric_data : IRMetricData
        Instance of :class:`IRMetricData`
        with indices of hits in both
        ``preds`` and ``target``,
        and the absolute difference between times
        in ``preds`` and ``target``
        for the hits.

    Notes
    -----
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
    """
    if metric not in {"precision", "recall", "fscore"}:
        raise ValueError(
            f'``metric`` must be one of: {{"precision", "recall", "fscore"}} but was: {metric}'
        )

    # edge case: if both target and preds have a length of zero, we have a score of 1.0
    # but no hits. This is to avoid punishing the correct preds that there are no boundaries.
    # See https://github.com/vocalpy/vocalpy/issues/170
    if len(target) == 0 and len(preds) == 0:
        return (
            1.0,
            0,
            IRMetricData(
                hits_ref=torch.tensor([]),
                hits_hyp=torch.tensor([]),
                diffs=torch.tensor([]),
            ),
        )

    # If we have no boundaries, we get no score.
    if len(target) == 0 or len(preds) == 0:
        return (
            0.0,
            0,
            IRMetricData(
                hits_ref=torch.tensor([]),
                hits_hyp=torch.tensor([]),
                diffs=torch.tensor([]),
            ),
        )

    hits_ref, hits_hyp, diffs = find_hits(
        preds, target, tolerance, decimals
    )
    metric_data = IRMetricData(hits_ref, hits_hyp, diffs)
    n_tp = hits_hyp.size
    if metric == "precision":
        precision_ = n_tp / preds.size
        return precision_, n_tp, metric_data
    elif metric == "recall":
        recall_ = n_tp / target.size
        return recall_, n_tp, metric_data
    elif metric == "fscore":
        precision_ = n_tp / preds.size
        recall_ = n_tp / target.size
        if torch.isclose(precision_, 0.0) and torch.isclose(recall_, 0.0):
            # avoids divide-by-zero that would give NaN
            return 0.0, n_tp, metric_data
        fscore_ = 2 * (precision_ * recall_) / (precision_ + recall_)
        return fscore_, n_tp, metric_data


def precision(
    preds: torch.Tensor,
    target: torch.Tensor,
    tolerance: float | int | None = None,
    decimals: int | bool | None = None,
) -> tuple[float, int, IRMetricData]:
    r"""Compute precision :math:`P` for a segmentation.

    Computes the metric from a hypothesized vector of boundaries
    ``preds`` returned by a segmentation algorithm
    and a target vector of boundaries ``target``,
    e.g., boundaries cleaned by a human expert
    or boundaries from a benchmark dataset.

    Precision is defined as the number of true positives (:math:`T_p`)
    over the number of true positives
    plus the number of false positives (:math:`F_p`).

    :math:`P = \\frac{T_p}{T_p+F_p}`.

    The number of true positives ``n_tp`` is computed by calling
    :func:`vocalpy.metrics.segmentation.ir.compute_true_positives`.
    This function then computes the precision as
    ``precision = n_tp / preds.size``.


    Both ``preds`` and ``target`` must be 1-dimensional
    tensors of non-negative, strictly increasing values.
    If you have two tensors ``onsets`` and ``offsets``,
    you can concatenate those into a single valid tensor
    of boundary times using :func:`concat_starts_and_stops`
    that you can then pass to this function.

    Parameters
    ----------
    preds : torch.Tensor
        Boundaries, e.g., onsets or offsets of segments,
        as computed by some method.
    target : torch.Tensor
        Ground truth boundaries that the hypothesized
        boundaries ``preds`` are compared to.
    tolerance : float or int
        Tolerance, in seconds.
        Elements in ``preds`` are considered
        a true positive if they are within a time interval
        around any target boundary :math:`t_0`
        in ``target`` plus or minus
        the ``tolerance``, i.e.,
        if a hypothesized boundary :math:`t_h`
        is within the interval
        :math:`t_0 - \Delta t < t < t_0 + \Delta t`.
        Default is None,
        in which case it is set to ``0``
        (either float or int, depending on the
        dtype of ``preds`` and ``target``).
    decimals: int
        The number of decimal places to round both
        ``preds`` and ``target`` to, using
        :func:`numpy.round`. This mitigates inflated
        error rates due to floating point error.
        Rounding is only applied
        if both ``preds`` and ``target``
        are floating point values. To avoid rounding,
        e.g. to compute strict precision and recall,
        pass in the value ``False``. Default is 3, which
        assumes that the values are in seconds
        and should be rounded to milliseconds.

    Returns
    -------
    precision : float
        Value for precision, computed as described above.
    n_tp : int
        The number of true positives.
    metric_data : IRMetricData
        Instance of :class:`IRMetricData`
        with indices of hits in both
        ``preds`` and ``target``,
        and the absolute difference between times
        in ``preds`` and ``target``
        for the hits.

    Examples
    --------
    >>> preds = torch.tensor([1, 6, 10, 16])
    >>> target = torch.tensor([0, 5, 10, 15])
    >>> prec, n_tp, ir_metric_data = vocalpy.metrics.segmentation.ir.precision(preds, target, tolerance=0)
    >>> print(prec)
    0.25
    >>> print(ir_metric_data.hits_hyp)
    torch.tensor([2])

    >>> preds = torch.tensor([0, 1, 5, 10])
    >>> target = torch.tensor([0, 5, 10])
    >>> fscore, n_tp, metric_data = vocalpy.metrics.segmentation.ir.precision(preds, target, tolerance=1)
    >>> print(fscore)
    0.75
    >>> print(ir_metric_data.hits_hyp)
    torch.tensor([0, 2, 3])

    Notes
    -----
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
       https://web.archive.org/web/20100606041157if_/http://www.itl.nist.gov/iad/mig/thyps/rt/2009/docs/rt09-meeting-eval-plan-v2.pdf

    .. [4] Du, P., & Troyer, T. W. (2006).
       A segmentation algorithm for zebra finch song at the note level.
       Neurocomputing, 69(10-12), 1375-1379.
    """
    return precision_recall_fscore(
        preds, target, "precision", tolerance, decimals
    )


def recall(
    preds: torch.Tensor,
    target: torch.Tensor,
    tolerance: float | int | None = None,
    decimals: int | bool | None = None,
) -> tuple[float, int, IRMetricData]:
    r"""Compute recall :math:`R` for a segmentation.

    Computes the metric from a hypothesized vector of boundaries
    ``preds`` returned by a segmentation algorithm
    and a target vector of boundaries ``target``,
    e.g., boundaries cleaned by a human expert
    or boundaries from a benchmark dataset.

    Recall (:math:`R`) is defined as the number of true positives (:math:`T_p`)
    over the number of true positives plus the number of false negatives
    (:math:`F_n`).

    :math:`R = \\frac{T_p}{T_p + F_n}`

    The number of true positives ``n_tp`` is computed by calling
    :func:`vocalpy.metrics.segmentation.ir.compute_true_positives`.
    This function then computes the recall as
    ``recall = n_tp / target.size``.

    Both ``preds`` and ``target`` must be 1-dimensional
    tensors of non-negative, strictly increasing values.
    If you have two tensors ``onsets`` and ``offsets``,
    you can concatenate those into a single valid tensor
    of boundary times using :func:`concat_starts_and_stops`
    that you can then pass to this function.

    Parameters
    ----------
    preds : torch.Tensor
        Boundaries, e.g., onsets or offsets of segments,
        as computed by some method.
    target : torch.Tensor
        Ground truth boundaries that the hypothesized
        boundaries ``preds`` are compared to.
    tolerance : float or int
        Tolerance, in seconds.
        Elements in ``preds`` are considered
        a true positive if they are within a time interval
        around any target boundary :math:`t_0`
        in ``target`` plus or minus
        the ``tolerance``, i.e.,
        if a hypothesized boundary :math:`t_h`
        is within the interval
        :math:`t_0 - \Delta t < t < t_0 + \Delta t`.
        Default is None,
        in which case it is set to ``0``
        (either float or int, depending on the
        dtype of ``preds`` and ``target``).
    decimals: int
        The number of decimal places to round both
        ``preds`` and ``target`` to, using
        :func:`numpy.round`. This mitigates inflated
        error rates due to floating point error.
        Rounding is only applied
        if both ``preds`` and ``target``
        are floating point values. To avoid rounding,
        e.g. to compute strict precision and recall,
        pass in the value ``False``. Default is 3, which
        assumes that the values are in seconds
        and should be rounded to milliseconds.

    Returns
    -------
    recall : float
        Value for recall, computed as described above.
    n_tp : int
        The number of true positives.
    metric_data : IRMetricData
        Instance of :class:`IRMetricData`
        with indices of hits in both
        ``preds`` and ``target``,
        and the absolute difference between times
        in ``preds`` and ``target``
        for the hits.

    Examples
    --------
    >>> preds = torch.tensor([1, 6, 10, 16])
    >>> target = torch.tensor([0, 5, 10, 15])
    >>> recall, n_tp, ir_metric_data = vocalpy.metrics.segmentation.ir.recall(preds, target, tolerance=0)
    >>> print(recall)
    0.25
    >>> print(ir_metric_data.hits_hyp)
    torch.tensor([2])

    >>> preds = torch.tensor([0, 1, 5, 10])
    >>> target = torch.tensor([0, 5, 10])
    >>> recall, n_tp, metric_data = vocalpy.metrics.segmentation.ir.recall(preds, target, tolerance=1)
    >>> print(recall)
    1.0
    >>> print(ir_metric_data.hits_hyp)
    torch.tensor([0, 2, 3])

    Notes
    -----
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
       https://web.archive.org/web/20100606041157if_/http://www.itl.nist.gov/iad/mig/thyps/rt/2009/docs/rt09-meeting-eval-plan-v2.pdf

    .. [4] Du, P., & Troyer, T. W. (2006).
       A segmentation algorithm for zebra finch song at the note level.
       Neurocomputing, 69(10-12), 1375-1379.
    """
    return precision_recall_fscore(
        preds, target, "recall", tolerance, decimals
    )


def fscore(
    preds: torch.Tensor,
    target: torch.Tensor,
    tolerance: float | int | None = None,
    decimals: int | bool | None = None,
) -> tuple[float, int, IRMetricData]:
    r"""Compute the F-score for a segmentation.

    Computes the metric from a
    hypothesized vector of boundaries ``preds``
    returned by a segmentation algorithm
    and a target vector of boundaries ``target``,
    e.g., boundaries cleaned by a human expert
    or boundaries from a benchmark dataset.

    The F-score can be interpreted as a harmonic mean of the precision and
    recall, where an F-score reaches its best value at 1 and worst score at 0.
    The relative contribution of precision and recall to the F-score are
    equal. The formula for the F-score is:

    ``f_score = 2 * (precision * recall) / (precision + recall)``

    Both ``preds`` and ``target`` must be 1-dimensional
    tensors of non-negative, strictly increasing values.
    If you have two tensors ``onsets`` and ``offsets``,
    you can concatenate those into a single valid tensor
    of boundary times using :func:`concat_starts_and_stops`
    that you can then pass to this function.

    Parameters
    ----------
    preds : torch.Tensor
        Boundaries, e.g., onsets or offsets of segments,
        as computed by some method.
    target : torch.Tensor
        Ground truth boundaries that the hypothesized
        boundaries ``preds`` are compared to.
    tolerance : float or int
        Tolerance, in seconds.
        Elements in ``preds`` are considered
        a true positive if they are within a time interval
        around any target boundary :math:`t_0`
        in ``target`` plus or minus
        the ``tolerance``, i.e.,
        if a hypothesized boundary :math:`t_h`
        is within the interval
        :math:`t_0 - \Delta t < t < t_0 + \Delta t`.
        Default is None,
        in which case it is set to ``0``
        (either float or int, depending on the
        dtype of ``preds`` and ``target``).
    decimals: int
        The number of decimal places to round both
        ``preds`` and ``target`` to, using
        :func:`numpy.round`. This mitigates inflated
        error rates due to floating point error.
        Rounding is only applied
        if both ``preds`` and ``target``
        are floating point values. To avoid rounding,
        e.g. to compute strict precision and recall,
        pass in the value ``False``. Default is 3, which
        assumes that the values are in seconds
        and should be rounded to milliseconds.

    Returns
    -------
    f_score : float
        Value for F-score, computed as described above.
    n_tp : int
        The number of true positives.
    metric_data : IRMetricData
        Instance of :class:`IRMetricData`
        with indices of hits in both
        ``preds`` and ``target``,
        and the absolute difference between times
        in ``preds`` and ``target``
        for the hits.

    Examples
    --------
    >>> preds = torch.tensor([1, 6, 10, 16])
    >>> target = torch.tensor([0, 5, 10, 15])
    >>> prec, n_tp, ir_metric_data = vocalpy.metrics.segmentation.ir.fscore(preds, target, tolerance=0)
    >>> print(prec)
    0.25
    >>> print(ir_metric_data.hits_hyp)
    torch.tensor([2])

    >>> preds = torch.tensor([0, 1, 5, 10])
    >>> target = torch.tensor([0, 5, 10])
    >>> prec, n_tp, metric_data = vocalpy.metrics.segmentation.ir.fscore(preds, target, tolerance=1)
    >>> print(prec)
    0.75
    >>> print(ir_metric_data.hits_hyp)
    torch.tensor([0, 2, 3])

    Notes
    -----
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
       https://web.archive.org/web/20100606041157if_/http://www.itl.nist.gov/iad/mig/thyps/rt/2009/docs/rt09-meeting-eval-plan-v2.pdf

    .. [4] Du, P., & Troyer, T. W. (2006).
       A segmentation algorithm for zebra finch song at the note level.
       Neurocomputing, 69(10-12), 1375-1379.
    """
    return precision_recall_fscore(
        preds, target, "fscore", tolerance, decimals
    )


def concat_starts_and_stops(
    starts: torch.Tensor, stops: torch.Tensor
) -> torch.Tensor:
    """Concatenate tensors of start and stop times
    into a single tensor of boundary times.

    Some segmenting algorithms return lists of segments
    denoted by the start and stop times of each segment.
    (You may also see these times called "onsets" and "offsets".)
    Typically, such segmenting algorithms work by setting a
    threshold on some acoustic feature, e.g. the Root-Mean-Square
    of the spectral power.
    This means the segments will be separated by brief
    "silent gaps" (periods below threshold).

    To compute metrics for segmentation like precision
    and recall, you may want to combine the start and stop
    times into a single tensor of boundary times.
    Such an approach is valid if we think of a "silent gaps"
    as a segment whose start time is the stop time/offset of the
    preceding segment.

    If you have tensors of start and stop times,
    you can concatenate into a single tensor of
    boundary times with this function.
    Both ``starts`` and ``stops`` must be 1-dimensional
    tensors of non-negative, strictly increasing values,
    with the same ``dtype``.
    The two tensors ``starts`` and ``stops``
    must be the same length, and all start times
    must be less than the corresponding stop times,
    i.e., ``torch.all(starts < stops)`` should evaluate
    to ``True``.

    Parameters
    ----------
    starts : torch.Tensor
        Tensor of start times of segments.
    stops : torch.Tensor
        Tensor of stop times of segments.

    Returns
    -------
    boundaries : torch.Tensor
        The tensor of boundary times,
        concatenated and then sorted,
        so that
        ``torch.all(boundaries[1:] > boundaries[:-1]``
        evaluates to ``True``.

    Examples
    --------

    >>> starts = torch.tensor([0, 8, 16, 24])
    >>> stops = torch.tensor([4, 12, 20, 28])
    >>> concat_starts_and_stops(starts, stops)
    torch.tensor([0, 4, 8, 12, 16, 20, 24, 28])

    >>> starts = torch.tensor([0.000, 8.000, 16.000, 24.000])
    >>> stops = torch.tensor([4.000, 12.000, 20.000, 28.000])
    >>> concat_starts_and_stops(starts, stops)
    torch.tensor([0.000, 4.000, 8.000, 12.000, 16.000, 20.000, 24.000, 28.000])
    """
    validators.is_valid_boundaries_tensor(
        starts
    )  # 1-d, non-negative, strictly increasing
    validators.is_valid_boundaries_tensor(stops)
    validators.have_same_dtype(starts, stops)
    if not starts.size == stops.size:
        raise ValueError(
            "Boundary tensors ``starts`` and ``stops`` must have same lengths--"
            "every element in ``starts`` must have a corresponding element in ``stops``--"
            f"but sizes were different: starts.size={starts.size}, stops.size={stops.size}"
        )
    if not torch.all(starts < stops):
        gt = torch.where(starts > stops)[0]
        raise ValueError(
            "Every element in ``starts`` must be less than the corresponding element in ``stops``,"
            f"but some values in ``starts`` were greater: values at indices {gt}"
        )

    return torch.sort(torch.concatenate((starts, stops)))
