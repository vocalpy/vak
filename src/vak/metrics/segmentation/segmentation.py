import numpy.typing as npt
import torch
import torchmetrics
import vocalpy as voc


class Precision(torchmetrics.Metric):
    """Compute precision :math:`P` for a segmentation.

    Computes the metric from a hypothesized vector of boundaries
    ``hypothesis`` returned by a segmentation algorithm
    and a reference vector of boundaries ``reference``,
    e.g., boundaries cleaned by a human expert
    or boundaries from a benchmark dataset.

    Precision is defined as the number of true positives (:math:`T_p`)
    over the number of true positives
    plus the number of false positives (:math:`F_p`).

    :math:`P = \\frac{T_p}{T_p+F_p}`.

    Attributes
    ----------
    tolerance : float or int
        Tolerance, in seconds.
        Elements in ``hypothesis`` are considered
        a true positive if they are within a time interval
        around any reference boundary :math:`t_0`
        in ``reference`` plus or minus
        the ``tolerance``, i.e.,
        if a hypothesized boundary :math:`t_h`
        is within the interval
        :math:`t_0 - \Delta t < t < t_0 + \Delta t`.
        Default is None,
        in which case it is set to ``0``
        (either float or int, depending on the
        dtype of ``hypothesis`` and ``reference``).
    decimals: int
        The number of decimal places to round both
        ``hypothesis`` and ``reference`` to, using
        :func:`numpy.round`. This mitigates inflated
        error rates due to floating point error.
        Rounding is only applied
        if both ``hypothesis`` and ``reference``
        are floating point values. To avoid rounding,
        e.g. to compute strict precision and recall,
        pass in the value ``False``. Default is 3, which
        assumes that the values are in seconds
        and should be rounded to milliseconds.
    """
    def __init__(self, tolerance: float | int | None = None, decimals: int | bool | None = None, **kwargs):
        super().__init__(**kwargs)
        self.tolerance = tolerance
        self.decimals = decimals
        self.add_state("n_tp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("n_detected", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, hypothesis: npt.NDArray, reference: npt.NDArray) -> None:
        _, hits_hyp, _ = voc.metrics.segmentation.ir.find_hits(
            hypothesis, reference, tolerance=self.tolerance, decimals=self.decimals
        )
        n_tp = hits_hyp.size
        self.n_tp += torch.tensor(n_tp)
        self.n_detected += torch.tensor(hypothesis.size)

    def compute(self) -> torch.Tensor:
        return self.n_tp.float() / self.n_detected


class Recall(torchmetrics.Metric):
    """Compute recall :math:`R` for a segmentation.

    Computes the metric from a hypothesized vector of boundaries
    ``hypothesis`` returned by a segmentation algorithm
    and a reference vector of boundaries ``reference``,
    e.g., boundaries cleaned by a human expert
    or boundaries from a benchmark dataset.

    Recall (:math:`R`) is defined as the number of true positives (:math:`T_p`)
    over the number of true positives plus the number of false negatives
    (:math:`F_n`).

    :math:`R = \\frac{T_p}{T_p + F_n}`

    Attributes
    ----------
    tolerance : float or int
        Tolerance, in seconds.
        Elements in ``hypothesis`` are considered
        a true positive if they are within a time interval
        around any reference boundary :math:`t_0`
        in ``reference`` plus or minus
        the ``tolerance``, i.e.,
        if a hypothesized boundary :math:`t_h`
        is within the interval
        :math:`t_0 - \Delta t < t < t_0 + \Delta t`.
        Default is None,
        in which case it is set to ``0``
        (either float or int, depending on the
        dtype of ``hypothesis`` and ``reference``).
    decimals: int
        The number of decimal places to round both
        ``hypothesis`` and ``reference`` to, using
        :func:`numpy.round`. This mitigates inflated
        error rates due to floating point error.
        Rounding is only applied
        if both ``hypothesis`` and ``reference``
        are floating point values. To avoid rounding,
        e.g. to compute strict precision and recall,
        pass in the value ``False``. Default is 3, which
        assumes that the values are in seconds
        and should be rounded to milliseconds.
    """
    def __init__(self, tolerance: float | int | None = None, decimals: int | bool | None = None, **kwargs):
        super().__init__(**kwargs)
        self.tolerance = tolerance
        self.decimals = decimals
        self.add_state("n_tp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("n_relevant", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, hypothesis: npt.NDArray, reference: npt.NDArray) -> None:
        _, hits_hyp, _ = voc.metrics.segmentation.ir.find_hits(
            hypothesis, reference, tolerance=self.tolerance, decimals=self.decimals
        )
        n_tp = hits_hyp.size
        self.n_tp += torch.tensor(n_tp)
        self.n_relevant += torch.tensor(reference.size)

    def compute(self) -> torch.Tensor:
        return self.n_tp.float() / self.n_relevant


class FScore(torchmetrics.Metric):
    """Compute the F-score for a segmentation.

    Computes the metric from a
    hypothesized vector of boundaries ``hypothesis``
    returned by a segmentation algorithm
    and a reference vector of boundaries ``reference``,
    e.g., boundaries cleaned by a human expert
    or boundaries from a benchmark dataset.

    The F-score can be interpreted as a harmonic mean of the precision and
    recall, where an F-score reaches its best value at 1 and worst score at 0.
    The relative contribution of precision and recall to the F-score are
    equal. The formula for the F-score is:

    ``f_score = 2 * (precision * recall) / (precision + recall)``


    Attributes
    ----------
    tolerance : float or int
        Tolerance, in seconds.
        Elements in ``hypothesis`` are considered
        a true positive if they are within a time interval
        around any reference boundary :math:`t_0`
        in ``reference`` plus or minus
        the ``tolerance``, i.e.,
        if a hypothesized boundary :math:`t_h`
        is within the interval
        :math:`t_0 - \Delta t < t < t_0 + \Delta t`.
        Default is None,
        in which case it is set to ``0``
        (either float or int, depending on the
        dtype of ``hypothesis`` and ``reference``).
    decimals: int
        The number of decimal places to round both
        ``hypothesis`` and ``reference`` to, using
        :func:`numpy.round`. This mitigates inflated
        error rates due to floating point error.
        Rounding is only applied
        if both ``hypothesis`` and ``reference``
        are floating point values. To avoid rounding,
        e.g. to compute strict precision and recall,
        pass in the value ``False``. Default is 3, which
        assumes that the values are in seconds
        and should be rounded to milliseconds.


    Examples
    --------
    >>> hypothesis = np.array([1, 6, 10, 16])
    >>> reference = np.array([0, 5, 10, 15])
    >>> prec, n_tp, ir_metric_data = vocalpy.metrics.segmentation.ir.fscore(hypothesis, reference, tolerance=0)
    >>> print(prec)
    0.25
    >>> print(ir_metric_data.hits_hyp)
    np.array([2])

    >>> hypothesis = np.array([0, 1, 5, 10])
    >>> reference = np.array([0, 5, 10])
    >>> prec, n_tp, metric_data = vocalpy.metrics.segmentation.ir.fscore(hypothesis, reference, tolerance=1)
    >>> print(prec)
    0.75
    >>> print(ir_metric_data.hits_hyp)
    np.array([0, 2, 3])
    """
    def __init__(self, tolerance: float | int | None = None, decimals: int | bool | None = None, **kwargs):
        super().__init__(**kwargs)
        self.tolerance = tolerance
        self.decimals = decimals
        self.add_state("n_tp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("n_detected", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("n_relevant", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, hypothesis: npt.NDArray, reference: npt.NDArray) -> None:
        _, hits_hyp, _ = voc.metrics.segmentation.ir.find_hits(
            hypothesis, reference, tolerance=self.tolerance, decimals=self.decimals
        )
        n_tp = hits_hyp.size
        self.n_tp += torch.tensor(n_tp)
        self.n_detected += torch.tensor(hypothesis.size)
        self.n_relevant += torch.tensor(reference.size)

    def compute(self) -> torch.Tensor:
        precision = self.n_tp.float() / self.n_detected
        recall = self.n_tp.float() / self.n_relevant
        return 2 * (precision * recall) / (precision * recall)
