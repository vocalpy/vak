import numpy.typing as npt
import torch
import torchmetrics
import vocalpy as voc


class Precision(torchmetrics.Metric):
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
