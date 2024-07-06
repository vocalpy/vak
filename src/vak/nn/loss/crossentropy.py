import torch


class CrossEntropyLoss(torch.nn.CrossEntropyLoss):
    """Wrapper around :class:`torch.nn.CrossEntropyLoss`

    Converts the argument ``weight`` to a :class:`torch.Tensor`
    if it is a :class:`list`.
    """

    def __init__(
        self,
        weight=None,
        size_average=None,
        ignore_index=-100,
        reduce=None,
        reduction="mean",
        label_smoothing=0.0,
    ):

        if weight is not None:
            if isinstance(weight, torch.Tensor):
                pass
            elif isinstance(weight, list):
                weight = torch.Tensor(weight)
        super().__init__(
            weight,
            size_average,
            ignore_index,
            reduce,
            reduction,
            label_smoothing,
        )
