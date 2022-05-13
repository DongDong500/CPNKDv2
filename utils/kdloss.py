import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.loss import _WeightedLoss
from torch.nn import _reduction as _Reduction
from typing import Optional

class _Loss(nn.Module):
    reduction: str

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction: str = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction


class _WeightedLoss(_Loss):
    def __init__(self, weight: Optional[Tensor] = None, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(_WeightedLoss, self).__init__(size_average, reduce, reduction)
        self.register_buffer('weight', weight)
        self.weight: Optional[Tensor]


class KDLoss(_WeightedLoss):
    
    __constants__ = ['ignore_index', 'reduction', 'label_smoothing']
    ignore_index: int
    label_smoothing: float

    def __init__(self, weight: Optional[Tensor] = None, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean', label_smoothing: float = 0.0, params: float = 1.0) -> None:
        super(KDLoss, self).__init__(weight, size_average, reduce, reduction)

        self.alpha = params.alpha
        self.T = params.temperature

    def forward(self, input: Tensor, target: Tensor, t_input: Tensor) -> Tensor:
        return nn.KLDivLoss()(F.log_softmax(input/self.T, dim=1),
                             F.softmax(t_input/self.T, dim=1)) * (self.alpha * self.T * self.T) + \
                                    F.cross_entropy(input, target) * (1. - self.alpha)