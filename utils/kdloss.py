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


class KLDivLoss(_Loss):

    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'batchmean', log_target: bool = False) -> None:
        super(KLDivLoss, self).__init__(size_average, reduce, reduction)
        self.log_target = log_target

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.kl_div(input, target, reduction=self.reduction, log_target=self.log_target)


class KDLoss(_WeightedLoss):
    
    __constants__ = ['ignore_index', 'reduction', 'label_smoothing']
    ignore_index: int
    label_smoothing: float

    def __init__(self, weight: Optional[Tensor] = None, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean', label_smoothing: float = 0.0,
                 alpha: float = 1.0, temperature: float = 1.0) -> None:

        super(KDLoss, self).__init__(weight, size_average, reduce, reduction)
        self.alpha = alpha
        self.T = temperature
        self.KLDivLoss = KLDivLoss(reduction='batchmean')

    def forward(self, s_input: Tensor, t_input: Tensor, target: Tensor) -> Tensor:
        return self.KLDivLoss(F.log_softmax(s_input/self.T, dim=1),
                             F.softmax(t_input/self.T, dim=1)) * (self.alpha * self.T * self.T) + \
                                    F.cross_entropy(s_input, target) * (1. - self.alpha)


if __name__ == "__main__":

    import torch
    import torch.nn as nn

    loss = KDLoss()
    m = nn.Softmax()

    s_input = torch.randn((5, 2, 256, 256), requires_grad=True)
    t_input = torch.rand((5, 2, 256, 256), requires_grad=True)
    target = torch.randint(0, 2, (5, 256, 256))

    print("s_input: \t{}".format(s_input.size()))
    print("t_input: \t{}".format(t_input.size()))
    print("target: \t{}".format(target.size()))
    print(loss(s_input, t_input, target))
