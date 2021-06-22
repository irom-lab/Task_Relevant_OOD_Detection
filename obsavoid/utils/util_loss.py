import torch.nn as nn
import torch

class AsymmetricBCELoss(nn.BCELoss):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean', asym_scale=1):
        super().__init__(weight, size_average, reduce, reduction)
        self.scale = asym_scale

    def __call__(self, x, y):
        c1 = x >= 0
        c2 = x <= 1
        c3 = y >= 0
        c4 = y <= 1
        if not (torch.min(c1) and torch.min(c2) and torch.min(c3) and torch.min(c4)):
            raise Exception("Inputs need to be between 0 and 1")

        t1 = self.scale * torch.multiply(- y, torch.log(x))
        t2 = torch.multiply(y - 1, torch.log(1-x))

        if self.reduction == 'mean':
            return torch.mean(t1 + t2)
        elif self.reduction == 'sum':
            return torch.sum(t1 + t2)
        elif self.reduction == 'none':
            return t1 + t2
        else:
            raise NotImplementedError