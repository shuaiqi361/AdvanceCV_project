import torch
import torch.nn as nn
import torch.nn.functional as F


# class FocalLoss(nn.Module):
#     def __init__(self, alpha=1, gamma=2, logits=False, reduce=True, reduction='mean'):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.logits = logits
#         self.reduce = reduce
#
#     def forward(self, inputs, targets):
#         if self.logits:
#             BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
#         else:
#             BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
#         pt = torch.exp(-BCE_loss)
#         F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
#
#         if self.reduce:
#             return torch.mean(F_loss)
#         else:
#             return F_loss

class FocalLoss(nn.Module):
    'Focal Loss - https://arxiv.org/abs/1708.02002'

    def __init__(self, alpha=0.25, gamma=2, size_average=True):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.average = size_average

    def forward(self, pred_logits, target):
        pred = pred_logits.sigmoid()
        ce = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='none')
        alpha = target * self.alpha + (1. - target) * (1. - self.alpha)
        pt = torch.where(target == 1,  pred, 1 - pred)
        focal_loss = alpha * (1. - pt) ** self.gamma * ce
        if self.average:
            return focal_loss.mean()
        else:
            return focal_loss

class SmoothL1Loss(nn.Module):
    'Smooth L1 Loss'

    def __init__(self, beta=0.11, size_average=True):
        super().__init__()
        self.beta = beta
        self.average = size_average

    def forward(self, pred, target):
        x = (pred - target).abs()
        l1 = x - 0.5 * self.beta
        l2 = 0.5 * x ** 2 / self.beta
        l1_loss = torch.where(x >= self.beta, l1, l2)
        if self.average:
            return l1_loss.mean()
        else:
            return l1_loss