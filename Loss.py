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

    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred_logits, target):
        pred = pred_logits.sigmoid()
        ce = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='none')
        alpha = target * self.alpha + (1. - target) * (1. - self.alpha)
        pt = torch.where(target == 1,  pred, 1 - pred)
        focal_loss = alpha * (1. - pt) ** self.gamma * ce
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class SmoothL1Loss(nn.Module):
    'Smooth L1 Loss'

    def __init__(self, beta=1.0/9.0, reduction='mean'):
        super().__init__()
        self.beta = beta
        self.reduction = reduction

    def forward(self, pred, target, weights=None):

        if weights is not None:
            # print(pred.size(), target.size(), weights.size())
            # assert pred.size(1) == target.size(1) == weights.size(0)
            x = (pred - target).abs()
            l1 = x - 0.5 * self.beta
            l2 = 0.5 * x ** 2 / self.beta
            l1_loss = torch.where(x >= self.beta, l1, l2)
            l1_loss = l1_loss / weights

            if self.reduction == 'mean':
                return l1_loss.mean()
            elif self.reduction == 'sum':
                return l1_loss.sum()
            else:
                return l1_loss

        x = (pred - target).abs()
        l1 = x - 0.5 * self.beta
        l2 = 0.5 * x ** 2 / self.beta
        l1_loss = torch.where(x >= self.beta, l1, l2)
        if self.reduction == 'mean':
            return l1_loss.mean()
        elif self.reduction == 'sum':
            return l1_loss.sum()
        else:
            return l1_loss


class SoftConstraintsLoss(nn.Module):
    'Soft constraint loss'
    '''
    1. Minimize the sum of all inter-point distances, so that the RepPoints can be as compact as possible
    2. Maximize the minimum inter-point distance so that the RepPoints can be as spread-out as possible
    3. minimize the difference between the minimum and maximum inter-point distance, so that the RepPoints can be 
    as equally distributed as possible
    '''

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred, target, weights=None):

        if weights is not None:
            # print(pred.size(), target.size(), weights.size())
            # assert pred.size(1) == target.size(1) == weights.size(0)
            x = (pred - target).abs()
            l1 = x - 0.5 * self.beta
            l2 = 0.5 * x ** 2 / self.beta
            l1_loss = torch.where(x >= self.beta, l1, l2)
            # l1_loss = l1_loss / torch.unsqueeze(weights, dim=1)
            l1_loss = l1_loss / weights

            if self.reduction == 'mean':
                return l1_loss.mean()
            elif self.reduction == 'sum':
                return l1_loss.sum()
            else:
                return l1_loss

        x = (pred - target).abs()
        l1 = x - 0.5 * self.beta
        l2 = 0.5 * x ** 2 / self.beta
        l1_loss = torch.where(x >= self.beta, l1, l2)
        if self.reduction == 'mean':
            return l1_loss.mean()
        elif self.reduction == 'sum':
            return l1_loss.sum()
        else:
            return l1_loss