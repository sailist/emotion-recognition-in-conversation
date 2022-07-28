import torch
import torch.nn as nn
import torch.nn.functional as F


def focal_loss(input_, target, gamma=2, alpha=None, ignore_index=-100, reduction='mean'):
    cross_entropy = F.cross_entropy(input_, target, weight=alpha, ignore_index=ignore_index, reduction='none')
    # Temporarily mask out ignore index to '0' for valid gather-indices input.
    # This won't contribute final loss as the cross_entropy contribution
    # for these would be zero.
    target = target * (target != ignore_index).long()
    input_prob = torch.gather(F.softmax(input_, 1), 1, target.unsqueeze(1))
    loss = torch.pow(1 - input_prob, gamma) * cross_entropy
    return torch.mean(loss) if reduction == 'mean' else \
        torch.sum(loss) if reduction == 'sum' else loss


class FocalLoss(nn.CrossEntropyLoss):
    """ Focal loss for classification tasks on imbalanced datasets """

    def __init__(self, gamma=2, alpha=None, ignore_index=-100, reduction='none'):
        super().__init__(weight=alpha, ignore_index=ignore_index, reduction='none')
        self.reduction = reduction
        self.gamma = gamma

    def forward(self, input_, target):
        return focal_loss(input_, target, gamma=self.gamma, ignore_index=self.ignore_index, reduction=self.reduction)
