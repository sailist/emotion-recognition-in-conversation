"""
Usually be used as reconstruction loss.
"""
from torch.nn import functional as F


def l2(a, b):
    return F.mse_loss(a, b, reduction='mean')
