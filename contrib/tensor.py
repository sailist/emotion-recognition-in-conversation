import torch
import numpy as np
from typing import List


def sharpen(x: torch.Tensor, T=0.5):
    """
    让概率分布变的更 sharp，即倾向于 onehot
    :param x: prediction, sum(x,dim=-1) = 1
    :param T: temperature, default is 0.5
    :return:
    """
    with torch.no_grad():
        temp = torch.pow(x, 1 / T)
        return temp / (temp.sum(dim=1, keepdims=True) + 1e-7)


def mixup(imgs: torch.Tensor, targets: torch.Tensor,
          beta=0.75, reids=None, target_b=None):
    """
    普通的mixup操作
    """
    if reids is not None:
        idx = reids
    else:
        idx = torch.randperm(imgs.size(0))

    input_a, input_b = imgs, imgs[idx]
    target_a = targets
    if target_b is None:
        target_b = targets[idx]
    else:
        target_b = target_b[idx]

    l = np.random.beta(beta, beta)
    l = np.max([l, 1 - l], axis=0)
    l = torch.tensor(l, device=input_a.device, dtype=torch.float)
    # torch.tensordot()

    # mixed_input = l * input_a + (1 - l) * input_b
    mixed_input = l * input_a + (1 - l) * input_b
    # mixed_target = l * target_a + (1 - l) * target_b
    mixed_target = l * target_a + (1 - l) * target_b

    return mixed_input, mixed_target


def mixmatch_up(sup_imgs: torch.Tensor, un_sup_imgs: List[torch.Tensor],
                sup_targets: torch.Tensor, un_targets: torch.Tensor,
                beta=0.75):
    """
    使用过MixMatch的方法对有标签和无标签数据进行mixup混合

    注意其中 un_sup_imgs 是一个list，包含K次增广图片batch
    而 un_targets 则只是一个 tensor，代表所有k次增广图片的标签
    """
    imgs = torch.cat((sup_imgs, *un_sup_imgs))
    targets = torch.cat([sup_targets, *[un_targets for _ in range(len(un_sup_imgs))]])
    return mixup(imgs, targets, beta)


def label_guesses(*logits):
    """根据K次增广猜测"""
    with torch.no_grad():
        k = len(logits)
        un_logits = torch.cat(logits)  # type:torch.Tensor
        targets_u = torch.softmax(un_logits, dim=1) \
                        .view(k, -1, un_logits.shape[-1]) \
                        .sum(dim=0) / k
        targets_u = targets_u.detach()
        return targets_u


def metric_knn(feature, ys, topk=5):
    from lumo.contrib.nn.functional import batch_cosine_similarity

    sim = batch_cosine_similarity(feature, feature)
    indices = sim.topk(topk, dim=-1).indices[:, 1:]  # skip the first which is sample itself
    return (ys[indices] == ys[:, None])
