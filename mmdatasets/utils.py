"""
refered to
https://github.com/YyzHarry/imbalanced-semi-self/blob/master/dataset/imbalance_cifar.py
"""
from collections import Counter

import numpy as np


def long_tail_exp_splits(ys, imb_factor=0.01):
    ys = np.array(ys)
    cls_counter = Counter(ys.astype(int).tolist())
    cls_ids = cls_counter.keys()
    cls_num = len(cls_ids)
    img_num_per_cls = []
    indexs = []
    for cls_idx, (cls_y, img_max) in enumerate(cls_counter.items()):
        cls_sample_num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
        img_num_per_cls.append(int(cls_sample_num))

        idx = np.where(ys=cls_y)[0]
        np.random.shuffle(idx)

        indexs.extend(idx[:cls_sample_num])

    return indexs, img_num_per_cls


def long_tail_step_splits(ys, imb_factor=0.01):
    raise NotImplementedError()
    for cls_idx in range(cls_num // 2):
        img_num_per_cls.append(int(img_max))
    for cls_idx in range(cls_num // 2):
        img_num_per_cls.append(int(img_max * imb_factor))
