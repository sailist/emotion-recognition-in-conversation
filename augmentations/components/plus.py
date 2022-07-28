import random

import torch
from torchvision import transforms
from PIL.Image import Image


class FixRotate:
    def __init__(self, v):
        self.v = v

    def __call__(self, img):
        return img.rotate(self.v)


class RandomRotation:
    def __init__(self, max_degree=90):
        self.max_degree = max_degree

    def __call__(self, img: Image):
        v = int(random.random() * self.max_degree)
        if random.random() > 0.5:
            v = -v
        return img.rotate(v)


class RandomMasked:
    def __init__(self, ratio=0.5, value=0):
        self.ratio = ratio
        self.val = value

    def __call__(self, img: torch.Tensor):
        if self.val == 0:
            return img

        mask = torch.rand_like(img) < self.ratio
        img[mask] = self.val
        return img


def identity(x):
    return x


def AutoResize(resize):
    if resize is None:
        return identity
    else:
        return transforms.Resize(resize)
