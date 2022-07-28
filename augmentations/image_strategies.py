from typing import Union

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as VF

from .components.plus import FixRotate, RandomRotation, RandomMasked, AutoResize
from .components.randaugment import RandAugment


def read(x):
    return Image.open(x).convert('RGB')


def basic(mean, std, size=32, resize=None):
    return transforms.Compose([
        AutoResize(resize),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])


def none(mean, std, size=32):
    return transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])


def standard(mean, std, size=32, resize=None):
    return transforms.Compose([
        AutoResize(size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size, padding=4, padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])


def standard_resize(mean, std, size=32, resize=None):
    return transforms.Compose([
        AutoResize(resize),
        transforms.RandomCrop(size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])


def standard_rotate(mean, std, size=32, resize=None):
    return transforms.Compose([
        AutoResize(resize),
        transforms.RandomCrop(size, padding=4),
        RandomRotation(90),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])


def rotate(mean, std, size=32, v=0):
    return transforms.Compose([
        transforms.CenterCrop(size),
        FixRotate(v),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])


def standard_multi_crop(mean, std, size=32, index=0, resize=None):
    return transforms.Compose([
        AutoResize(resize),
        MultiCrop(index=index, size=size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])


def simclr(mean, std, size=32, scale=(0.08, 1.0), resize=None):
    return transforms.Compose([
        AutoResize(resize),
        transforms.RandomResizedCrop(size, scale=scale),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])


def simclr_randmask(mean, std, size=32, scale=[0.2, 1.0], ratio=0.5, resize=None):
    return transforms.Compose([
        AutoResize(resize),
        transforms.RandomResizedCrop(size, scale=scale),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        RandomMasked(ratio)
    ])


def randaugment(mean, std, size=32, resize=None, nm=(3, 5)):
    return transforms.Compose([
        AutoResize(size),
        RandAugment(n=nm[0], m=nm[1]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size, padding=size // 8, padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


class MultiCrop:
    def __init__(self, index, size=32):
        self.index = index
        if isinstance(size, int):
            size = (size, size)
        self.ch, self.cw = size

    def get_params(self, ih, iw):
        image_height, image_width, crop_height, crop_width = ih, iw, self.ch, self.cw

        param = [
            lambda: (VF.center_crop, crop_height, crop_width),
            lambda: (VF.crop, 0, 0, crop_width, crop_height),
            lambda: (VF.crop, image_width - crop_width, 0, image_width, crop_height),
            lambda: (VF.crop, 0, image_height - crop_height, crop_width, image_height),
            lambda: (VF.crop, image_width - crop_width, image_height - crop_height, image_width, image_height),
        ][self.index % 5]
        return param

    def __call__(self, x: Union[Image.Image, torch.Tensor]):
        if isinstance(x, torch.Tensor):
            ih, iw = x.shape[:2]
        elif isinstance(x, Image.Image):
            ih, iw = x.height, x.width
        else:
            raise NotImplementedError()

        if self.index >= 5:
            x = VF.hflip(x)

        func, *args = self.get_params(ih, iw)

        return func(x, *args)
