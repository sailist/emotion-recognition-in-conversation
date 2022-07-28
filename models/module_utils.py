from dataclasses import dataclass
from typing import List, Optional
from collections import OrderedDict
import torch
from lumo.core import BaseParams
from torch import nn


class ModelParams(BaseParams):

    def __init__(self):
        super().__init__()
        self.model = self.choice(
            'resnet18',
            'resnet34',
            'resnet50',
            'cifar_resnet18',
            'stl10_resnet18',
            'cifar_resnet50',
            'wrn282',
            'wrn288',
            'wrn372',
            'preresnet50',
        )


class ModelOutput(OrderedDict):
    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __getattribute__(self, item):
        if item in self:
            return self.__getitem__(item)
        else:
            return super(ModelOutput, self).__getattribute__(item)


class ResnetOutput(ModelOutput):
    feature_map: Optional[torch.Tensor] = None
    feature: Optional[torch.Tensor] = None
    hidden_states: Optional[List[torch.Tensor]] = None
    last_hidden_state: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None


class MemoryBank(torch.Tensor):
    def __new__(cls, queue_size=65535, feature_dim=128):
        data = torch.rand(queue_size, feature_dim)
        self = torch.Tensor._make_subclass(cls, data, False)
        self.queue_size = queue_size
        self.cursor = 0
        self.detach_()
        return self

    def to(self, *args, **kwargs):
        ncls = super(MemoryBank, self).to(*args, **kwargs)
        ncls.queue_size = self.queue_size
        ncls.cursor = self.cursor
        return ncls

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        else:
            result = type(self)(self.data.clone(memory_format=torch.preserve_format), self.requires_grad)
            result.cursor = self.cursor
            result.queue_size = self.queue_size
            memo[id(self)] = result
            return result

    def push(self, item: torch.Tensor):
        assert item.ndim == 2 and item.shape[1:] == self.shape[1:], f'ndim: {item.ndim} | shape: {item.shape}'
        with torch.no_grad():
            item = item.to(self.data.device)
            isize = len(item)
            if self.cursor + isize > self.queue_size:
                right = self.queue_size - self.cursor
                left = isize - right
                self.data[self.cursor:] = item[:right]
                self.data[:left] = item[right:]
            else:
                self.data[self.cursor:self.cursor + len(item)] = item
            self.cursor = (self.cursor + len(item)) % self.queue_size

    def tensor(self):
        return torch.Tensor._make_subclass(torch.Tensor, self.data, False)

    def clone(self, *args, **kwargs):
        ncls = super(MemoryBank, self).clone(*args, **kwargs)
        ncls.queue_size = self.queue_size
        ncls.cursor = self.cursor
        return ncls

    def half(self, memory_format=None):
        return self.to(torch.float16)


class LongTensorMemoryBank(torch.Tensor):
    def __new__(cls, queue_size=65535):
        data = torch.zeros(queue_size, dtype=torch.long)
        self = torch.Tensor._make_subclass(cls, data, False)
        self.queue_size = queue_size
        self.cursor = 0
        self.detach_()
        return self

    def to(self, *args, **kwargs):
        ncls = super(LongTensorMemoryBank, self).to(*args, **kwargs)
        ncls.queue_size = self.queue_size
        ncls.cursor = self.cursor
        return ncls

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        else:
            result = type(self)(self.data.clone(memory_format=torch.preserve_format), self.requires_grad)
            result.cursor = self.cursor
            result.queue_size = self.queue_size
            memo[id(self)] = result
            return result

    def push(self, item: torch.Tensor):
        with torch.no_grad():
            item = item.to(self.device)
            isize = len(item)
            if self.cursor + isize > self.queue_size:
                right = self.queue_size - self.cursor
                left = isize - right
                self.data[self.cursor:] = item[:right]
                self.data[:left] = item[right:]
            else:
                self.data[self.cursor:self.cursor + len(item)] = item
            self.cursor = (self.cursor + len(item)) % self.queue_size

    def tensor(self):
        return torch.Tensor._make_subclass(torch.Tensor, self.data, False)

    def clone(self, *args, **kwargs):
        ncls = super(LongTensorMemoryBank, self).clone(*args, **kwargs)
        ncls.queue_size = self.queue_size
        ncls.cursor = self.cursor
        return ncls

    def half(self, memory_format=None):
        return self.to(torch.float16)


def pick_model_name(model_name) -> nn.Module:
    from . import resnet, resnet_stl, resnet_cifar, wideresnet, preresnet

    if model_name in {'resnet18'}:
        model = resnet.resnet18()
    elif model_name in {'cifar_resnet18'}:
        model = resnet_cifar.resnet18()
    elif model_name in {'cifar_resnet50'}:
        model = resnet_cifar.resnet50()
    elif model_name in {'stl10_resnet18'}:
        model = resnet_stl.resnet18()
    elif model_name in {'resnet50'}:
        model = resnet.resnet50()
    elif model_name in {'resnet34'}:
        model = resnet.resnet34()
    elif model_name in {'wrn282'}:
        model = wideresnet.WideResnet(2, 28)
    elif model_name in {'wrn288'}:
        model = wideresnet.WideResnet(8, 28)
    elif model_name in {'wrn372'}:
        model = wideresnet.WideResnet4(2, 28)
    elif model_name in {'preresnet50'}:
        model = preresnet.PreActResNet50()
    else:
        raise NotImplementedError()

    return model
