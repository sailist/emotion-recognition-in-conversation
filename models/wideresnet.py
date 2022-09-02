# !/usr/bin/python
# -*- encoding: utf-8 -*-
import math

import torch
import torch.nn as nn

import torch.nn.functional as F

from torch.nn import BatchNorm2d

'''
    As in the paper, the wide resnet only considers the resnet of the pre-activated version, 
    and it only considers the basic blocks rather than the bottleneck blocks.
'''


class BasicBlockPreAct(nn.Module):
    def __init__(self, in_chan, out_chan, drop_rate=0, stride=1, pre_res_act=False):
        super(BasicBlockPreAct, self).__init__()
        self.bn1 = BatchNorm2d(in_chan, momentum=0.001)
        self.relu1 = nn.LeakyReLU(inplace=True, negative_slope=0.1)
        self.conv1 = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = BatchNorm2d(out_chan, momentum=0.001)
        self.relu2 = nn.LeakyReLU(inplace=True, negative_slope=0.1)
        self.dropout = nn.Dropout(drop_rate) if not drop_rate == 0 else None
        self.conv2 = nn.Conv2d(out_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
        self.downsample = None
        if in_chan != out_chan or stride != 1:
            self.downsample = nn.Conv2d(
                in_chan, out_chan, kernel_size=1, stride=stride, bias=False
            )
        self.pre_res_act = pre_res_act
        # self.init_weight()

    def forward(self, x):
        bn1 = self.bn1(x)
        act1 = self.relu1(bn1)
        residual = self.conv1(act1)
        residual = self.bn2(residual)
        residual = self.relu2(residual)
        if self.dropout is not None:
            residual = self.dropout(residual)
        residual = self.conv2(residual)

        shortcut = act1 if self.pre_res_act else x
        if self.downsample is not None:
            shortcut = self.downsample(shortcut)

        out = shortcut + residual
        return out

    def init_weight(self):
        # for _, md in self.named_modules():
        #     if isinstance(md, nn.Conv2d):
        #         nn.init.kaiming_normal_(
        #             md.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        #         if md.bias is not None:
        #             nn.init.constant_(md.bias, 0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class WideResnet(nn.Module):
    def __init__(self, k=2, n=28, drop_rate=0):
        super(WideResnet, self).__init__()
        self.k, self.n = k, n
        assert (self.n - 4) % 6 == 0
        n_blocks = (self.n - 4) // 6  # =4
        n_layers = [16, ] + [self.k * 16 * (2 ** i) for i in range(3)]

        self.conv1 = nn.Conv2d(
            3,
            n_layers[0],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.layer1 = self.create_layer(
            n_layers[0],
            n_layers[1],
            bnum=n_blocks,
            stride=1,
            drop_rate=drop_rate,
            pre_res_act=True,
        )
        self.layer2 = self.create_layer(
            n_layers[1],
            n_layers[2],
            bnum=n_blocks,
            stride=2,
            drop_rate=drop_rate,
            pre_res_act=False,
        )
        self.layer3 = self.create_layer(
            n_layers[2],
            n_layers[3],
            bnum=n_blocks,
            stride=2,
            drop_rate=drop_rate,
            pre_res_act=False,
        )
        self.bn_last = BatchNorm2d(n_layers[3], momentum=0.001)
        self.relu_last = nn.LeakyReLU(inplace=True, negative_slope=0.1)
        self.fc = nn.Identity()
        self.feature_dim = 64 * k
        self.avgpool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1),
        )

    def create_layer(
            self,
            in_chan,
            out_chan,
            bnum,
            stride=1,
            drop_rate=0,
            pre_res_act=False,
    ):
        layers = [
            BasicBlockPreAct(
                in_chan,
                out_chan,
                drop_rate=drop_rate,
                stride=stride,
                pre_res_act=pre_res_act), ]
        for _ in range(bnum - 1):
            layers.append(
                BasicBlockPreAct(
                    out_chan,
                    out_chan,
                    drop_rate=drop_rate,
                    stride=1,
                    pre_res_act=False, ))
        return nn.Sequential(*layers)

    def forward(self, x):
        feat = self.conv1(x)

        feat = self.layer1(feat)
        feat2 = self.layer2(feat)  # 1/2
        feat4 = self.layer3(feat2)  # 1/4

        feat4 = self.bn_last(feat4)
        feat4 = self.relu_last(feat4)

        feat4 = self.avgpool(feat4)
        return feat4


class WideResnet4(nn.Module):
    """
    used for stl10
    """

    def __init__(self, k=2, n=28, drop_rate=0):
        super(WideResnet4, self).__init__()
        self.k, self.n = k, n
        assert (self.n - 4) % 6 == 0
        n_blocks = (self.n - 4) // 6  # =4
        n_layers = [16, ] + [self.k * 16 * (2 ** i) for i in range(4)]

        self.conv1 = nn.Conv2d(
            3,
            n_layers[0],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.layer1 = self.create_layer(
            n_layers[0],
            n_layers[1],
            bnum=n_blocks,
            stride=1,
            drop_rate=drop_rate,
            pre_res_act=True,
        )
        self.layer2 = self.create_layer(
            n_layers[1],
            n_layers[2],
            bnum=n_blocks,
            stride=2,
            drop_rate=drop_rate,
            pre_res_act=False,
        )
        self.layer3 = self.create_layer(
            n_layers[2],
            n_layers[3],
            bnum=n_blocks,
            stride=2,
            drop_rate=drop_rate,
            pre_res_act=False,
        )
        self.layer4 = self.create_layer(
            n_layers[3],
            n_layers[4],
            bnum=n_blocks,
            stride=2,
            drop_rate=drop_rate,
            pre_res_act=False,
        )
        self.bn_last = BatchNorm2d(n_layers[3], momentum=0.001)
        self.relu_last = nn.LeakyReLU(inplace=True, negative_slope=0.1)
        self.fc = nn.Identity()
        self.feature_dim = 64 * k
        self.avgpool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1),
        )

    def create_layer(
            self,
            in_chan,
            out_chan,
            bnum,
            stride=1,
            drop_rate=0,
            pre_res_act=False,
    ):
        layers = [
            BasicBlockPreAct(
                in_chan,
                out_chan,
                drop_rate=drop_rate,
                stride=stride,
                pre_res_act=pre_res_act), ]
        for _ in range(bnum - 1):
            layers.append(
                BasicBlockPreAct(
                    out_chan,
                    out_chan,
                    drop_rate=drop_rate,
                    stride=1,
                    pre_res_act=False, ))
        return nn.Sequential(*layers)

    def forward(self, x):
        feat = self.conv1(x)

        feat = self.layer1(feat)
        feat2 = self.layer2(feat)  # 1/2
        feat4 = self.layer3(feat2)  # 1/4

        feat4 = self.bn_last(feat4)
        feat4 = self.relu_last(feat4)

        feat4 = self.avgpool(feat4)
        return feat4


if __name__ == "__main__":
    from lumo.utils.random import fix_seed

    fix_seed(1)
    x = torch.randn(2, 3, 72, 32)
    lb = torch.randint(0, 10, (2,)).long()

    net = WideResnet4(2, 28)
    for n, m in net.named_parameters():
        print(n, m.data.numel())
    print(net)
    # out = net(x)
    # print(out[0][:4])

