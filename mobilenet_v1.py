#!/usr/bin/env python3
# coding: utf-8

from __future__ import division

""" 
Creates a MobileNet Model as defined in:
Andrew G. Howard Menglong Zhu Bo Chen, et.al. (2017). 
MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications. 
Copyright (c) Yang Lu, 2017

Modified By cleardusk
"""
import math
import torch
import torch.nn as nn

__all__ = ['mobilenet_2', 'mobilenet_1', 'mobilenet_075', 'mobilenet_05', 'mobilenet_025']


class DepthWiseBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, prelu=False):
        super(DepthWiseBlock, self).__init__()
        inplanes, planes = int(inplanes), int(planes)
        self.conv_dw = nn.Conv2d(inplanes, inplanes, kernel_size=3, padding=1, stride=stride, groups=inplanes,
                                 bias=False)
        self.bn_dw = nn.BatchNorm2d(inplanes)
        self.conv_sep = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_sep = nn.BatchNorm2d(planes)
        if prelu:
            self.relu = nn.PReLU()
        else:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv_dw(x)
        out = self.bn_dw(out)
        out = self.relu(out)

        out = self.conv_sep(out)
        out = self.bn_sep(out)
        out = self.relu(out)

        return out


class MobileNet(nn.Module):
    def __init__(self, widen_factor=1.0, num_classes=1000, prelu=False, input_channel=3):
        """ Constructor
        Args:
            widen_factor: config of widen_factor
            num_classes: number of classes
        """
        super(MobileNet, self).__init__()

        block = DepthWiseBlock
        self.conv1 = nn.Conv2d(input_channel, int(32 * widen_factor), kernel_size=3, stride=2, padding=1,
                               bias=False)

        self.bn1 = nn.BatchNorm2d(int(32 * widen_factor))
        if prelu:
            self.relu = nn.PReLU()
        else:
            self.relu = nn.ReLU(inplace=True)

        self.dw2_1 = block(32 * widen_factor, 64 * widen_factor, prelu=prelu)
        self.dw2_2 = block(64 * widen_factor, 128 * widen_factor, stride=2, prelu=prelu)

        self.dw3_1 = block(128 * widen_factor, 128 * widen_factor, prelu=prelu)
        self.dw3_2 = block(128 * widen_factor, 256 * widen_factor, stride=2, prelu=prelu)

        self.dw4_1 = block(256 * widen_factor, 256 * widen_factor, prelu=prelu)
        self.dw4_2 = block(256 * widen_factor, 512 * widen_factor, stride=2, prelu=prelu)

        self.dw5_1 = block(512 * widen_factor, 512 * widen_factor, prelu=prelu)
        self.dw5_2 = block(512 * widen_factor, 512 * widen_factor, prelu=prelu)
        self.dw5_3 = block(512 * widen_factor, 512 * widen_factor, prelu=prelu)
        self.dw5_4 = block(512 * widen_factor, 512 * widen_factor, prelu=prelu)
        self.dw5_5 = block(512 * widen_factor, 512 * widen_factor, prelu=prelu)
        self.dw5_6 = block(512 * widen_factor, 1024 * widen_factor, stride=2, prelu=prelu)

        self.dw6 = block(1024 * widen_factor, 1024 * widen_factor, prelu=prelu)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(int(1024 * widen_factor), num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)  # 3  32  (120 -> 60)
        x = self.bn1(x)    # 32 32
        x = self.relu(x)

        x = self.dw2_1(x)  # 32   64    (60 -> 60)
        x = self.dw2_2(x)  # 64   128   (60 -> 30)

        x = self.dw3_1(x)  # 128  128   (30 -> 30)
        x = self.dw3_2(x)  # 128  256   (30 -> 15)

        x = self.dw4_1(x)  # 256  256   (15 -> 15)
        x = self.dw4_2(x)  # 256  512   (8 -> 8)

        x = self.dw5_1(x)  # 512  512   (8 -> 8)
        x = self.dw5_2(x)  # 512  512   (8 -> 8)
        x = self.dw5_3(x)  # 512  512   (8 -> 8)
        x = self.dw5_4(x)  # 512  512   (8 -> 8)
        x = self.dw5_5(x)  # 512  512   (8 -> 8)
        x = self.dw5_6(x)  # 512  1024  (8 -> 4)

        x = self.dw6(x)    # 1024 1024  (4 -> 4)

        x = self.avgpool(x)  # 1024 1024  (4 -> 1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)  # 1024 62
        return x


class FullConnectBlock(nn.Module):
    def __init__(self, inplane, outplane):
        super(FullConnectBlock, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(inplane, outplane)

    def forward(self, x):
        out = self.avgpool(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class MyNet(nn.Module):
    def __init__(self, widen_factor=1.0, num_classes=1000, prelu=False, input_channel=3):
        """ Constructor
        Args:
            widen_factor: config of widen_factor
            num_classes: number of classes
        """
        super(MyNet, self).__init__()

        block = DepthWiseBlock
        self.conv1 = nn.Conv2d(input_channel, int(32 * widen_factor), kernel_size=3, stride=2, padding=1,
                               bias=False)

        self.bn1 = nn.BatchNorm2d(int(32 * widen_factor))
        if prelu:
            self.relu = nn.PReLU()
        else:
            self.relu = nn.ReLU(inplace=True)

        self.dw2_1 = block(32 * widen_factor, 64 * widen_factor, prelu=prelu)
        self.dw2_2 = block(64 * widen_factor, 128 * widen_factor, stride=2, prelu=prelu)

        self.dw3_1 = block(128 * widen_factor, 128 * widen_factor, prelu=prelu)
        self.dw3_2 = block(128 * widen_factor, 256 * widen_factor, stride=2, prelu=prelu)

        self.dw4_1 = block(256 * widen_factor, 256 * widen_factor, prelu=prelu)
        self.dw4_2 = block(256 * widen_factor, 512 * widen_factor, stride=2, prelu=prelu)

        self.dw5_1 = block(512 * widen_factor, 512 * widen_factor, prelu=prelu)
        self.dw5_2 = block(512 * widen_factor, 512 * widen_factor, prelu=prelu)
        self.dw5_3 = block(512 * widen_factor, 512 * widen_factor, prelu=prelu)
        self.dw5_4 = block(512 * widen_factor, 512 * widen_factor, prelu=prelu)
        self.dw5_5 = block(512 * widen_factor, 512 * widen_factor, prelu=prelu)
        self.dw5_6 = block(512 * widen_factor, 1024 * widen_factor, stride=2, prelu=prelu)

        self.dw6 = block(1024 * widen_factor, 1024 * widen_factor, prelu=prelu)

        self.fc1 = FullConnectBlock(128, 12)
        self.fc2 = FullConnectBlock(512, 40)
        self.fc3 = FullConnectBlock(1024, 10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)  # 3  32  (120 -> 60)
        x = self.bn1(x)    # 32 32
        x = self.relu(x)

        x = self.dw2_1(x)  # 32   64    (60 -> 60)
        x = self.dw2_2(x)  # 64   128   (60 -> 30)

        x1 = self.fc1(x)

        x = self.dw3_1(x)  # 128  128   (30 -> 30)
        x = self.dw3_2(x)  # 128  256   (30 -> 15)
        x = self.dw4_1(x)  # 256  256   (15 -> 15)
        x = self.dw4_2(x)  # 256  512   (8 -> 8)

        x2 = self.fc2(x)

        x = self.dw5_1(x)  # 512  512   (8 -> 8)
        x = self.dw5_2(x)  # 512  512   (8 -> 8)
        x = self.dw5_3(x)  # 512  512   (8 -> 8)
        x = self.dw5_4(x)  # 512  512   (8 -> 8)
        x = self.dw5_5(x)  # 512  512   (8 -> 8)
        x = self.dw5_6(x)  # 512  1024  (8 -> 4)
        x = self.dw6(x)    # 1024 1024  (4 -> 4)

        x3 = self.fc3(x)
        out = torch.cat([x1, x2, x3], dim=1)
        return out


def mynet(num_classes=62, input_channel=3):
    model = MyNet(widen_factor=1.0, num_classes=num_classes, input_channel=input_channel)
    return model


def mobilenet(widen_factor=1.0, num_classes=1000):
    """
    Construct MobileNet.
    widen_factor=1.0  for mobilenet_1
    widen_factor=0.75 for mobilenet_075
    widen_factor=0.5  for mobilenet_05
    widen_factor=0.25 for mobilenet_025
    """
    model = MobileNet(widen_factor=widen_factor, num_classes=num_classes)
    return model


def mobilenet_2(num_classes=62, input_channel=3):
    model = MobileNet(widen_factor=2.0, num_classes=num_classes, input_channel=input_channel)
    return model


def mobilenet_1(num_classes=62, input_channel=3):
    model = MobileNet(widen_factor=1.0, num_classes=num_classes, input_channel=input_channel)
    return model


def mobilenet_075(num_classes=62, input_channel=3):
    model = MobileNet(widen_factor=0.75, num_classes=num_classes, input_channel=input_channel)
    return model


def mobilenet_05(num_classes=62, input_channel=3):
    model = MobileNet(widen_factor=0.5, num_classes=num_classes, input_channel=input_channel)
    return model


def mobilenet_025(num_classes=62, input_channel=3):
    model = MobileNet(widen_factor=0.25, num_classes=num_classes, input_channel=input_channel)
    return model
