# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

#### Resnet stuff
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        
    def forward(self, x):        
        return x.view(x.size(0), -1)
    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, zero_init_residual=False, iSz=224, flatten=True):
        super(ResNet, self).__init__()
        self.inplanes = 64
        if iSz == 84:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.maxpool = nn.Sequential()
        elif iSz == 224:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            raise ValueError(f'iSz was {iSz}')
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if flatten:
            self.flatten = nn.Sequential(*[
                nn.AdaptiveAvgPool2d((1,1)),
                Flatten()
            ])
        else:
            self.flatten = nn.Sequential()
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
        
        # print total number of parameters
        print(f'| total num of parameters: {sum([p.numel() for p in self.parameters()])}')

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x) # if iSz ==224 kernel size=7 and stride =2, else kernel size = 3 stride =1
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) # only if iSz == 224
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.flatten(x)
#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
        return x
    
def get_output_size(m, iSz, flatten):
    x = torch.rand(2,3,iSz,iSz)
    y = m(x)
    if flatten:
        m.final_feat_dim = y.size(1)
    else:
        m.final_feat_dim = [y.size(i) for i in range(1, y.dim())]
    print('Output feat dim: ', m.final_feat_dim)
    
def ResNet18(flatten=True):
    m = ResNet(BasicBlock, [2, 2, 2, 2], iSz=84, flatten=flatten)
    get_output_size(m, 84, flatten)
    return m
def ResNet18_224(flatten=True):
    m = ResNet(BasicBlock, [2, 2, 2, 2], iSz=224, flatten=flatten)
    get_output_size(m, 224, flatten)
    return m
def ResNet10(flatten=True):
    m = ResNet(BasicBlock, [1, 1, 1, 1], iSz=84, flatten=flatten)
    get_output_size(m, 84, flatten)
    return m
def ResNet10_224(flatten=True):
    m = ResNet(BasicBlock, [1, 1, 1, 1], iSz=224, flatten=flatten)
    get_output_size(m, 224, flatten)
    return m

def ResNet34_224(flatten=True):
    m = ResNet(BasicBlock, [3, 4, 6, 3], iSz=224, flatten=flatten)
    get_output_size(m, 224, flatten)
    return m

def ResNet50_224(flatten=True):
    m = ResNet(Bottleneck, [3, 4, 6, 3], iSz=224, flatten=flatten)
    get_output_size(m, 224, flatten)
    return m

def WRN(flatten=True):
    from architectures import WideResNet
    m = WideResNet(flatten=flatten)
    get_output_size(m, 84, flatten)
    return m