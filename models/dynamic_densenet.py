from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn

import math
from layers import Conv, DynamicMultiHeadConv

__all__ = ['Dydensenet']


class _DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, args):
        super(_DenseLayer, self).__init__()
        ### 1x1 conv: i --> bottleneck * k
        self.conv_1 = DynamicMultiHeadConv(
                in_channels, args.bottleneck * growth_rate, 
                kernel_size=1, heads=args.heads, squeeze_rate=args.squeeze_rate, 
                gate_factor=args.gate_factor)

        ### 3x3 conv: bottleneck * k --> k
        self.conv_2 = Conv(args.bottleneck * growth_rate, growth_rate,
                           kernel_size=3, padding=1, groups=args.group_3x3)

    def forward(self, x):
        _lasso_loss = x[1]
        x_ = x[0]
        x, lasso_loss = self.conv_1(x[0])
        x = self.conv_2(x)
        x = torch.cat([x_, x], 1)
        _lasso_loss.append(lasso_loss)
        return [x, _lasso_loss]


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, in_channels, growth_rate, args):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(in_channels + i * growth_rate, growth_rate, args)
            self.add_module('denselayer_%d' % (i + 1), layer)


class _Transition(nn.Module):
    def __init__(self, in_channels, args):
        super(_Transition, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        _lasso_loss = x[1]
        x_ = x[0]
        x = self.pool(x_)
        return [x, _lasso_loss]

class Conv2d_lasso(nn.Conv2d):
    def forward(self, x):
        x = super(Conv2d_lasso, self).forward(x)
        return [x, []]

class DydenseNet(nn.Module):
    def __init__(self, args):

        super(DydenseNet, self).__init__()

        self.stages = args.stages
        self.growth = args.growth
        assert len(self.stages) == len(self.growth)
        self.args = args
        self.progress = 0.0
        if args.data in ['cifar10', 'cifar100']:
            self.init_stride = 1
            self.pool_size = 8
        else:
            self.init_stride = 2
            self.pool_size = 7

        self.features = nn.Sequential()
        ### Initial nChannels should be 3
        self.num_features = 2 * self.growth[0]
        ### Dense-block 1 (224x224)
        self.features.add_module('init_conv', Conv2d_lasso(3, self.num_features,
                                                        kernel_size=3,
                                                        stride=self.init_stride,
                                                        padding=1,
                                                        bias=False))
        for i in range(len(self.stages)):
            ### Dense-block i
            self.add_block(i)

        ### Linear layer
        self.bn_last = nn.BatchNorm2d(self.num_features)
        self.relu_last = nn.ReLU(inplace=True)
        self.pool_last = nn.AvgPool2d(self.pool_size)
        self.classifier = nn.Linear(self.num_features, args.num_classes)
        self.classifier.bias.data.zero_()

        ### initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        return

    def add_block(self, i):
        ### Check if ith is the last one
        last = (i == len(self.stages) - 1)
        block = _DenseBlock(
            num_layers=self.stages[i],
            in_channels=self.num_features,
            growth_rate=self.growth[i],
            args=self.args,
        )
        self.features.add_module('denseblock_%d' % (i + 1), block)
        self.num_features += self.stages[i] * self.growth[i]
        if not last:
            trans = _Transition(in_channels=self.num_features,
                                args=self.args)
            self.features.add_module('transition_%d' % (i + 1), trans)

    def forward(self, x, progress=None, threshold=None):
        if progress:
            DynamicMultiHeadConv.global_progress = progress
        features, _lasso_loss = self.features(x)
        features = self.bn_last(features)
        features = self.relu_last(features)
        features = self.pool_last(features)
        out = features.view(features.size(0), -1)
        out = self.classifier(out)
        return out, _lasso_loss
