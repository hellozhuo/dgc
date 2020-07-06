"""
Modified based on Official Pytorch repository
"""


import torch
import torch.nn as nn
from layers import DynamicMultiHeadConv

__all__ = ['dyresnet18']



def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, args, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = DynamicMultiHeadConv(inplanes, planes, kernel_size=3, stride=stride, 
                padding=1, heads=args.heads, squeeze_rate=args.squeeze_rate, 
                gate_factor=args.gate_factor)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = DynamicMultiHeadConv(planes, planes, kernel_size=3, stride=1, 
                padding=1, heads=args.heads, squeeze_rate=args.squeeze_rate, 
                gate_factor=args.gate_factor)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        _lasso_loss = x[1]
        identity = x[0]

        out = self.conv1(x[0])
        _lasso_loss.append(out[1])
        out = self.bn1(out[0])
        out = self.relu(out)

        out = self.conv2(out)
        _lasso_loss.append(out[1])
        out = self.bn2(out[0])

        if self.downsample is not None:
            x_down = self.downsample(x[0])
            identity = x_down[0]
            _lasso_loss.append(x_down[1])

        out += identity
        out = self.relu(out)

        return [out, _lasso_loss]

class Norm_after_downsample(nn.Module):

    def __init__(self, norm_layer, planes):
        super(Norm_after_downsample, self).__init__()
        self.norm = norm_layer(planes) 

    def forward(self, x):
        _lasso_loss = x[1]
        out = self.norm(x[0])
        return [out, _lasso_loss]


class ResNet(nn.Module):

    def __init__(self, args, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.args = args

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
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

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                DynamicMultiHeadConv(self.inplanes, planes * block.expansion, 
                    kernel_size=1, stride=stride, padding=0, heads=self.args.heads, 
                    squeeze_rate=self.args.squeeze_rate, gate_factor=self.args.gate_factor
                ),
                Norm_after_downsample(norm_layer, planes * block.expansion),
            )

        layers = []
        layers.append(block(self.args, self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.args, self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x, progress=None, threshold=None):
        if progress:
            DynamicMultiHeadConv.global_progress = progress
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1([x,[]])
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        _lasso_loss = x[1]

        x = self.avgpool(x[0])
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x, _lasso_loss

def _resnet(args, arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(args, block, layers, **kwargs)
    return model


def dyresnet18(args):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(args, 'resnet18', BasicBlock, [2, 2, 2, 2], pretrained=False, progress=True)

