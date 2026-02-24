"""ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""

import torch
from torch import nn
import torch.nn.functional as F

from ..utils.helpers import SequentialWithArgs, FakeReLU


class BasicBlock(nn.Module):
    """Basic residual block for ResNet."""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        """Initialize BasicBlock with conv layers and optional shortcut."""
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x, fake_relu=False):  # pylint: disable=arguments-differ
        """Forward pass through the basic block."""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        if fake_relu:
            return FakeReLU.apply(out)
        return F.relu(out)


class Bottleneck(nn.Module):
    """Bottleneck residual block for ResNet."""

    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        """Initialize Bottleneck with three conv layers and optional shortcut."""
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x, fake_relu=False):  # pylint: disable=arguments-differ
        """Forward pass through the bottleneck block."""
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        if fake_relu:
            return FakeReLU.apply(out)
        return F.relu(out)


class ResNet_(nn.Module):  # pylint: disable=invalid-name,too-many-instance-attributes
    """ResNet architecture for CIFAR-scale datasets."""

    # feat_scale lets us deal with CelebA, other non-32x32 datasets
    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self, block, num_blocks, num_classes=10, feat_scale=1, wm=1
    ):
        """Initialize ResNet with given block type and layer configuration."""
        super().__init__()

        widths = [64, 128, 256, 512]
        widths = [int(w * wm) for w in widths]

        self.in_planes = widths[0]
        self.conv1 = nn.Conv2d(
            3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(block, widths[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, widths[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, widths[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, widths[3], num_blocks[3], stride=2)
        self.linear = nn.Linear(feat_scale * widths[3] * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        """Build a residual layer from repeated blocks."""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:  # pylint: disable=invalid-name
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return SequentialWithArgs(*layers)

    def forward(  # pylint: disable=arguments-differ
        self, x, with_latent=False, fake_relu=False, no_relu=False
    ):
        """Forward pass through the full ResNet."""
        assert not no_relu, "no_relu not yet supported for this architecture"
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out, fake_relu=fake_relu)
        out = F.avg_pool2d(out, 4)  # pylint: disable=not-callable
        pre_out = out.view(out.size(0), -1)
        final = self.linear(pre_out)
        if with_latent:
            return final, pre_out
        return final


def ResNet18(**kwargs):  # pylint: disable=invalid-name
    """Return a ResNet-18 model."""
    return ResNet_(BasicBlock, [2, 2, 2, 2], **kwargs)


def ResNet18Wide(**kwargs):  # pylint: disable=invalid-name
    """Return a wide ResNet-18 model (width multiplier 5)."""
    return ResNet_(BasicBlock, [2, 2, 2, 2], wm=5, **kwargs)


def ResNet18Thin(**kwargs):  # pylint: disable=invalid-name
    """Return a thin ResNet-18 model (width multiplier 0.75)."""
    return ResNet_(BasicBlock, [2, 2, 2, 2], wm=0.75, **kwargs)


def ResNet34(**kwargs):  # pylint: disable=invalid-name
    """Return a ResNet-34 model."""
    return ResNet_(BasicBlock, [3, 4, 6, 3], **kwargs)


def ResNet50(**kwargs):  # pylint: disable=invalid-name
    """Return a ResNet-50 model."""
    return ResNet_(Bottleneck, [3, 4, 6, 3], **kwargs)


def ResNet101(**kwargs):  # pylint: disable=invalid-name
    """Return a ResNet-101 model."""
    return ResNet_(Bottleneck, [3, 4, 23, 3], **kwargs)


def ResNet152(**kwargs):  # pylint: disable=invalid-name
    """Return a ResNet-152 model."""
    return ResNet_(Bottleneck, [3, 8, 36, 3], **kwargs)


# class ResNet(delphi):
# def __init__(self,
# args: Parameters,
# block, num_blocks, num_classes=10, feat_scale=1, wm=1):
# super().__init__(args)
# self.model = ResNet18()
# self.params = self.model.parameters()
# self.loss =  nn.CrossEntropyLoss()

# widths = [64, 128, 256, 512]
# widths = [int(w * wm) for w in widths]

# self.in_planes = widths[0]
# self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1,
#    padding=1, bias=False)
# self.bn1 = nn.BatchNorm2d(self.in_planes)
# self.layer1 = self._make_layer(block, widths[0], num_blocks[0], stride=1)
# self.layer2 = self._make_layer(block, widths[1], num_blocks[1], stride=2)
# self.layer3 = self._make_layer(block, widths[2], num_blocks[2], stride=2)
# self.layer4 = self._make_layer(block, widths[3], num_blocks[3], stride=2)
# self.linear = nn.Linear(feat_scale*widths[3]*block.expansion, num_classes)

# def __call__(self, x, with_latent=False, fake_relu=False, no_relu=False):
# assert (not no_relu),  \
# "no_relu not yet supported for this architecture"
# out = F.relu(self.bn1(self.conv1(x)))
# out = self.layer1(out)
# out = self.layer2(out)
# out = self.layer3(out)
# out = self.layer4(out, fake_relu=fake_relu)
# out = F.avg_pool2d(out, 4)
# pre_out = out.view(out.size(0), -1)
# final = self.linear(pre_out)
# if with_latent:
# return final, pre_out
# return final

# def to(self, device):
# """
# Wrapper method to put DNN onto a specific device GPU/CPU.
# Args:
# :param device: string that says the device to put on.
# """
# self.model = self.model.to(device)

# @property
# def parameters(self):
# return self.model.parameters()


resnet50 = ResNet50
resnet18 = ResNet18
resnet34 = ResNet34
resnet101 = ResNet101
resnet152 = ResNet152
resnet18wide = ResNet18Wide


# resnet18thin = ResNet18Thin
def test():
    """Run a quick sanity check on ResNet18."""
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())
