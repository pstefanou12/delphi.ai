# Author: pstefanou12@
"""Neural-network layers shared by the attacker and the model zoos."""

import torch as ch
from torch import nn


class InputNormalize(ch.nn.Module):
    """Normalize inputs to a fixed, user-specified mean and std."""

    def __init__(self, new_mean, new_std):
        """Initialize InputNormalize with mean and standard deviation."""
        super().__init__()
        new_std = new_std[..., None, None]
        new_mean = new_mean[..., None, None]

        self.register_buffer("new_mean", new_mean)
        self.register_buffer("new_std", new_std)

    def forward(self, x):
        """Normalize x by clamping to [0,1] and applying stored mean/std."""
        x = ch.clamp(x, 0, 1)
        x_normalized = (x - self.new_mean) / self.new_std
        return x_normalized


class FakeReLU(ch.autograd.Function):  # pylint: disable=abstract-method
    """Custom autograd function implementing a fake ReLU (identity backward)."""

    @staticmethod
    def forward(ctx, inp):  # pylint: disable=arguments-differ,unused-argument
        """Forward pass: clamp input to be non-negative."""
        return inp.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):  # pylint: disable=arguments-differ,unused-argument
        """Backward pass: pass gradient through unchanged."""
        return grad_output


class FakeReLUM(nn.Module):
    """Module wrapper around FakeReLU autograd function."""

    def forward(self, x):
        """Apply FakeReLU to x."""
        return FakeReLU.apply(x)


class SequentialWithArgs(ch.nn.Sequential):
    """Sequential module that passes extra args/kwargs to the last layer."""

    def forward(self, inp, *args, **kwargs):  # pylint: disable=arguments-differ
        """Forward pass, passing args and kwargs to the final module only."""
        vs = list(self._modules.values())
        num_layers = len(vs)
        for i in range(num_layers):
            if i == num_layers - 1:
                inp = vs[i](inp, *args, **kwargs)
            else:
                inp = vs[i](inp)
        return inp
