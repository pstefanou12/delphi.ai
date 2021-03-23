"""
Truncated regression
"""

import torch as ch
from torch import Tensor
import torch.nn as nn
from torch.nn import Linear
from torch.utils.data import DataLoader
from cox.utils import Parameters
import config

from ..Function import TruncatedMSE, TruncatedUnknownVarianceMSE
from ..defaults import REGRESSION_DEFAULTS
from ..main import main
from ..train import train_model
from ..utils.helpers import Bounds, setup_store_with_metadata


def truncated_regression(args, S, known=False, store=None):
    """
    Truncated regression module with known and unknown noise variance.
    """
    if known:
        args = setup_args(args)
        setattr(args, 'custom_criterion', TruncatedMSE.apply)
    else:
        args = setup_args(args)
        setattr(args, 'custom_criterion', TruncatedUnknownVarianceMSE.apply)
    if store:
        store = setup_store_with_metadata(args)
    # initialize model

    self._lin_reg = LinearUnknownVariance(S.dataset.v, S.dataset.lambda_, bias=S.dataset.v0)



    return train_model(self._lin_reg, (S, None), update_params=update_params, device=config.args.device)



def pretrain_hook(args):
    # initialize model with empirical estimates
    if args.var:
        lin_reg = Linear(in_features=S.dataset.w.size(0), out_features=1, bias=args.bias)
        self._lin_reg.weight.data = S.dataset.w
        self._lin_reg.bias = S.dataset.w0 if config.args.bias else None
        self.projection_set = TruncatedRegressionProjectionSet(self._lin_reg)
        update_params = None
    else:
        self.projection_set = TruncatedUnknownVarianceProjectionSet(self._lin_reg)
        update_params = [
                {'params': self._lin_reg.v},
                {'params': self._lin_reg.bias},
                {'params': self._lin_reg.lambda_, 'lr': config.args.var_lr}]

        args.__setattr__('iteration_hook', self.projection_set)


class TruncatedRegressionProjectionSet:
    """
    Project to domain for linear regression with known variance
    """
    def __init__(self, args, emp_lin_reg):
        self.emp_weight = emp_lin_reg.weight.data
        self.emp_bias = emp_lin_reg.bias.data if config.args.bias else None
        self.radius = config.args.radius * (4.0 * ch.log(2.0 / config.args.alpha) + 7.0)
        if config.args.clamp:
            self.weight_bounds = Bounds(self.emp_weight.flatten() - config.args.radius,
                                        self.emp_weight.flatten() + config.args.radius)
            self.bias_bounds = Bounds(self.emp_bias.flatten() - config.args.radius,
                                      self.emp_bias.flatten() + config.args.radius) if config.args.bias else None
        else:
            pass



    def __call__(self, M, i, loop_type, inp, target):
        if config.args.clamp:
            M.weight.data = ch.stack(
                [ch.clamp(M.weight[i], self.weight_bounds.lower[i], self.weight_bounds.upper[i]) for i in
                 range(M.weight.size(0))])
            if config.args.bias:
                M.bias.data = ch.clamp(M.bias, self.bias_bounds.lower, self.bias_bounds.upper).reshape(
                    M.bias.size())
        else:
            pass


class TruncatedUnknownVarianceProjectionSet:
    """
    Project parameter estimation back into domain of expected results for censored normal distributions.
    """

    def __init__(self, emp_lin_reg):
        """
        :param emp_lin_reg: empirical regression with unknown noise variance
        """
        self.emp_var = emp_lin_reg.lambda_.data.inverse()
        self.emp_weight = emp_lin_reg.v.data * self.emp_var
        self.emp_bias = emp_lin_reg.bias.data * self.emp_var if config.args.bias else None
        self.param_radius = config.args.radius * (12.0 + 4.0 * ch.log(2.0 / config.args.alpha))

        if config.args.clamp:
            self.weight_bounds, self.var_bounds = Bounds(self.emp_weight.flatten() - self.param_radius,
                                                         self.emp_weight.flatten() + self.param_radius), Bounds(
                self.emp_var.flatten() / config.args.radius, (self.emp_var.flatten()) / config.args.alpha.pow(2))
            self.bias_bounds = Bounds(self.emp_bias.flatten() - self.param_radius,
                                      self.emp_bias.flatten() + self.param_radius) if config.args.bias else None
        else:
            pass

    def __call__(self, M, i, loop_type, inp, target):
        var = M.lambda_.inverse()
        weight = M.v.data * var

        if config.args.clamp:
            # project noise variance
            M.lambda_.data = ch.clamp(var, float(self.var_bounds.lower), float(self.var_bounds.upper)).inverse()
            # project weights
            M.v.data = ch.cat(
                [ch.clamp(weight[i].unsqueeze(0), float(self.weight_bounds.lower[i]),
                          float(self.weight_bounds.upper[i]))
                 for i in range(weight.size(0))]) * M.lambda_
            # project bias
            if config.args.bias:
                bias = M.bias * var
                M.bias.data = ch.clamp(bias, float(self.bias_bounds.lower), float(self.bias_bounds.upper)) * M.lambda_
        else:
            pass


class LinearUnknownVariance(nn.Module):
    """
    Linear layer with unknown noise variance.
    """
    def __init__(self, v, lambda_, bias=None):
        """
        :param lambda_: 1/empirical variance
        :param v: empirical weight*lambda_ estimate
        :param bias: (optional) empirical bias*lambda_ estimate
        """
        super(LinearUnknownVariance, self).__init__()
        self.register_parameter(name='v', param=ch.nn.Parameter(v))
        self.register_parameter(name='lambda_', param=ch.nn.Parameter(lambda_))
        self.register_parameter(name='bias', param=ch.nn.Parameter(bias))

    def forward(self, x):
        var = self.lambda_.clone().detach().inverse()
        w = self.v*var
        if self.bias.nelement() > 0:
            return x.matmul(w) + self.bias * var
        return x.matmul(w)
