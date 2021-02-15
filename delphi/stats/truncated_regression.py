"""
Truncated regression
"""

import torch as ch
from torch import Tensor
import torch.nn as nn
from torch.nn import Linear
from torch.nn import init
from torch.utils.data import DataLoader
from cox.utils import Parameters
from sklearn.linear_model import LinearRegression

from ..Function import TruncatedMSE, TruncatedUnknownVarianceMSE
from ..defaults import REGRESSION_DEFAULTS
from ..main import main
from ..train import train_model
from ..utils.helpers import Bounds, setup_store_with_metadata


def truncated_regression(args, X, y, known=False, store=None):
    """
    Truncated regression module with known and unknown noise variance.
    """
    # setup model and training procedure
    if known:
        # check all parameters
        args = setup_args(args)
        setattr(args, 'custom_criterion', TruncatedMSE.apply)
        trunc_reg = Lienar(in_features=X.size(1), out_features=1, bias=args.bias)
        # assign emprical estimates
        lin_reg = LinearRegression(intercept_=args.bias)
        lin_reg.fit(X, y)
        trunc_reg.weight = lin_reg.coef_
        if args.bias: trunc_reg.bias = lin_reg.intercept_
        params = None
    else:
        # check all parameters
        args = setup_args(args)
        setattr(args, 'custom_criterion', TruncatedUnknownVarianceMSE.apply)
        trunc_reg = LinearUnknownVariance(in_features=X.size(0), bias=args.bias)
        # assign emprical estimates
        lin_reg = LinearRegression(intercept_=args.bias)
        lin_reg.fit(X, y)
        trunc_reg.lambda_ = ch.var(Tensor(lin_reg.predict(X)) - y, dim=0).unsqueeze(0).inverse()
        trunc_reg.v = lin_reg.coef_*trunc_reg.lambda_
        if args.bias: trunc_reg.bias = lin_reg.intercept_*trunc_reg.lambda_
        params = [
            {'params': trunc_reg.v},
            {'params': trunc_reg.bias},
            {'params': trunc_reg.lambda_, 'lr': args.var_lr}]

    # dataset
    dataset = TensorDataset(X, y)
    S = DataLoader(dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=args.shuffle)

    if store:
        store = setup_store_with_metadata(args)

    return train_model(trunc_reg, (S, None), update_params=params, device=args.device)



class LinearUnknownVariance(nn.Module):
    """
    Linear layer with unknown noise variance. Used for regression models.
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool=True):
        """
        :param lambda_: 1/empirical variance
        :param v: empirical weight*lambda_ estimate
        :param bias: (optional) empirical bias*lambda_ estimate
        """
        super(LinearUnknownVariance, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.v = Parameter(Tensor(out_features, in_features))
        self.lambda_ = Parameter(Tensor(out_features))
        if bias:
            self.bias = Parameter(Tensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        var = self.lambda_.clone().detach().inverse()
        w = self.v*var
        if self.bias.nelement() > 0:
            return x.matmul(w) + self.bias * var
        return x.matmul(w)

def pretrain_hook(args, trunc_reg):
    # initialize model with empirical estimates
    if args.var:
        projection_set = TruncatedRegressionProjectionSet(trunc_reg)
    else:
        projection_set = TruncatedUnknownVarianceProjectionSet(trunc_reg)
    args.__setattr__('iteration_hook', projection_set)


class TruncatedRegressionProjectionSet:
    """
    Project to domain for linear regression with known variance
    """
    def __init__(self, trunc_reg, radius, alpha, clamp):
        self.emp_weight = trunc_reg.weight.data
        self.emp_bias = trunc_reg.bias.data if trunc_reg.bias else None
        self.radius = radius * (4.0 * ch.log(2.0 / alpha) + 7.0)
        self.clamp = clamp
        if self.clamp:
            self.weight_bounds = Bounds(self.emp_weight.flatten() - self.radius,
                                        self.emp_weight.flatten() + self.radius)
            self.bias_bounds = Bounds(self.emp_bias.flatten() - self.radius,
                                      self.emp_bias.flatten() + self.radius) if trunc_reg.bias is not None else None
        else:
            pass

    def __call__(self, trunc_reg, i, loop_type, inp, target):
        if self.clamp:
            trunc_reg.weight.data = ch.stack(
                [ch.clamp(trunc_reg.weight[i], self.weight_bounds.lower[i], self.weight_bounds.upper[i]) for i in
                 range(trunc_reg.weight.size(0))])
            if trunc_reg.bias is not None:
                trunc_reg.bias.data = ch.clamp(trunc_reg.bias, self.bias_bounds.lower, self.bias_bounds.upper).reshape(
                    trunc_reg.bias.size())
        else:
            pass


class TruncatedUnknownVarianceProjectionSet:
    """
    Project parameter estimation back into domain of expected results for censored normal distributions.
    """

    def __init__(self, trunc_reg, radius, alpha, clamp):
        """
        :param emp_lin_reg: empirical regression with unknown noise variance
        """
        self.emp_var = turnc_reg.lambda_.data.inverse()
        self.emp_weight = trunc_reg.v.data * self.emp_var
        self.emp_bias = trunc_reg.bias.data * self.emp_var if trunc_reg.bias is not None else None
        self.radius = radius * (12.0 + 4.0 * ch.log(2.0 / config.args.alpha))
        self.clamp = clamp
        if self.clamp:
            self.weight_bounds, self.var_bounds = Bounds(self.emp_weight.flatten() - self.radius,
                                                         self.emp_weight.flatten() + self.radius), Bounds(
                self.emp_var.flatten() / radius, (self.emp_var.flatten()) / alpha.pow(2))
            self.bias_bounds = Bounds(self.emp_bias.flatten() - self.radius,
                                      self.emp_bias.flatten() + self.radius) if trunc_reg.bias else None
        else:
            pass

    def __call__(self, trunc_reg, i, loop_type, inp, target):
        # reparameterize
        var = trunc_reg.lambda_.inverse()
        weight = trunc_reg.v.data * var

        if self.clamp:
            # project noise variance
            trunc_reg.lambda_.data = ch.clamp(var, float(self.var_bounds.lower), float(self.var_bounds.upper)).inverse()
            # project weights
            trunc_reg.v.data = ch.cat(
                [ch.clamp(weight[i].unsqueeze(0), float(self.weight_bounds.lower[i]),
                          float(self.weight_bounds.upper[i]))
                 for i in range(weight.size(0))]) * trunc_reg.lambda_
            # project bias
            if trunc_reg.bias is not None:
                bias = trunc_reg.bias * var
                trunc_reg.bias.data = ch.clamp(bias, float(self.bias_bounds.lower), float(self.bias_bounds.upper)) * M.lambda_
        else:
            pass
