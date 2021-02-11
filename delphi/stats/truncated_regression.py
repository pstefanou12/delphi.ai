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

from .stats import stats
from ..oracle import oracle
from ..train import train_model
from ..utils.helpers import Bounds


class truncated_regression(stats):
    """
    """
    def __init__(
            self,
            phi: oracle,
            alpha: float,
            bias: bool=True,
            var: float = None,
            epochs: int=50,
            lr: float=1e-1,
            var_lr: float=1e-3,
            num_samples: int=100,
            radius: float=2.0,
            clamp: bool=True,
            eps: float=1e-10,
            device="cpu",
            **kwargs):
        """
        """
        super().__init__()
        # initialize hyperparameters for algorithm
        config.args = Parameters({
            'phi': phi,
            'epochs': epochs,
            'lr': lr,
            'var_lr': var_lr,
            'num_samples': num_samples,
            'alpha': Tensor([alpha]),
            'radius': Tensor([radius]),
            'var': Tensor([var]) if var else None,
            'bias': bias,
            'clamp': clamp,
            'eps': eps,
            'momentum': 0.9,
            'weight_decay': 5e-4,
            'step_lr': 10,
            'step_lr_gamma': .9,
            'device': device,
        })
        self._lin_reg = None
        self.projection_set = None
        # intialize loss function and add custom criterion to hyperparameters
        if not config.args.var:
            self.criterion = TruncatedUnknownVarianceMSE.apply
        else:
            self.criterion = TruncatedMSE.apply
        config.args.__setattr__('custom_criterion', self.criterion)

    def fit(
            self,
            S: DataLoader):
        """
        """
        # initialize model with empirical estimates
        if config.args.var:
            self._lin_reg = Linear(in_features=S.dataset.w.size(0), out_features=1, bias=config.args.bias)
            self._lin_reg.weight.data = S.dataset.w
            self._lin_reg.bias = S.dataset.w0 if config.args.bias else None
            self.projection_set = TruncatedRegressionProjectionSet(self._lin_reg)
            update_params = None
        else:
            self._lin_reg = LinearUnknownVariance(S.dataset.v, S.dataset.lambda_, bias=S.dataset.v0)
            self.projection_set = TruncatedUnknownVarianceProjectionSet(self._lin_reg)
            update_params = [
                    {'params': self._lin_reg.v},
                    {'params': self._lin_reg.bias},
                    {'params': self._lin_reg.lambda_, 'lr': config.args.var_lr}]

        config.args.__setattr__('iteration_hook', self.projection_set)
        # run PGD for parameter estimation
        return train_model(self._lin_reg, (S, None), update_params=update_params, device=config.args.device)


class TruncatedUnknownVarianceMSE(ch.autograd.Function):
    """
    Computes the gradient of negative population log likelihood for truncated linear regression
    with unknown noise variance.
    """
    @staticmethod
    def forward(ctx, pred, targ, lambda_):
        ctx.save_for_backward(pred, targ, lambda_)
        return 0.5 * (pred.float() - targ.float()).pow(2).mean(0)

    @staticmethod
    def backward(ctx, grad_output):
        pred, targ, lambda_ = ctx.saved_tensors
        # calculate std deviation of noise distribution estimate
        sigma, z = ch.sqrt(lambda_.inverse()), Tensor([]).to(config.args.device)

        for i in range(pred.size(0)):
            # add random noise to logits
            noised = pred[i] + sigma*ch.randn(ch.Size([config.args.num_samples, 1])).to(config.args.device)
            # filter out copies within truncation set
            filtered = config.args.phi(noised).bool()
            z = ch.cat([z, noised[filtered.nonzero(as_tuple=False)][0]]) if ch.any(filtered) else ch.cat([z, pred[i].unsqueeze(0)])
        """
        multiply the v gradient by lambda, because autograd computes 
        v_grad*x*variance, thus need v_grad*(1/variance) to cancel variance 
        factor
        """
        return lambda_*(z - targ) / pred.size(0), targ / pred.size(0),\
                (0.5 * targ.pow(2) - 0.5 * z.pow(2)) / pred.size(0)


class TruncatedMSE(ch.autograd.Function):
    """
    Computes the gradient of the negative population log likelihood for censored regression
    with known noise variance.
    """

    @staticmethod
    def forward(ctx, pred, targ):
        ctx.save_for_backward(pred, targ)
        return 0.5 * (pred.float() - targ.float()).pow(2).mean(0)

    @staticmethod
    def backward(ctx, grad_output):
        pred, targ = ctx.saved_tensors
        # make args.num_samples copies of pred, N x B x 1
        stacked = pred[None, ...].repeat(config.args.num_samples, 1, 1)
        # add random noise to each copy
        noised = stacked + ch.randn_like(stacked)
        # filter out copies where pred is in bounds
        filtered = ch.stack([config.args.phi(batch).unsqueeze(1) for batch in noised]).float()
        # average across truncated indices
        out = (filtered * noised).sum(dim=0) / (filtered.sum(dim=0) + config.args.eps)
        return (out - targ) / pred.size(0), targ / pred.size(0)


class TruncatedRegressionProjectionSet:
    """
    Project to domain for linear regression with known variance
    """
    def __init__(self, emp_lin_reg):
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

