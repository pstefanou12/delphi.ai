"""
Truncated Linear Regression.
"""


import torch as ch
from torch import Tensor
import torch.nn as nn
from torch.nn import Linear
from torch.utils.data import TensorDataset
from sklearn.linear_model import LinearRegression
from cox.utils import Parameters
from cox.store import Store
import config

from .stats import stats
from ..oracle import oracle
from ..train import train_model
from ..grad import TruncatedMSE, TruncatedUnknownVarianceMSE
from ..utils.helpers import Bounds, LinearUnknownVariance, setup_store_with_metadata
from ..utils import defaults
from ..utils.datasets import DataSet, TENSOR_REQUIRED_ARGS, TENSOR_OPTIONAL_ARGS


class TruncatedLinearRegression(stats):
    """
    """
    def __init__(
            self,
            phi: oracle,
            alpha: float,
            args: Parameters,
            bias: bool=True,
            var: float = None,
            device: str="cpu",
            store: Store=None, 
            table: str=None,
            **kwargs):
        """
        """
        super(TruncatedLinearRegression).__init__()
        # instance variables
        self.phi = phi 
        self.alpha = alpha 
        self.bias = bias 
        self.device = device
        self.var = var 
        self.store, self.table = store, table
        self._lin_reg = None
        self.projection_set = None
        self.criterion = None

        args.__setattr__('phi', phi)
        args.__setattr__('alpha', alpha)
        args.__setattr__('device', device)
        args.__setattr__('var', var)
        config.args = defaults.check_and_fill_args(args, defaults.REGRESSION_ARGS, TensorDataset)

    def fit(self, X: Tensor, y: Tensor):
        """
        """
        # create dataset and dataloader
        ds_kwargs = {
            'custom_class_args': {
                'X': X, 'y': y},
            'custom_class': TensorDataset,
            'transform_train': None,
            'transform_test': None,
        }
        ds = DataSet('tensor', TENSOR_REQUIRED_ARGS, TENSOR_OPTIONAL_ARGS, data_path=None,
                     **ds_kwargs)
        loaders = ds.make_loaders(workers=config.args.workers, batch_size=config.args.batch_size)
        self.emp_lin_reg = LinearRegression(fit_intercept=self.bias).fit(X, y)
        self.emp_weight = Tensor(self.emp_lin_reg.coef_)
        self.emp_bias = Tensor(self.emp_lin_reg.intercept_) if self.bias else None
        self.emp_var = ch.var(Tensor(self.emp_lin_reg.predict(X)) - y, dim=0)[..., None]
        
        if config.args.var: # known variance
            self.criterion = TruncatedMSE.apply
            self._lin_reg = Linear(in_features=X.size(1), out_features=1, bias=self.bias)
            # assign empirical estimates
            self._lin_reg.weight.data, self._lin_reg.bias.data = self.emp_weight, self.emp_bias
            self.projection_set = TruncatedRegressionProjectionSet(X, y, config.args.radius, config.args.alpha, bias=config.args.bias, clamp=config.args.clamp)
            update_params = None
        else:  # unknown variance
            self.criterion = TruncatedUnknownVarianceMSE.apply
            self._lin_reg = LinearUnknownVariance(in_features=X.size(1), out_features=y.size(1), bias=self.bias)
            # assign empirical estimates
            self._lin_reg.lambda_.data = self.emp_var.inverse()
            self._lin_reg.weight.data, self._lin_reg.bias.data = self.emp_weight * self._lin_reg.lambda_ , self.emp_bias * self._lin_reg.lambda_

            self.projection_set = TruncatedUnknownVarianceProjectionSet(X, y, config.args.radius, config.args.alpha, bias=config.args.bias, clamp=config.args.clamp)
            # update_params = [{'params': [self._lin_reg.weight, self._lin_reg.bias]},
            #                 {'params': self._lin_reg.lambda_, 'lr': config.args.var_lr}]
            update_params = None

        config.args.__setattr__('custom_criterion', self.criterion)
        config.args.__setattr__('iteration_hook', self.projection_set)
        # run PGD for parameter estimation
        return train_model(config.args, self._lin_reg, loaders, update_params=update_params)


    def __call__(self, x: Tensor): 
        """
        """
        return self._lin_reg(x)


class TruncatedRegressionProjectionSet:
    """
    Project to domain for linear regression with known variance
    """
    def __init__(self, X, y, r, alpha, bias=True, clamp=True):
        # use OLS as empirical estimate to define projection set
        self.bias = bias
        self.r = r
        self.alpha = alpha
        self.clamp = clamp
        self.emp_lin_reg = LinearRegression(fit_intercept=self.bias).fit(X, y)
        self.emp_weight = Tensor(self.emp_lin_reg.coef_)
        self.emp_bias = Tensor(self.emp_lin_reg.intercept_) if self.bias else None
        self.radius = r * (4.0 * ch.log(2.0 / self.alpha) + 7.0)

        if self.clamp:
            self.weight_bounds = Bounds(self.emp_weight.flatten() - self.radius,
                                        self.emp_weight.flatten() + self.radius)
            self.bias_bounds = Bounds(self.emp_bias.flatten() - self.radius,
                                      self.emp_bias.flatten() + self.radius) if self.bias else None
        else:
            pass
<<<<<<< HEAD
        
=======

>>>>>>> b182ac21c5c9c50e1dc4d9bb932de812d0795eac
    def __call__(self, M, i, loop_type, inp, target):
        if self.clamp:
            M.weight.data = ch.stack(
                [ch.clamp(M.weight[i], self.weight_bounds.lower[i], self.weight_bounds.upper[i]) for i in
                 range(M.weight.size(0))])
            if config.args.bias:
                M.bias.data = ch.clamp(M.bias, float(self.bias_bounds.lower), float(self.bias_bounds.upper)).reshape(
                    M.bias.size())
        else:
            pass


class TruncatedUnknownVarianceProjectionSet:
    """
    Project parameter estimation back into domain of expected results for censored normal distributions.
    """

    def __init__(self, X, y, r, alpha, bias=True, clamp=True):
        """
        :param lin_reg: empirical regression with unknown noise variance
        """
        self.bias = bias
        self.r = r
        self.alpha = alpha
        self.clamp = clamp
        self.emp_lin_reg = LinearRegression(fit_intercept=self.bias).fit(X, y)
        self.emp_weight = Tensor(self.emp_lin_reg.coef_)
        self.emp_bias = Tensor(self.emp_lin_reg.intercept_) if self.bias else None
        self.emp_var = ch.var(Tensor(self.emp_lin_reg.predict(X)) - y, dim=0)[..., None]
        self.radius = self.r * (12.0 + 4.0 * ch.log(2.0 / self.alpha))

        if self.clamp:
            self.weight_bounds, self.var_bounds = Bounds(self.emp_weight.flatten() - self.radius,
                                                         self.emp_weight.flatten() + self.radius), Bounds(
                self.emp_var.flatten() / self.r, (self.emp_var.flatten()) / self.alpha.pow(2))
            self.bias_bounds = Bounds(self.emp_bias.flatten() - self.radius,
                                      self.emp_bias.flatten() + self.radius) if self.bias else None
        else:
            pass

    def __call__(self, M, i, loop_type, inp, target):
        var = M.lambda_.inverse()
        weight = M.weight * var

        if self.clamp:
            # project noise variance
            M.lambda_.data = ch.clamp(var, float(self.var_bounds.lower), float(self.var_bounds.upper)).inverse()
            # project weights
            M.weight.data = ch.cat(
                [ch.clamp(weight[i].unsqueeze(0), float(self.weight_bounds.lower[i]),
                          float(self.weight_bounds.upper[i]))
                 for i in range(weight.size(0))]) * M.lambda_
            # project bias
            if self.bias:
<<<<<<< HEAD
                bias = M.bias * var
                M.bias.data = (ch.clamp(bias, float(self.bias_bounds.lower), float(self.bias_bounds.upper)) * M.lambda_).reshape(M.bias.size())
=======
                bias = M.layer.bias * var
                M.layer.bias.data = (ch.clamp(bias, float(self.bias_bounds.lower), float(self.bias_bounds.upper)) * M.lambda_).reshape(M.layer.bias.size())
>>>>>>> b182ac21c5c9c50e1dc4d9bb932de812d0795eac
        else:
            pass

