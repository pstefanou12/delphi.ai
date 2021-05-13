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
        # use OLS as empirical estimate
        lin_reg = LinearRegression(fit_intercept=self.bias).fit(X, y)
        
        if config.args.var: # known variance
            self.criterion = TruncatedMSE.apply
            self._lin_reg = Linear(in_features=X.size(1), out_features=1, bias=config.args.bias)
            self._lin_reg.weight.data = ch.nn.Parameter(Tensor(lin_reg.coef_))
            self._lin_reg.bias = ch.nn.Parameter(Tensor(lin_reg.intercept_)) if config.args.bias else None
            self.projection_set = TruncatedRegressionProjectionSet(self._lin_reg)
            update_params = None
        else:  # unknown variance
            self.criterion = TruncatedUnknownVarianceMSE.apply
            self.emp_lambda = ch.var(Tensor(lin_reg.predict(X)) - y, dim=0)[..., None].inverse()
            self._lin_reg = LinearUnknownVariance(Tensor(lin_reg.coef_).T * self.emp_lambda, self.emp_lambda,
                                                      bias=Tensor(lin_reg.intercept_) * self.emp_lambda)
            self.projection_set = TruncatedUnknownVarianceProjectionSet(self._lin_reg)
            update_params = [
                    {'params': self._lin_reg.v},
                    {'params': self._lin_reg.bias},
                    {'params': self._lin_reg.lambda_, 'lr': config.args.var_lr}]

        config.args.__setattr__('custom_criterion', self.criterion)
        config.args.__setattr__('iteration_hook', self.projection_set)
        # run PGD for parameter estimation
        return train_model(config.args, self._lin_reg, loaders, update_params=update_params)


class TruncatedRegressionProjectionSet:
    """
    Project to domain for linear regression with known variance
    """
    def __init__(self, lin_reg):
        self.weight = lin_reg.weight.data
        self.bias = lin_reg.bias.data if config.args.bias else None
        self.radius = config.args.radius * (4.0 * ch.log(2.0 / config.args.alpha) + 7.0)
        if config.args.clamp:
            self.weight_bounds = Bounds(self.weight.flatten() - config.args.radius,
                                        self.weight.flatten() + config.args.radius)
            self.bias_bounds = Bounds(self.bias.flatten() - config.args.radius,
                                      self.bias.flatten() + config.args.radius) if config.args.bias else None
        else:
            pass

    def __call__(self, M, i, loop_type, inp, target):
        if config.args.clamp:
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

    def __init__(self, lin_reg):
        """
        :param lin_reg: empirical regression with unknown noise variance
        """
        self.var = lin_reg.lambda_.data.inverse()
        self.weight = lin_reg.v.data * self.var
        self.bias = lin_reg.bias.data * self.var if config.args.bias else None
        self.param_radius = config.args.radius * (12.0 + 4.0 * ch.log(2.0 / config.args.alpha))

        if config.args.clamp:
            self.weight_bounds, self.var_bounds = Bounds(self.weight.flatten() - self.param_radius,
                                                         self.weight.flatten() + self.param_radius), Bounds(
                self.var.flatten() / config.args.radius, (self.var.flatten()) / config.args.alpha.pow(2))
            self.bias_bounds = Bounds(self.bias.flatten() - self.param_radius,
                                      self.bias.flatten() + self.param_radius) if config.args.bias else None
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

