"""
Truncated Logistic Regression.
"""


import torch as ch
from torch import Tensor
from torch.nn import Linear
from torch.utils.data import TensorDataset, DataLoader
from torch import sigmoid as sig
from torch.distributions import Uniform
from torch.distributions.transforms import SigmoidTransform
from torch.distributions.transformed_distribution import TransformedDistribution
from cox.utils import Parameters
from cox.store import Store
import config
from typing import Any
from sklearn.linear_model import LogisticRegression

from .stats import stats
from ..oracle import oracle
from ..grad import TruncatedBCE, TruncatedCE
from ..train import train_model
from ..utils.helpers import Bounds
from ..utils import defaults
from ..utils.datasets import DataSet, TENSOR_REQUIRED_ARGS, TENSOR_OPTIONAL_ARGS


class TruncatedLogisticRegression(stats):
    """
    """
    def __init__(
            self,
            phi: oracle,
            alpha: float,
            args: Parameters,
            bias: bool=True,
            scale: float = None,
            device: str="cpu",
            multi_class='ovr',
            store: Store=None,
            table: str=None,
            **kwargs):
        """
        """
        super(LogisticRegression).__init__()
        # instance variables
        self._log_reg, self.projectioin_set = None, None
        self.phi = phi
        self.alpha = alpha
        self.bias = bias 
        self.scale = scale
        self.device = device
        self.multi_class = multi_class
        self.store, self.table = store, table

        # add membership oracle to algorithm hyperparameters
        args.__setattr__('phi', self.phi)
        args.__setattr__('alpha', self.alpha)
        args.__setattr__('device', self.device)
        config.args = defaults.check_and_fill_args(args, defaults.LOGISTIC_ARGS, TensorDataset)

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

        # empirical estimates for logistic regression
        # standard_log_reg = LogisticRegression_(penalty='none', fit_intercept=self.bias, multi_class=self.multi_class)
        # standard_log_reg.fit(X, y.flatten())

        # use standard predictions as empirical estimates
        self._log_reg = ch.nn.Linear(in_features=X.size(1), out_features=int(ch.max(y) + 1), bias=self.bias)
        # print("coef: ", standard_log_reg.coef_)
        # print("coef shape: ", standard_log_reg.coef_.shape)
        # self._log_reg.weight = ch.nn.Parameter(Tensor(standard_log_reg.coef_))
        # if self.bias: 
        #     self._log_reg.bias = ch.nn.Parameter(Tensor(standard_log_reg.intercept_))

        # intialize loss function & iteration hook and add to hyperparameters
        config.args.__setattr__('custom_criterion', TruncatedCE.apply if self.multi_class == 'multinomial' else TruncatedBCE.apply)
        # config.args.__setattr__('iteration_hook', TruncLogRegIterationHook(self._log_reg, config.args.alpha, config.args.radius))
        # run PGD to predict actual estimates
        return train_model(config.args, self._log_reg, loaders, store=self.store, table=self.table)


class TruncLogRegIterationHook:
    """
    Censored logistic regression projection set
    """
    def __init__(self, emp_log_reg, alpha, r, clamp=True):
        """
        Args:
            emp_log_reg (nn.Linear): empirical logistic regression weights and bias
        """
        # instance variables
        self.emp_log_reg = emp_log_reg
        self.alpha = alpha
        self.r = r
        self.clamp = clamp

        # projection set radius
        self.radius = config.args.radius * (ch.sqrt(2.0 * ch.log(Tensor([1.0 / self.alpha]))))
        if self.clamp:
            self.weight_bounds = Bounds((self.emp_log_reg.weight.data - self.r).flatten(),
                                        (self.emp_log_reg.weight.data + self.r).flatten())
            if self.emp_log_reg.bias:
                self.bias_bounds = Bounds(float(self.emp_log_reg.bias.data - config.args.radius),
                                          float(self.emp_log_reg.bias.data + config.args.radius))

    def __call__(self, model, i, loop_type, inp, target):
        if self.clamp:
            # project weight coefficients
            model.weight.data = ch.stack([ch.clamp(model.weight.data[i], float(self.weight_bounds.lower[i]),
                                                         float(self.weight_bounds.upper[i])) for i in
                                                range(model.weight.size(0))])
            # project bias coefficient
            if model.bias:
                model.bias.data = ch.clamp(model.bias, self.bias_bounds.lower, self.bias_bounds.upper).reshape(
                    model.bias.size())
        else:
            pass