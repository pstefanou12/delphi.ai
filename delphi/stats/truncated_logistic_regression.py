import torch as ch
from torch import Tensor
from torch.nn import Linear
from torch.utils.data import DataLoader
from torch import sigmoid as sig
from torch.distributions import Uniform
from torch.distributions.transforms import SigmoidTransform
from torch.distributions.transformed_distribution import TransformedDistribution
from cox.utils import Parameters
import config
from typing import Any

from .stats import stats
from ..oracle import oracle
from ..grad import TruncatedBCE
from ..train import train_model
from ..utils.helpers import Bounds
from ..utils import defaults
from ..utils.datasets import DataSet, TRUNC_LOG_REG_OPTIONAL_ARGS, TRUNC_LOG_REG_REQUIRED_ARGS, TruncatedLogisticRegression


class truncated_logistic_regression(stats):
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
            **kwargs):
        """
        """
        super(truncated_logistic_regression).__init__()
        config.args = defaults.check_and_fill_args(args, defaults.LOGISTIC_ARGS, TruncatedLogisticRegression)
        # initialize hyperparameters for algorithm
        self._emp_log_reg = None
        self.projection_set = None
        # intialize loss function and add custom criterion to hyperparameters
        self.criterion = TruncatedBCE.apply
        config.args.__setattr__('custom_criterion', self.criterion)

    def fit(self, X: Tensor, y: Tensor):
        """
        """
        # create dataset and dataloader
        ds_kwargs = {
            'custom_class_args': {
                'X': X, 'y': y, 'bias': config.args.bias},
            'custom_class': TruncatedLogisticRegression,
            'transform_train': None,
            'transform_test': None,
        }
        ds = DataSet('truncated_logistic_regression', TRUNC_LOG_REG_REQUIRED_ARGS, TRUNC_LOG_REG_OPTIONAL_ARGS, data_path=None,
                     **ds_kwargs)
        loaders = ds.make_loaders(workers=config.args.workers, batch_size=config.args.batch_size)
        # initialize model with empirical estimates
        self._emp_log_reg = ch.nn.Linear(in_features=loaders[0].dataset.log_reg.coef_.shape[1], out_features=1, bias=loaders[0].dataset.log_reg.intercept_)
        self.iteration_hook = TruncLogRegIterationHook(self._emp_log_reg, config.args.alpha, config.args.radius)
        config.args.__setattr__('iteration_hook', self.iteration_hook)
        # run PGD to predict actual estimates
        return train_model(config.args, self._emp_log_reg, loaders)


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
        self.radius = config.args.radius * (ch.sqrt(2.0 * ch.log(1.0 / self.alpha)))
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