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
            'custom_class': TruncatedRegression,
            'transform_train': None,
            'transform_test': None,
        }
        ds = DataSet('truncated_logistic_regression', TRUNC_LOG_REG_REQUIRED_ARGS, TRUNC_LOG_REG_OPTIONAL_ARGS, data_path=None,
                     **ds_kwargs)
        loaders = ds.make_loaders(workers=config.args.workers, batch_size=config.args.batch_size)
        # initialize model with empirical estimates
        self._emp_log_reg = ch.nn.Linear(loaders[0].dataset.log_reg.coef_, bias=loaders[0].dataset.log_reg.intercept_)
        self.projection_set = TruncatedLogisticRegressionProjectionSet(self._emp_log_reg)
        config.args.__setattr__('iteration_hook', self.projection_set)
        # run PGD to predict actual estimates
        return train_model(config.args, self._emp_log_reg, loaders)


class TruncatedBCE(ch.autograd.Function):
    @staticmethod
    def forward(ctx, pred, targ):
        ctx.save_for_backward(pred, targ)
        loss = ch.nn.BCEWithLogitsLoss()
        return loss(pred, targ)

    @staticmethod
    def backward(ctx, grad_output):
        pred, targ = ctx.saved_tensors

        # logistic distribution
        base_distribution = Uniform(0, 1)
        transforms_ = [SigmoidTransform().inv]
        logistic = TransformedDistribution(base_distribution, transforms_)

        stacked = pred[None, ...].repeat(config.args.num_samples, 1, 1)
        # add noise
        noised = stacked + logistic.sample(stacked.size())
        # filter
        filtered = ch.stack([config.args.phi(batch).unsqueeze(1) for batch in noised]).float()
        out = (noised * filtered).sum(dim=0) / (filtered.sum(dim=0) + config.args.eps)
        grad = ch.where(ch.abs(out) > config.args.eps, sig(out), targ) - targ
        return grad / pred.size(0), -grad / pred.size(0)


class TruncatedLogisticRegressionProjectionSet:
    """
    Censored logistic regression projection set
    """
    def __init__(self, emp_log_reg):
        """
        Args:
            emp_log_reg (nn.Linear): empirical logistic regression weights and bias
        """
        self.emp_log_reg = emp_log_reg
        self.radius = config.args.radius * (ch.sqrt(2.0 * ch.log(1.0 / config.args.alpha)))
        if config.args.clamp:
            self.weight_bounds = Bounds((self.emp_log_reg.weight.data - config.args.radius).flatten(),
                                        (self.emp_log_reg.weight.data + config.args.radius).flatten())
            if self.emp_log_reg.bias:
                self.bias_bounds = Bounds(float(self.emp_log_reg.bias.data - config.args.radius),
                                          float(self.emp_log_reg.bias.data + config.args.radius))

    def __call__(self, est_log_reg, i, loop_type, inp, target):
        if config.args.clamp:
            # project weight coefficients
            est_log_reg.weight.data = ch.stack([ch.clamp(est_log_reg.weight.data[i], float(self.weight_bounds.lower[i]),
                                                         float(self.weight_bounds.upper[i])) for i in
                                                range(est_log_reg.weight.size(0))])
            # project bias coefficient
            if est_log_reg.bias:
                est_log_reg.bias.data = ch.clamp(est_log_reg.bias, self.bias_bounds.lower, self.bias_bounds.upper).reshape(
                    est_log_reg.bias.size())
        else:
            pass