"""
Truncated normal distribution without oracle access (ie. unknown truncation set)
"""

import torch as ch
from torch import Tensor
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data import DataLoader
from cox.utils import Parameters
import config
from typing import Any

from .stats import stats
from ..oracle import oracle
from ..train import train_model
from ..utils.helpers import Bounds, Exp_h


class truncated_normal(stats):
    """
    Truncated normal distribution class.
    """
    def __init__(
            self,
            phi: oracle,
            alpha: float,
            epochs: int=50,
            lr: float=1e-1,
            num_samples: int=100,
            radius: float=2.0,
            clamp: bool=True,
            tol: float = 1e-1,
            custom_lr_multiplier: Any = None,
            step_lr: int = 10,
            gamma: float = .9,
            weight_decay: float = 0.0,
            momentum: float = 0.0,
            device: str = 'cpu',
            **kwargs):
        super(truncated_normal, self).__init__()
        config.args = Parameters({
            'phi': phi,
            'epochs': epochs,
            'lr': lr,
            'num_samples': num_samples,
            'alpha': Tensor([alpha]),
            'radius': Tensor([radius]),
            'clamp': clamp,
            'tol': tol,
            'custom_lr_multiplier': custom_lr_multiplier,
            'step_lr_gamma': step_lr,
            'gamma': gamma,
            'momentum': weight_decay,
            'weight_decay': momentum,
            'device': device,
        })
        self._normal = None
        # intialize loss function and add custom criterion to hyperparameters
        self.criterion = TruncatedMultivariateNormalNLL.apply
        config.args.__setattr__('custom_criterion', self.criterion)

    def fit(
            self,
            S: DataLoader):
        # initialize model with empiricial estimates
        self._normal = MultivariateNormal(S.dataset.loc, S.dataset.var.unsqueeze(0))
        # keep track of gradients for mean and covariance matrix
        self._normal.loc.requires_grad, self._normal.covariance_matrix.requires_grad = True, True
        # initialize projection set and add iteration hook to hyperparameters
        self.projection_set = TruncatedNormalProjectionSet(self._normal.loc, self._normal.covariance_matrix)
        config.args.__setattr__('iteration_hook', self.projection_set)
        # exponent class
        self.exp_h = Exp_h(S.dataset.loc, S.dataset.var.unsqueeze(0))
        config.args.__setattr__('exp_h', self.exp_h)
        # run PGD to predict actual estimates
        return train_model(self._normal, (S, None),
                           update_params=[self._normal.loc, self._normal.covariance_matrix])


class TruncatedMultivariateNormalNLL(ch.autograd.Function):
    """
    Computes the negative population log likelihood for truncated multivariate normal distribution with unknown truncation.
    """

    @staticmethod
    def forward(ctx, u, B, x, loc_grad, cov_grad):
        ctx.save_for_backward(u, B, x, loc_grad, cov_grad)
        return ch.ones(1)

    @staticmethod
    def backward(ctx, grad_output):
        u, B, x, loc_grad, cov_grad = ctx.saved_tensors
        exp = config.args.exp_h(u, B, x)
        psi = config.args.phi.psi_k(x).unsqueeze(1)
        return (loc_grad * exp * psi).mean(0), ((cov_grad.flatten(1) * exp * psi).unflatten(1, B.size())).mean(
            0), None, None, None


class TruncatedNormalProjectionSet:
    """
    Truncated normal distribution with unknown truncation projection set.
    """

    def __init__(self, emp_loc, emp_scale):
        """
        Args:
            emp_loc (torch.Tensor): empirical mean
            emp_scale (torch.Tensor): empirical variance
        """
        # projection set parameters
        self.emp_loc = emp_loc
        self.emp_scale = emp_scale
        self.radius = config.args.radius * ch.sqrt(ch.log(1.0 / config.args.alpha))

        # upper and lower bounds
        if config.args.clamp:
            self.loc_bounds, self.scale_bounds = Bounds(self.emp_loc - self.radius, self.emp_loc + self.radius), \
                                                 Bounds(ch.max(config.args.alpha.pow(2) / 12,
                                                               self.emp_scale - self.radius),
                                                        self.emp_scale + self.radius)
        else:
            pass

    def __call__(self, M, i, loop_type, inp, target):
        if config.args.clamp:
            M.loc.data = ch.clamp(M.loc.data, float(self.loc_bounds.lower), float(self.loc_bounds.upper))
            M.covariance_matrix.data = ch.clamp(M.covariance_matrix.data, float(self.scale_bounds.lower),
                                                float(self.scale_bounds.upper))
        else:
            pass