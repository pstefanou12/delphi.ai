"""
Censored normal distribution with oracle access (ie. known truncation set)
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
from ..utils.helpers import Bounds, censored_sample_nll


class censored_normal(stats):
    """
    Censored normal distribution class.
    """
    def __init__(
            self,
            phi: oracle,
            alpha: float,
            epochs: int=10,
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
            device: str = "cpu",
            **kwargs):
        """
        """
        super(censored_normal, self).__init__()
        # initialize hyperparameters for algorithm
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
        self.criterion = CensoredMultivariateNormalNLL.apply
        config.args.__setattr__('custom_criterion', self.criterion)

    def fit(
            self,
            S: DataLoader):
        """
        """
        # initialize model with empiricial estimates
        self._normal = MultivariateNormal(S.dataset.loc, S.dataset.var.unsqueeze(0))
        # keep track of gradients for mean and covariance matrix
        self._normal.loc.requires_grad, self._normal.covariance_matrix.requires_grad = True, True
        # initialize projection set and add iteration hook to hyperparameters
        self.projection_set = CensoredNormalProjectionSet(self._normal.loc, self._normal.covariance_matrix)
        config.args.__setattr__('iteration_hook', self.projection_set)
        # run PGD to predict actual estimates
        return train_model(config.args, self._normal, (S, None),
                           update_params=[self._normal.loc, self._normal.covariance_matrix])


class CensoredMultivariateNormalNLL(ch.autograd.Function):
    """
    Computes the negative population log likelihood for censored multivariate normal distribution.
    """

    @staticmethod
    def forward(ctx, loc, covariance_matrix, x):
        ctx.save_for_backward(loc, covariance_matrix, x)
        return ch.zeros(1)

    @staticmethod
    def backward(ctx, grad_output):
        loc, covariance_matrix, x = ctx.saved_tensors
        # reparameterize distribution
        T = covariance_matrix.inverse()
        v = T.matmul(loc.unsqueeze(1)).flatten()
        # rejection sampling
        y = Tensor([])
        M = MultivariateNormal(v, T)
        while y.size(0) < x.size(0):
            s = M.sample(sample_shape=ch.Size([config.args.num_samples, ]))
            y = ch.cat([y, s[config.args.phi(s).nonzero(as_tuple=False).flatten()]])
        # calculate gradient
        grad = (-x + censored_sample_nll(y[:x.size(0)])).mean(0)
        return grad[loc.size(0) ** 2:], grad[:loc.size(0) ** 2].reshape(covariance_matrix.size()), None


class CensoredNormalProjectionSet:
    """
    Censored normal distribution projection set
    """
    def __init__(self, emp_loc, emp_scale):
        """
        Args:
            emp_loc (torch.Tensor): empirical mean
            emp_scale (torch.Tensor): empirical variance
        """
        self.emp_loc = emp_loc.clone().detach()
        self.emp_scale = emp_scale.clone().detach()
        self.radius = config.args.radius*(ch.log(1.0/config.args.alpha)/ch.square(config.args.alpha))
        # parameterize projection set
        if config.args.clamp:
            self.loc_bounds, self.scale_bounds = Bounds(self.emp_loc-self.radius, self.emp_loc+self.radius), \
             Bounds(ch.max(ch.square(config.args.alpha/12.0), self.emp_scale - self.radius), self.emp_scale + self.radius)
        else:
            pass

    def __call__(self, M, i, loop_type, inp, target):
        if config.args.clamp:
            M.loc.data = ch.clamp(M.loc.data, float(self.loc_bounds.lower), float(self.loc_bounds.upper))
            M.covariance_matrix.data = ch.clamp(M.covariance_matrix.data, float(self.scale_bounds.lower), float(self.scale_bounds.upper))
        else:
            pass