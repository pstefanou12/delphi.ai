
"""
Truncated multivariate normal distribution without oracle access (ie. unknown truncation set)
"""

import torch as ch
from torch import Tensor
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data import DataLoader
from cox.utils import Parameters
import config
from typing import Any

from .stats import stats
from .unknown_truncation_normal import TruncatedMultivariateNormalNLL, TruncatedNormalProjectionSet
from ..oracle import oracle
from ..train import train_model
from ..utils.helpers import Exp_h


class truncated_multivariate_normal(stats):
    """
    Truncated multivariate normal distribution class.
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
        super(truncated_multivariate_normal, self).__init__()
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
            'step_lr': step_lr,
            'gamma': gamma,
            'momentum': weight_decay,
            'weight_decay': momentum,
            'device': device,
        })
        self._multivariate_normal = None
        # intialize loss function and add custom criterion to hyperparameters
        self.criterion = TruncatedMultivariateNormalNLL.apply
        config.args.__setattr__('custom_criterion', self.criterion)

    def fit(
            self,
            S: DataLoader):
        # initialize model with empiricial estimates
        self._multivariate_normal = MultivariateNormal(S.dataset.loc, S.dataset.covariance_matrix.unsqueeze(0))
        # keep track of gradients for mean and covariance matrix
        self._multivariate_normal.loc.requires_grad, self._multivariate_normal.covariance_matrix.requires_grad = True, True
        # initialize projection set and add iteration hook to hyperparameters
        self.projection_set = TruncatedMultivariateNormalProjectionSet(self._multivariate_normal.loc,
                                                                       self._multivariate_normal.covariance_matrix)
        config.args.__setattr__('iteration_hook', self.projection_set)
        # exponent class
        self.exp_h = Exp_h(S.dataset.loc, S.dataset.covariance_matrix)
        config.args.__setattr__('exp_h', self.exp_h)
        # run PGD to predict actual estimates
        return train_model(self._multivariate_normal, (S, None),
                           update_params=[self._multivariate_normal.loc, self._multivariate_normal.covariance_matrix])


class TruncatedMultivariateNormalProjectionSet(TruncatedNormalProjectionSet):
    """
    Truncated multivariate normal distribution with unknown truncation projection set.
    """

    def __init__(self, emp_loc, emp_covariance_matrix):
        """
        Args:
            emp_loc (torch.Tensor): empirical mean
            emp_scale (torch.Tensor): empirical variance
            alpha (torch.Tensor): lower bound on survival probability for distribution
            r (float): projection set radius
            clamp (bool): boolean for clamp heuristic
        """
        super().__init__(emp_loc, emp_covariance_matrix.svd()[1])

    def __call__(self, M, i, loop_type, inp, target):
        if config.args.clamp:
            u, s, v = M.covariance_matrix.svd()  # decompose covariance estimate
            M.loc.data = ch.cat(
                [ch.clamp(M.loc[i], self.loc_bounds.lower[i], self.loc_bounds.upper[i]).unsqueeze(0) for i in
                 range(M.loc.shape[0])])
            M.covariance_matrix.data = u.matmul(ch.diag(ch.cat(
                [ch.clamp(s[i], self.scale_bounds.lower[i], self.scale_bounds.upper[i]).unsqueeze(0) for i in
                 range(s.shape[0])]))).matmul(v.t())
        else:
            pass