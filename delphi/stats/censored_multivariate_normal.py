import torch as ch
from torch import Tensor
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data import DataLoader
from cox.utils import Parameters
import config

from .stats import stats
from ..oracle import oracle
from .censored_normal import CensoredMultivariateNormalNLL, CensoredNormalProjectionSet
from ..train import train_model


class censored_multivariate_normal(stats):
    """
    Censored multivariate distribution class.
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
            tol: float=1e-1,
            **kwargs):
        """
        """
        super().__init__()
        # initialize hyperparameters for algorithm
        config.args = Parameters({
            'phi': phi,
            'epochs': epochs,
            'lr': lr,
            'num_samples': num_samples,
            'alpha': Tensor([alpha]),
            'radius': Tensor([radius]),
            'clamp': clamp,
            'momentum': 0.0,
            'weight_decay': 0.0,
            'tol': tol,
        })
        self._multivariate_normal = None
        self.projection_set = None
        # intialize loss function and add custom criterion to hyperparameters
        self.criterion = CensoredMultivariateNormalNLL.apply
        config.args.__setattr__('custom_criterion', self.criterion)

    def fit(
            self,
            S: DataLoader):
        """
        """
        # initialize model with empiricial estimates
        self._multivariate_normal = MultivariateNormal(S.dataset.loc, S.dataset.covariance_matrix)
        # keep track of gradients for mean and covariance matrix
        self._multivariate_normal.loc.requires_grad, self._multivariate_normal.covariance_matrix.requires_grad = True, True
        # initialize projection set and add iteration hook to hyperparameters
        self.projection_set = CensoredMultivariateNormalProjectionSet(self._multivariate_normal.loc,
                                                                      self._multivariate_normal.covariance_matrix)
        config.args.__setattr__('iteration_hook', self.projection_set)
        # run PGD to predict actual estimates
        return train_model(self._multivariate_normal, (S, None),
                           update_params=[self._multivariate_normal.loc, self._multivariate_normal.covariance_matrix])


class CensoredMultivariateNormalProjectionSet(CensoredNormalProjectionSet):
    """
    Censored multivariate normal projection set
    """
    def __init__(self, emp_loc, emp_covariance_matrix):
        """
        Args:
            emp_loc (torch.Tensor): empirical mean
            emp_covariance_matrix (torch.Tensor): empirical covariance
        """
        super().__init__(emp_loc, emp_covariance_matrix.svd()[1])

    def __call__(self, M, i, loop_type, inp, target):
        if config.args.clamp:
            u, s, v = M.covariance_matrix.svd()  # decompose covariance estimate
            M.loc.data = ch.cat([ch.clamp(M.loc[i], self.loc_bounds.lower[i], self.loc_bounds.upper[i]).unsqueeze(0) for i in range(M.loc.shape[0])])
            M.covariance_matrix.data = u.matmul(ch.diag(ch.cat([ch.clamp(s[i], self.scale_bounds.lower[i], self.scale_bounds.upper[i]).unsqueeze(0) for i in range(s.shape[0])]))).matmul(v.t())
        else:
            pass