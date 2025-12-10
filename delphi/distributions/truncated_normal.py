"""
Truncated normal distribution with oracle access (ie. known truncation set)
"""

from torch import Tensor
from typing import Callable

from .truncated_multivariate_normal import TruncatedMultivariateNormal
from ..utils.datasets import make_train_and_val_distr
from ..utils.helpers import Parameters 


class TruncatedNormal(TruncatedMultivariateNormal):
    """
    Truncated normal distribution class with known truncation set.
    """
    def __init__(self, 
                 args: Parameters,
                 phi: Callable, 
                 alpha: float,
                 dims: int, 
                 variance: Tensor=None, 
                 sampler=None):
        """
        Args:
           args (delphi.utils.helpers.Parameters): hyperparameters for censored algorithm 
        """
        super().__init__(args, phi, alpha, dims, covariance_matrix=variance, sampler=sampler)

    @property 
    def best_variance_(self): 
        """
        Returns the variance for the normal distribution.
        """
        return self.best_covariance_matrix_ 
    
    @property
    def ema_variance_(self): 
        """
        Returns the ema variance estimate for the multivariate normal distribution based off of the loss function.
        """
        return self.ema_covariance_matrix

    @property
    def avg_variance_(self): 
        """
        Returns the avg variance estimate for the multivariate normal distribution based off of the loss function.
        """
        return self.avg_covariance_matrix