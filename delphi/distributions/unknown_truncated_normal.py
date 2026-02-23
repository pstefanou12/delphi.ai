"""
Truncated normal distribution without oracle access (ie. unknown truncation set)
"""

from typing import Callable, Optional
import torch as ch

from .unknown_truncated_multivariate_normal import UnknownTruncationMultivariateNormalUnknownCovariance, UnknownTruncationMultivariateNormalKnownCovariance 
from ..utils.helpers import Parameters


class UnknownTruncationNormalKnownVariance(UnknownTruncationMultivariateNormalKnownCovariance):
    """
    Truncated multivariate normal distribution class with known truncation set.
    """
    def __init__(self,
                args: Parameters,
                k: int,
                alpha: float,
                dims: int,
                variance: Optional[ch.Tensor]):
        """
        """
        super().__init__(args, k, alpha, dims,  variance)
    
    def __str__(self): 
        return "truncated normal distribution with unknown truncation and known variance"
    
    
class UnknownTruncatedNormalUnknownVariance(UnknownTruncationMultivariateNormalUnknownCovariance):
    """
    Truncated multivariate normal distribution class with known truncation set.
    """
    def __init__(self,
                args: Parameters,
                k: int, 
                alpha: float,
                dims: int):
        """
        """
        super().__init__(args, k, alpha, dims) 
    
    @property
    def best_variance_(self): 
        """
        Returns the best covariance matrix estimate for the multivariate normal distribution based off of the loss function.
        """
        return self.best_params[:self.dims**2].view(self.dims, self.dims)

    @property
    def final_variance_(self): 
        """
        Returns the final covariance matrix estimate for the multivariate normal distribution based off of the loss function.
        """
        self.final_params[:self.dims**2].view(self.dims, self.dims)

    @property
    def ema_variance_(self): 
        """
        Returns the ema covariance matrix estimate for the multivariate normal distribution based off of the loss function.
        """
        return self.ema_params[:self.dims**2].view(self.dims, self.dims)

    @property
    def avg_variance_(self): 
        """
        Returns the avg covariance matrix estimate for the multivariate normal distribution based off of the loss function.
        """
        return self.avg_params[:self.dims**2].view(self.dims, self.dims)
    
    def __str__(self): 
        return "truncated normal distribution with unknown truncation"


def UnknownTruncationNormal(
          args: Parameters,
          k: int,
          alpha: float,
          dims: int, 
          variance: Optional[ch.Tensor] = None):
    assert isinstance(args, Parameters), "args is type: {}. expecting args to be type delphi.utils.helpers.Parameters"
    if variance is not None: 
        return UnknownTruncationNormalKnownVariance(args, k, alpha, dims, variance)
    return UnknownTruncatedNormalUnknownVariance(args, k, alpha, dims)

