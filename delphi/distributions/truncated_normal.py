"""
Truncated multivariate normal distribution with oracle access (ie. known truncation set).
"""

import torch as ch
from typing import Callable, Optional
from functools import partial
import logging

from .truncated_exponential_family_distributions import TruncatedExponentialFamilyDistribution
from .truncated_multivariate_normal import TruncatedMultivariateNormalKnownCovariance, TruncatedMultivariateNormalUnknownCovariance
from ..delphi_logger import delphiLogger
from ..grad import ExponentialFamilyMultivariateNormal, ExponentialFamilyMultivariateNormalKnownCovariance, calc_multi_norm_suff_stat_known_cov, calc_multi_norm_suff_stat 
from ..utils.helpers import Parameters 
from ..utils.defaults import check_and_fill_args, TRUNC_MULTI_NORM_DEFAULTS


class TruncatedNormalKnownCovariance(TruncatedMultivariateNormalKnownCovariance):
    """
    Truncated multivariate normal distribution class with known truncation set.
    """
    def __init__(self,
                args: Parameters,
                phi: Callable, 
                alpha: float,
                dims: int,
                variance: Optional[ch.Tensor],
                sampler: Callable = None):
        """
        """
        super().__init__(args, phi, alpha, dims,  variance, sampler)
    
    def __str__(self): 
        return "truncated normal distribution known covariance"
    
class TruncatedNormalUnknownVariance(TruncatedMultivariateNormalUnknownCovariance):
    """
    Truncated multivariate normal distribution class with known truncation set.
    """
    def __init__(self,
                args: Parameters,
                phi: Callable, 
                alpha: float,
                dims: int,
                sampler: Callable = None):
        """
        """
        super().__init__(args, phi, alpha, dims, sampler) 
    
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
        return "truncated normal distribution"

"""
Truncated normal distribution class with known truncation set.
"""
def TruncatedNormal(
                args: Parameters,
                phi: Callable, 
                alpha: float,
                dims: int,
                variance: Optional[ch.Tensor] = None,
                sampler: Callable = None):
    """
    """
    assert isinstance(args, Parameters), "args is type: {}. expecting args to be type delphi.utils.helpers.Parameters"
    args = check_and_fill_args(args, TRUNC_MULTI_NORM_DEFAULTS)
    if variance is not None: 
        return TruncatedNormalKnownCovariance(args, phi, alpha, dims, variance, sampler)
    else: 
        return TruncatedNormalUnknownVariance(args, phi, alpha, dims, sampler)






