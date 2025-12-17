"""
Truncated Exponential Distribution.
"""

import torch as ch
from typing import Callable
from functools import partial
import logging

from .truncated_exponential_family_distributions import TruncatedExponentialFamilyDistribution
from ..delphi_logger import delphiLogger
from ..grad import ExponentialFamilyWeibull, calc_weibull_suff_stat 
from ..utils.helpers import Parameters
from ..utils.defaults import check_and_fill_args, TRUNC_EXP_DEFAULTS


class TruncatedWeibull(TruncatedExponentialFamilyDistribution):
    """
    Model for truncated exponential distributions to be passed into trainer.
    """
    def __init__(self, 
                args: Parameters,
                phi: Callable, 
                alpha: float,
                dims: int, 
                k: int): 
        """
        Args: 
            args (cox.utils.Parameters) : parameter object holding hyperparameters
        """
        assert isinstance(args, Parameters), "args is type: {}. expecting args to be type delphi.utils.helpers.Parameters"
        args = check_and_fill_args(args, TRUNC_EXP_DEFAULTS)
        
        logger = delphiLogger() if args.verbose else delphiLogger(level=logging.CRITICAL)
        super().__init__(args, phi, alpha, dims, partial(ExponentialFamilyWeibull, k), partial(calc_weibull_suff_stat, k), logger)
        self.k = k

    def _constraints(self, theta):
        return ch.clamp(theta, max=-1e-6)
    
    def _reparameterize_nat_form(self, 
                                 theta): 
        return -1.0/theta.pow(self.k)
    
    def _reparameterize_canon_form(self, 
                                   theta): 
        return (-1/theta).pow(1/self.k)
    
    @property
    def best_lambda_(self): 
        return self.best_params
    
    @property
    def final_lambda_(self): 
        return self.final_params
    
    @property
    def ema_lambda_(self): 
        return self.ema_params
    
    @property
    def avg_lambda_(self): 
        return self.avg_params
    
    def __str__(self): 
        return "truncated weibull distribution"



