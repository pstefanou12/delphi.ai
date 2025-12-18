"""
Truncated Exponential Distribution.
"""

import torch as ch
from typing import Callable
import logging

from .truncated_exponential_family_distributions import TruncatedExponentialFamilyDistribution
from ..delphi_logger import delphiLogger
from ..grad import ExponentialFamilyPoisson, calc_poiss_suff_stat 
from ..utils.helpers import Parameters
from ..utils.defaults import check_and_fill_args, TRUNC_POISS_DEFAULTS


class TruncatedPoisson(TruncatedExponentialFamilyDistribution):
    """
    Model for truncated exponential distributions to be passed into trainer.
    """
    def __init__(self, 
                args: Parameters,
                phi: Callable, 
                alpha: float,
                dims: int,): 
        """
        Args: 
            args (cox.utils.Parameters) : parameter object holding hyperparameters
        """
        assert isinstance(args, Parameters), "args is type: {}. expecting args to be type delphi.utils.helpers.Parameters"
        args = check_and_fill_args(args, TRUNC_POISS_DEFAULTS)
        
        logger = delphiLogger() if args.verbose else delphiLogger(level=logging.CRITICAL)
        super().__init__(args, phi, alpha, dims, ExponentialFamilyPoisson, calc_poiss_suff_stat, logger)

    def _reparameterize_nat_form(self, 
                                 theta):
        return ch.log(theta) 

    def _reparameterize_canon_form(self, 
                        theta): 
        return ch.exp(theta)
    
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
        return "truncated poisson distribution"



