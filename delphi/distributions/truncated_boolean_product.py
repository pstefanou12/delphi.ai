"""
Truncated Boolean Product Distributions.
"""

import torch as ch
from typing import Callable
import logging

from .truncated_exponential_family_distributions import TruncatedExponentialFamilyDistribution
from ..delphi_logger import delphiLogger
from ..grad import ExponentialFamilyBooleanProduct, calc_bool_prod_suff_stat
from ..utils.helpers import Parameters
from ..utils.defaults import check_and_fill_args, TRUNC_BOOL_PROD_DEFAULTS


class TruncatedBooleanProduct(TruncatedExponentialFamilyDistribution):
    """
    Model for truncated boolean product distributions to be passed into trainer.
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
        args = check_and_fill_args(args, TRUNC_BOOL_PROD_DEFAULTS)
        
        logger = delphiLogger() if args.verbose else delphiLogger(level=logging.CRITICAL)
        super().__init__(args, phi, alpha, dims, ExponentialFamilyBooleanProduct, calc_bool_prod_suff_stat, logger)

    def _calc_emp_model(self): 
        S = self.train_loader_.dataset.S
        self.emp_params = S.mean(0)
        self.emp_theta = ch.log(self.emp_params / (1 - self.emp_params))

    def _reparameterize(self, 
                        theta): 
        return ch.exp(theta) / (1 + ch.exp(theta))
    
    @property
    def best_p_(self): 
        return self.best_params
    
    @property
    def final_p_(self): 
        return self.final_params
    
    @property
    def ema_p_(self): 
        return self.ema_params
    
    @property
    def avg_p_(self): 
        return self.avg_params



