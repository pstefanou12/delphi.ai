"""
Truncated Exponential Distribution.
"""

import torch as ch
from torch import Tensor
import torch.nn as nn
from typing import Callable
import math
import logging

from .distributions import distributions
from ..delphi_logger import delphiLogger
from ..utils.datasets import TruncatedExponentialDistributionDataset, make_train_and_val_distr
from ..grad import TruncatedExponentialFamilyDistributionNLL, ExponentialFamilyExponential, calc_exp_suff_stat 
from ..trainer import Trainer
from ..utils.helpers import Parameters
from ..utils.defaults import check_and_fill_args, TRUNC_EXP_DEFAULTS


class TruncatedExponential(distributions):
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
        args = check_and_fill_args(args, TRUNC_EXP_DEFAULTS)
        
        logger = delphiLogger() if args.verbose else delphiLogger(level=logging.CRITICAL)
        super().__init__(args, logger)

        self.phi = phi
        self.alpha = alpha
        self.dims = dims

        del self.criterion
        self.criterion = TruncatedExponentialFamilyDistributionNLL.apply

        self.emp_p = None
        self.S = None

        self.best_lambda, self.best_loss = None, None
        self.final_lambda, self.final_loss = None, None
        self.ema_lambda = None
        self.avg_lambda = None

    def fit(self, S: Tensor):
        """
        """
        assert isinstance(S, Tensor), "S is type: {}. expected type torch.Tensor.".format(type(S))
        assert S.size(0) > S.size(1), "input expected to be shape num samples by dimenions, current input is size {}.".format(S.size()) 
        assert self.args.batch_size <= self.args.num_samples, "batch size must be smaller than or equal to the number of samples being sampled"
        
        self.S = S
        self.criterion_params = [self.phi, self.dims, ExponentialFamilyExponential, calc_exp_suff_stat, self.args.num_samples, self.args.eps]
        self.train_loader_, self.val_loader_ = make_train_and_val_distr(self.args, 
                                                                        self.S, 
                                                                        TruncatedExponentialDistributionDataset, 
                                                                        {'calc_suff_stat': calc_exp_suff_stat})
        self.trainer = Trainer(
            self,
            self.args, 
            self.logger
        )
        self.trainer.train_model(self.train_loader_, self.val_loader_)
        return self
    
    def _calc_emp_model(self): 
        self.S = self.train_loader_.dataset.S
        self.emp_lambda_ = 1.0/self.S.mean(0)
        self.emp_theta = -self.emp_lambda_

    def pretrain_hook(self, train_loader):
        self._calc_emp_model()
        self.radius = self.args.r * math.log((1 / self.alpha) ** .5)
        self.register_parameter('theta', nn.Parameter(self.emp_theta))

    def __call__(self, batch, targ):
        """
        Training step for defined model.
        Args: 
            i (int) : gradient step or epoch number
            batch (Iterable) : iterable of inputs that 
        """
        return self.theta

    def step_post_hook(self, 
                       optimizer, 
                       args, 
                       kwargs) -> None:
        """
        Iteration hook for defined model. Method is called after each 
        training update.
        Args:

        """
        theta_diff = (self.theta - self.emp_theta)[...,None].norm()
        if theta_diff > self.radius: 
            theta_diff = theta_diff.renorm(p=2, dim=0, maxnorm=self.radius).flatten()
            self.theta.copy_(self.emp_theta + theta_diff)

    def post_training_hook(self): 
        self.args.r *= self.args.rate
        # remove distribution from the computation graph
        self.best_lambda, self.best_loss = self._reparameterize(self.trainer.best_params), self.trainer.best_loss
        self.final_lambda, self.final_loss = self._reparameterize(self.trainer.final_params), self.trainer.final_loss 
        self.ema_lambda = self._reparameterize(self.trainer.ema_params)
        self.avg_lambda = self._reparameterize(self.trainer.avg_params)

    def _reparameterize(self, 
                        theta): 
        return -theta
    
    @property
    def best_lambda_(self): 
        return self.best_lambda
    
    @property
    def final_lambda_(self): 
        return self.final_lambda
    
    @property
    def ema_lambda_(self): 
        return self.ema_lambda
    
    @property
    def avg_lambda_(self): 
        return self.avg_lambda



