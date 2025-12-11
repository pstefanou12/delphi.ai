"""
Truncated Boolean Product Distributions.
"""

import torch as ch
from torch import Tensor
import torch.nn as nn
from typing import Callable
import math

from .distributions import distributions
from ..delphi_logger import delphiLogger
from ..utils.datasets import TruncatedExponentialDistributionDataset, make_train_and_val_distr
from ..grad import TruncatedExponentialDistributionNLL, delphiBooleanProduct, calc_bool_prod_suff_stat
from ..trainer import Trainer
from ..utils.helpers import Parameters
from ..utils.defaults import check_and_fill_args, TRUNC_BOOL_PROD_DEFAULTS


class TruncatedBooleanProduct(distributions):
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
        super().__init__(args, logger)

        self.phi = phi
        self.alpha = alpha
        self.dims = dims

        del self.criterion
        self.criterion = TruncatedExponentialDistributionNLL.apply

        self.emp_p = None
        self.S = None

        self.best_p, self.best_loss = None, None
        self.final_p, self.final_loss = None, None
        self.ema_p = None
        self.avg_p = None

    def fit(self, S: Tensor):
        """
        """
        assert isinstance(S, Tensor), "S is type: {}. expected type torch.Tensor.".format(type(S))
        assert S.size(0) > S.size(1), "input expected to be shape num samples by dimenions, current input is size {}.".format(S.size()) 
        assert self.args.batch_size <= self.args.num_samples, "batch size must be smaller than or equal to the number of samples being sampled"
        
        self.S = S
        self.criterion_params = [self.phi, self.dims, delphiBooleanProduct, calc_bool_prod_suff_stat, self.args.num_samples, self.args.eps]
        self.train_loader_, self.val_loader_ = make_train_and_val_distr(self.args, 
                                                                        self.S, 
                                                                        TruncatedExponentialDistributionDataset, 
                                                                        {'calc_suff_stat': calc_bool_prod_suff_stat})
        self.trainer = Trainer(
            self,
            self.args, 
            self.logger
        )
        self.trainer.train_model(self.train_loader_, self.val_loader_)
        return self
    
    def _calc_emp_model(self): 
        # percentage of points in S that have label 1
        self.S = self.train_loader_.dataset.S
        self.emp_p = self.S.mean(0)
        self.emp_z = ch.log(self.emp_p / (1 - self.emp_p))

    def pretrain_hook(self, train_loader):
        self._calc_emp_model()
        self.radius = self.args.r * math.log((1 / self.alpha) ** .5)

        self.register_parameter('z', nn.Parameter(self.emp_z))

    def __call__(self, batch, targ):
        """
        Training step for defined model.
        Args: 
            i (int) : gradient step or epoch number
            batch (Iterable) : iterable of inputs that 
        """
        return self.z 

    def step_post_hook(self, 
                       optimizer, 
                       args, 
                       kwargs) -> None:
        """
        Iteration hook for defined model. Method is called after each 
        training update.
        Args:

        """
        logit_diff = (self.z - self.emp_z)[...,None].norm()
        # import ipdb; ipdb.set_trace()
        if logit_diff > self.radius: 
            logit_diff = logit_diff.renorm(p=2, dim=0, maxnorm=self.radius).flatten()
            self.z.copy_(self.emp_z + logit_diff)

    def post_training_hook(self): 
        self.args.r *= self.args.rate
        # remove distribution from the computation graph
        self.best_p, self.best_loss = self._reparameterize(self.trainer.best_params), self.trainer.best_loss
        self.final_p, self.final_loss = self._reparameterize(self.trainer.final_params), self.trainer.final_loss 
        self.ema_p = self._reparameterize(self.trainer.ema_params)
        self.avg_p = self._reparameterize(self.trainer.avg_params)

    def _reparameterize(self, 
                        theta): 
        return ch.exp(theta) / (1 + ch.exp(theta))
    
    @property
    def best_p_(self): 
        return self.best_p
    
    @property
    def final_p_(self): 
        return self.final_p
    
    @property
    def ema_p_(self): 
        return self.ema_p
    
    @property
    def avg_p_(self): 
        return self.avg_p



