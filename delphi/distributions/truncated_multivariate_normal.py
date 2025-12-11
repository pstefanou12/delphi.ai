"""
Truncated multivariate normal distribution with oracle access (ie. known truncation set).
"""

import torch as ch
from torch import Tensor
from torch.distributions.multivariate_normal import MultivariateNormal
import math
import torch.nn as nn
from typing import Callable, Optional
import logging

from .distributions import distributions
from ..delphi_logger import delphiLogger
from ..utils.datasets import TruncatedExponentialDistributionDataset, make_train_and_val_distr
from ..grad import TruncatedExponentialFamilyDistributionNLL, ExponentialFamilyMultivariateNormal, calc_multi_norm_suff_stat 
from ..trainer import Trainer
from ..utils.helpers import Parameters, cov
from ..utils.defaults import check_and_fill_args, TRUNC_MULTI_NORM_DEFAULTS


class TruncatedMultivariateNormal(distributions):
    """
    Truncated multivariate normal distribution class with known truncation set.
    """
    def __init__(self,
                args: Parameters,
                phi: Callable, 
                alpha: float,
                dims: int,
                covariance_matrix: Optional[ch.Tensor] = None,
                sampler: Callable = None):
        """
        """
        assert isinstance(args, Parameters), "args is type: {}. expecting args to be type delphi.utils.helpers.Parameters"
        args = check_and_fill_args(args, TRUNC_MULTI_NORM_DEFAULTS)
        
        logger = delphiLogger() if args.verbose else delphiLogger(level=logging.CRITICAL)
        super().__init__(args, logger)
 
        self.phi = phi
        self.alpha = alpha
        self.dims = dims
        self.covariance_matrix = covariance_matrix
        
        del self.criterion
        self.criterion = TruncatedExponentialFamilyDistributionNLL.apply

        self.emp_loc, self.emp_covariance_matrix = None, None
        self.S = None

        self.best_covariance_matrix, self.best_loc, self.best_loss = None, None, None
        self.final_covariance_matrix, self.final_loc, self.final_loss = None, None, None

    def fit(self, S: Tensor):
        """
        """
        assert isinstance(S, Tensor), "S is type: {}. expected type torch.Tensor.".format(type(S))
        assert S.size(0) > S.size(1), "input expected to be shape num samples by dimenions, current input is size {}.".format(S.size()) 
        assert self.args.batch_size <= self.args.num_samples, "batch size must be smaller than or equal to the number of samples being sampled"
        
        self.S = S
        self.criterion_params = [self.phi, self.dims, ExponentialFamilyMultivariateNormal, calc_multi_norm_suff_stat, self.args.num_samples, self.args.eps]
        self.train_loader_, self.val_loader_ = make_train_and_val_distr(self.args, 
                                                                        self.S, 
                                                                        TruncatedExponentialDistributionDataset, 
                                                                        {'calc_suff_stat': calc_multi_norm_suff_stat})
        self.trainer = Trainer(
            self,
            self.args, 
            self.logger
        )
        self.trainer.train_model(self.train_loader_, self.val_loader_)
        return self
            
    def _calc_emp_model(self): 
        # empirical mean and variance
        S = self.train_loader_.dataset.S
        self.emp_loc = ch.mean(S, dim=0)
        if self.covariance_matrix is not None: 
            self.emp_covariance_matrix = self.covariance_matrix
        else:   
            self.emp_covariance_matrix = cov(S)
        self.model = MultivariateNormal(self.emp_loc, self.emp_covariance_matrix)
        self.dims = self.emp_loc.size(0)
            
    def pretrain_hook(self, train_loader):
        self._calc_emp_model()
        # parameterize projection set
        self.radius = self.args.r * math.log(1.0 / self.alpha) + 12
        self.eigenvalue_lower_bound = self.alpha ** 2
        self.eigenvalue_lower_bound = 1e-2
        if self.covariance_matrix is not None:
            T = self.covariance_matrix.clone().inverse()
        else:
            T = self.emp_covariance_matrix.clone().inverse()
        v = self.emp_loc.clone() @ T

        self.emp_v = v.clone()
        self.emp_T = T.clone()

        # Initialize empirical model 
        self.register_parameter('T', nn.Parameter(T.flatten()))
        if self.covariance_matrix is not None: self.T.requires_grad = False # remove from the computation graph
        self.register_parameter('v', nn.Parameter(v))

    def __call__(self, batch, targ):
        """
        Training step for defined model.
        Args: 
            i (int) : gradient step or epoch number
            batch (Iterable) : iterable of inputs that 
        """
        return ch.cat([self.T, self.v]) 
    
    def project_to_psd(self, M, eps=1e-6):
        """Projects a symmetric matrix M onto the Positive Semi-Definite (PSD) cone."""
        L, Q = ch.linalg.eigh(M)
        L_clipped = ch.clamp(L, min=eps)
        return Q @ ch.diag_embed(L_clipped) @ Q.T

    def step_post_hook(self, 
                       optimizer, 
                       args, 
                       kwargs) -> None:
        with ch.no_grad():
            # import ipdb; ipdb.set_trace()
            T = self.T.clone().view(self.dims, self.dims)
            v = self.v.clone()

            # --- 1) Project mean into L2 ball ---
            loc_diff = self.emp_v - v
            dist = ch.norm(loc_diff)

            if dist > self.radius:
                v = self.emp_v - loc_diff / dist * self.radius

            # --- 2) Frobenius ball projection for T ---
            cov_diff = T - self.emp_T
            frob_norm = ch.linalg.norm(cov_diff, ord='fro')

            if frob_norm > self.radius:
                T = self.emp_T + cov_diff * (self.radius / frob_norm)

            # Symmetrize after projection (important!)
            T = 0.5 * (T + T.T)
            # --- 3) Final PSD projection ---
            T = self.project_to_psd(T, eps=self.eigenvalue_lower_bound)
            # Symmetrize again after PSD projection
            T = 0.5 * (T + T.T)
            self.T.copy_(T.flatten())
            self.v.copy_(v)
            # if T[0, 1] != T[1, 0]: import ipdb; ipdb.set_trace()
            
            from delphi.utils.helpers import is_psd
            covariance_matrix = self.T.view(self.dims, self.dims).inverse()
            if not is_psd(covariance_matrix): import ipdb; ipdb.set_trace()

    def post_training_hook(self): 
        self.args.r *= self.args.rate
        # reparameterize distribution
        self.best_covariance_matrix, self.best_loc, self.best_theta_loss = *self._reparameterize(self.trainer.best_params), self.trainer.best_loss
        self.final_covariance_matrix, self.final_loc, self.final_theta_loss = *self._reparameterize(self.trainer.final_params), self.trainer.final_loss
        self.ema_covariance_matrix, self.ema_loc = self._reparameterize(self.trainer.ema_params)
        self.avg_covariance_matrix, self.avg_loc = self._reparameterize(self.trainer.avg_params)

    def parameters_(self):
        if self.args.covariance_matrix_lr is not None: 
            return [
                {'params': self.T, 'lr': self.args.covariance_matrix_lr},   
                {'params': self.v, 'lr': self.args.lr},
            ]
        return self.parameters()

    def _reparameterize(self, 
                        theta): 
        if self.covariance_matrix is not None: 
            T = self.T.view(self.dims, self.dims)
            v = theta 
        else:
            T = theta[:self.dims**2].resize(self.dims, self.dims)
            v = theta[self.dims**2:] 

        covariance_matrix = T.inverse()
        loc = v @ covariance_matrix

        return covariance_matrix, loc
    
    @property 
    def best_loc_(self): 
        """
        Returns the best mean vector estimate for the multivariate normal distribution based off of the loss function.
        """
        return self.best_loc
    
    @property
    def best_covariance_matrix_(self): 
        """
        Returns the best covariance matrix estimate for the multivariate normal distribution based off of the loss function.
        """
        return self.best_covariance_matrix 
    
    @property
    def final_loc_(self): 
        """
        Returns the final mean vector estimate for the multivariate normal distribution based off of the loss function.
        """
        return self.final_loc

    @property
    def final_covariance_matrix_(self): 
        """
        Returns the final covariance matrix estimate for the multivariate normal distribution based off of the loss function.
        """
        return self.final_covariance_matrix
    
    @property
    def ema_loc_(self): 
        """
        Returns the ema mean vector estimate for the multivariate normal distribution based off of the loss function.
        """
        return self.ema_loc

    @property
    def ema_covariance_matrix_(self): 
        """
        Returns the ema covariance matrix estimate for the multivariate normal distribution based off of the loss function.
        """
        return self.ema_covariance_matrix

    @property
    def avg_loc_(self): 
        """
        Returns the avg mean vector estimate for the multivariate normal distribution based off of the loss function.
        """
        return self.avg_loc

    @property
    def avg_covariance_matrix_(self): 
        """
        Returns the avg covariance matrix estimate for the multivariate normal distribution based off of the loss function.
        """
        return self.avg_covariance_matrix

    def calc_suff_stat(self, S): 
        return calc_multi_norm_suff_stat(S)
    
    def calculate_loss(self, S): 
        return self.criterion(self.theta, self.calc_suff_stat(S), *self.criterion_params)
