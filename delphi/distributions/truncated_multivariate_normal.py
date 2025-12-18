"""
Truncated multivariate normal distribution with oracle access (ie. known truncation set).
"""

import torch as ch
import torch.nn as nn
from typing import Callable, Optional
from functools import partial
import logging

from .truncated_exponential_family_distributions import TruncatedExponentialFamilyDistribution
from ..delphi_logger import delphiLogger
from ..grad import ExponentialFamilyMultivariateNormal, ExponentialFamilyMultivariateNormalKnownCovariance, calc_multi_norm_suff_stat_known_cov, calc_multi_norm_suff_stat 
from ..utils.helpers import Parameters 
from ..utils.defaults import check_and_fill_args, TRUNC_MULTI_NORM_DEFAULTS


class TruncatedMultivariateNormalKnownCovariance(TruncatedExponentialFamilyDistribution):
    """
    Truncated multivariate normal distribution class with known truncation set.
    """
    def __init__(self,
                args: Parameters,
                phi: Callable, 
                alpha: float,
                dims: int,
                covariance_matrix: Optional[ch.Tensor],
                sampler: Callable = None):
        """
        """
        assert isinstance(args, Parameters), "args is type: {}. expecting args to be type delphi.utils.helpers.Parameters"
        args = check_and_fill_args(args, TRUNC_MULTI_NORM_DEFAULTS)
        
        logger = delphiLogger() if args.verbose else delphiLogger(level=logging.CRITICAL)
        super().__init__(args, phi, alpha, dims, partial(ExponentialFamilyMultivariateNormalKnownCovariance, covariance_matrix), calc_multi_norm_suff_stat_known_cov, logger)
        self.covariance_matrix = covariance_matrix
        
    def _reparameterize_nat_form(self, 
                                 theta):
        T = self.covariance_matrix.inverse()
        v = theta@T

        return v.flatten()

    def _reparameterize_canon_form(self, 
                        theta): 
        loc = theta @ self.covariance_matrix

        return loc.flatten()
    
    @property 
    def best_loc_(self): 
        """
        Returns the best mean vector estimate for the multivariate normal distribution based off of the loss function.
        """
        return self.best_params
    
    @property
    def final_loc_(self): 
        """
        Returns the final mean vector estimate for the multivariate normal distribution based off of the loss function.
        """
        return self.final_params

    @property
    def ema_loc_(self): 
        """
        Returns the ema mean vector estimate for the multivariate normal distribution based off of the loss function.
        """
        return self.ema_params

    @property
    def avg_loc_(self): 
        """
        Returns the avg mean vector estimate for the multivariate normal distribution based off of the loss function.
        """
        return self.avg_params

    def calc_suff_stat(self, S): 
        return calc_multi_norm_suff_stat(S)
    
    def calculate_loss(self, S): 
        return self.criterion(self.theta, self.calc_suff_stat(S), *self.criterion_params)
    
    def __str__(self): 
        return "truncated multivariate normal distribution known covariance"
    
class TruncatedMultivariateNormalUnknownCovariance(TruncatedExponentialFamilyDistribution):
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
        assert isinstance(args, Parameters), "args is type: {}. expecting args to be type delphi.utils.helpers.Parameters"
        args = check_and_fill_args(args, TRUNC_MULTI_NORM_DEFAULTS)
        self.eigenvalue_lower_bound = args.eigenvalue_lower_bound
        
        logger = delphiLogger() if args.verbose else delphiLogger(level=logging.CRITICAL)
        super().__init__(args, phi, alpha, dims, ExponentialFamilyMultivariateNormal, calc_multi_norm_suff_stat, logger) 

    def _calc_emp_model(self): 
        S = self.train_loader_.dataset.S
        self.emp_canon_params = self.calc_suff_stat(S).mean(0)
        self.emp_theta = self._reparameterize_nat_form(self.emp_canon_params)
        self.emp_T = self.emp_theta[:self.dims**2].view(self.dims, self.dims)
        self.emp_v = self.emp_theta[self.dims**2:]
        self.register_parameter('T', nn.Parameter(self.emp_T))
        self.register_parameter('v', nn.Parameter(self.emp_v))
    
    def project_to_neg_definite(self, M, eps=1e-6):
        """Projects a symmetric matrix M onto the Positive Semi-Definite (PSD) cone."""
        L, Q = ch.linalg.eigh(M)
        L_clipped = ch.clamp(L, max=-eps)
        return Q @ ch.diag_embed(L_clipped) @ Q.T

    def step_post_hook(self, 
                       optimizer, 
                       args, 
                       kwargs) -> None:
        with ch.no_grad():
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
            T = self.project_to_neg_definite(T, eps=self.eigenvalue_lower_bound)
            # Symmetrize again after PSD projection
            T = 0.5 * (T + T.T)
            
            self.T.copy_(T)
            self.v.copy_(v)
            
    def parameters_(self):
        if self.args.covariance_matrix_lr is not None: 
            return [
                {'params': self.T, 'lr': self.args.covariance_matrix_lr},   
                {'params': self.v, 'lr': self.args.lr},
            ]
        return self.parameters()
    
    def _reparameterize_nat_form(self, 
                                 theta):
        cov_matrix = theta[:self.dims**2].resize(self.dims, self.dims)
        loc = theta[self.dims**2:]

        T = cov_matrix.inverse()
        v = loc@T

        return ch.cat([-.5*T.flatten(), v.flatten()])

    def _reparameterize_canon_form(self, 
                        theta): 
        T = theta[:self.dims**2].resize(self.dims, self.dims)
        v = theta[self.dims**2:] 

        covariance_matrix = (-2*T).inverse()
        loc = v @ covariance_matrix

        return ch.cat([covariance_matrix.flatten(), loc.flatten()])
    
    @property
    def theta(self): 
        return ch.cat([self.T.flatten(), self.v])
    
    @property 
    def best_loc_(self): 
        """
        Returns the best mean vector estimate for the multivariate normal distribution based off of the loss function.
        """
        return self.best_params[self.dims**2:]
    
    @property
    def best_covariance_matrix_(self): 
        """
        Returns the best covariance matrix estimate for the multivariate normal distribution based off of the loss function.
        """
        return self.best_params[:self.dims**2].view(self.dims, self.dims)
    
    @property
    def final_loc_(self): 
        """
        Returns the final mean vector estimate for the multivariate normal distribution based off of the loss function.
        """
        return self.final_params[self.dims**2:]

    @property
    def final_covariance_matrix_(self): 
        """
        Returns the final covariance matrix estimate for the multivariate normal distribution based off of the loss function.
        """
        self.final_params[:self.dims**2].view(self.dims, self.dims)
    
    @property
    def ema_loc_(self): 
        """
        Returns the ema mean vector estimate for the multivariate normal distribution based off of the loss function.
        """
        return self.ema_params[self.dims**2:]

    @property
    def ema_covariance_matrix_(self): 
        """
        Returns the ema covariance matrix estimate for the multivariate normal distribution based off of the loss function.
        """
        return self.ema_params[:self.dims**2].view(self.dims, self.dims)

    @property
    def avg_loc_(self): 
        """
        Returns the avg mean vector estimate for the multivariate normal distribution based off of the loss function.
        """
        return self.avg_params[self.dims**2:]

    @property
    def avg_covariance_matrix_(self): 
        """
        Returns the avg covariance matrix estimate for the multivariate normal distribution based off of the loss function.
        """
        return self.avg_params[:self.dims**2].view(self.dims, self.dims)

    def calc_suff_stat(self, S): 
        return calc_multi_norm_suff_stat(S)
    
    def calculate_loss(self, S): 
        return self.criterion(self.theta, self.calc_suff_stat(S), *self.criterion_params)
    
    def __str__(self): 
        return "truncated multivariate normal distribution"

"""
Truncated multivariate normal distribution class with known truncation set.
"""
def TruncatedMultivariateNormal(
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
    if covariance_matrix is not None: 
        return TruncatedMultivariateNormalKnownCovariance(args, phi, alpha, dims, covariance_matrix, sampler)
    else: 
        return TruncatedMultivariateNormalUnknownCovariance(args, phi, alpha, dims, sampler)
