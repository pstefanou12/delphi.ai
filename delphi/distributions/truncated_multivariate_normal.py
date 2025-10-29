"""
Truncated multivariate normal distribution with oracle access (ie. known truncation set).
"""

import torch as ch
from torch import Tensor
from torch.distributions.multivariate_normal import MultivariateNormal
import math
import torch.nn as nn
from typing import Callable, Optional

from .distributions import distributions
from ..utils.datasets import TruncatedNormalDataset, make_train_and_val_distr
from ..grad import TruncatedMultivariateNormalNLL, TruncatedMultivariateNormalScore
from ..trainer import Trainer
from ..utils.helpers import PSDError, Parameters 
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
                covariance_matrix: Optional[ch.Tensor] = None):
        """
        """
        # instance variables
        assert isinstance(args, Parameters), "args is type: {}. expecting args to be type delphi.utils.helpers.Parameters"
        args = check_and_fill_args(args, TRUNC_MULTI_NORM_DEFAULTS)
        super().__init__(args)
 
        self.phi = phi
        self.alpha = alpha
        self.dims = dims
        self.covariance_matrix = covariance_matrix
        self.trunc_multi_norm_score = TruncatedMultivariateNormalScore(self.covariance_matrix is not None)
        
        del self.criterion
        self.criterion = TruncatedMultivariateNormalNLL.apply

        self.emp_loc, self.emp_covariance_matrix = None, None
        self.S = None

        if self.args.verbose:
            print(f'args: {self.args}')

    def fit(self, S: Tensor):
        """
        """
        assert isinstance(S, Tensor), "S is type: {}. expected type torch.Tensor.".format(type(S))
        assert S.size(0) > S.size(1), "input expected to be shape num samples by dimenions, current input is size {}.".format(S.size()) 

        self.S = S
        M = MultivariateNormal(ch.zeros(self.dims), ch.eye(self.dims))
        self.samples = M.sample([10000])
        self.criterion_params = [self.phi, self.dims, self.trunc_multi_norm_score, self.args.num_samples, self.args.eps]

        try: 
            self.train_loader_, self.val_loader_ = make_train_and_val_distr(self.args, 
                                                                            self.S, 
                                                                            TruncatedNormalDataset,
                                                                            {'trunc_multi_norm_score': self.trunc_multi_norm_score})

            # run PGD to predict actual estimates
            trainer = Trainer(
                self,
                self.args
            )
            trainer.train_model(self.train_loader_, self.val_loader_)
            return self
        except PSDError as psd:
            raise PSDError(psd, "covariance matrix became not positive semi-definite. try decreasing the the projection set radius and try again.")
        except Exception as e: 
            raise e
            
    def _calc_emp_model(self): 
        # initialize projection set
        if self.covariance_matrix is not None: 
            self.emp_covariance_matrix = self.covariance_matrix
        else:   
            self.emp_covariance_matrix = self.train_loader_.dataset.covariance_matrix
        self.emp_loc = self.train_loader_.dataset.loc
        self.model = MultivariateNormal(self.emp_loc, self.emp_covariance_matrix)

        self.dims = self.emp_loc.size(0)
            
    def pretrain_hook(self, train_loader):
        self._calc_emp_model()
        # parameterize projection set
        self.radius = self.args.r * (math.log(1.0 / self.alpha) / (self.alpha ** 2)) + 12
        if self.covariance_matrix is not None:
            self.T = self.covariance_matrix.clone().inverse()
        else:
            self.T = self.emp_covariance_matrix.clone().inverse()
        self.v = self.emp_loc.clone() @ self.T

        # Initialize empirical model 
        self.model = MultivariateNormal(self.v, self.T)
        # Register parameters with Pytorch
        theta = ch.cat([self.T.flatten(), self.v])
        self.register_parameter('theta', nn.Parameter(theta))
            
    def __call__(self, batch, targ):
        """
        Training step for defined model.
        Args: 
            i (int) : gradient step or epoch number
            batch (Iterable) : iterable of inputs that 
        """
        return self.theta
    
    def iteration_hook(self, i, is_train, loss, batch) -> None:
        # pass
        # Project location to ball around v
        theta = self.theta
        v = theta[self.dims**2:]
        T = theta[:self.dims**2].resize(self.dims, self.dims)
        loc_diff = v - self.v
        loc_norm = ch.norm(loc_diff)
        if loc_norm > self.radius:
            v = self.v + (loc_diff / loc_norm) * self.radius
            
        # Project covariance to PSD cone with bounded deviation from T
        cov = T.inverse()
        # Ensure PSD via eigenvalue clipping
        L, Q = ch.linalg.eigh(cov)
        L_clipped = ch.clamp(L, min=1e-6)  # Ensure positive eigenvalues
        cov_psd = Q @ ch.diag_embed(L_clipped) @ Q.T
    
        # Project to Frobenius ball around T if needed
        cov_diff = cov_psd - self.T
        frob_norm = ch.linalg.norm(cov_diff, ord='fro')
        if frob_norm > self.radius:
            cov_projected = self.T + (cov_diff / frob_norm) * self.radius
            # Re-ensure PSD after projection
            L, Q = ch.linalg.eigh(cov_projected)
            L_clipped = ch.clamp(L, min=1e-6)
            T = (Q @ ch.diag_embed(L_clipped) @ Q.T).inverse()
        else:
            T = cov_psd.inverse()

        theta = ch.cat([T.flatten(), v])
        self.theta.data = theta

    def post_training_hook(self): 
        self.args.r *= self.args.rate
        # reparamterize distribution
        # self.model.covariance_matrix.requires_grad, self.model.loc.requires_grad = False, False
        theta = self.theta

        T = theta[:self.dims**2].resize(self.dims, self.dims)
        v = theta[self.dims**2:]

        self.model.covariance_matrix.data = T.inverse()
        self.model.loc.data = v @ self.model.covariance_matrix

        self.theta.requires_grad = False
    
    @property 
    def loc_(self): 
        """
        Returns the mean of the normal disribution.
        """
        return self.model.loc.clone()
    
    @property
    def covariance_matrix_(self): 
        """
        Returns the covariance matrix of the distribution.
        """
        return self.model.covariance_matrix.clone()
    
    def calculate_score(self, S): 
        return self.trunc_multi_norm_score(S)
    
    def calculate_loss(self, S): 
        data = ch.cat([S, self.calculate_score(S)], dim=1)
        return self.criterion(self.theta, data, *self.criterion_params)
