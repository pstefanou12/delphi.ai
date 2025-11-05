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
from ..grad import TruncatedMultivariateNormalNLL, TruncatedMultivariateNormalScore, PreSampler
from ..trainer import Trainer
from ..utils.helpers import PSDError, Parameters, is_psd
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
        # instance variables
        assert isinstance(args, Parameters), "args is type: {}. expecting args to be type delphi.utils.helpers.Parameters"
        args = check_and_fill_args(args, TRUNC_MULTI_NORM_DEFAULTS)
        super().__init__(args)
 
        self.phi = phi
        self.alpha = alpha
        self.dims = dims
        self.covariance_matrix = covariance_matrix
        self.sampler = PreSampler(self.dims, self.args.num_samples) if sampler is None else sampler
        self.trunc_multi_norm_score = TruncatedMultivariateNormalScore(self.covariance_matrix is not None)
        
        del self.criterion
        self.criterion = TruncatedMultivariateNormalNLL.apply

        self.emp_loc, self.emp_covariance_matrix = None, None
        self.S = None

        self.best_covariance_matrix, self.best_loc, self.best_loss = None, None, None
        self.final_covariance_matrix, self.final_loc, self.final_loss = None, None, None

        if self.args.verbose:
            print(f'args: {self.args}')

    def fit(self, S: Tensor):
        """
        """
        assert isinstance(S, Tensor), "S is type: {}. expected type torch.Tensor.".format(type(S))
        assert S.size(0) > S.size(1), "input expected to be shape num samples by dimenions, current input is size {}.".format(S.size()) 
        assert self.args.batch_size <= self.args.num_samples, "batch size must be smaller than or equal to the number of samples being sampled"
        
        self.S = S
        M = MultivariateNormal(ch.zeros(self.dims), ch.eye(self.dims))
        self.samples = M.sample([10000])
        self.criterion_params = [self.phi, self.dims, self.trunc_multi_norm_score, self.sampler, self.args.num_samples, self.args.eps]

        try: 
            self.train_loader_, self.val_loader_ = make_train_and_val_distr(self.args, 
                                                                            self.S, 
                                                                            TruncatedNormalDataset,
                                                                            {'trunc_multi_norm_score': self.trunc_multi_norm_score})

            # run PGD to predict actual estimates
            self.trainer = Trainer(
                self,
                self.args
            )
            self.trainer.train_model(self.train_loader_, self.val_loader_)
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
            T = self.covariance_matrix.clone().inverse()
        else:
            T = self.emp_covariance_matrix.clone().inverse()
        v = self.emp_loc.clone() @ T

        self.emp_v = v
        self.emp_T = T

        # Initialize empirical model 
        self.register_parameter('T', nn.Parameter(T))
        self.register_parameter('v', nn.Parameter(v))

    def __call__(self, batch, targ):
        """
        Training step for defined model.
        Args: 
            i (int) : gradient step or epoch number
            batch (Iterable) : iterable of inputs that 
        """
        return self.T, self.v
    
    def project_to_psd(self, M, eps=1e-6):
        """Projects a symmetric matrix M onto the Positive Semi-Definite (PSD) cone."""
        L, Q = ch.linalg.eigh(M)
        L_clipped = ch.clamp(L, min=eps)
        return Q @ ch.diag_embed(L_clipped) @ Q.T

    def iteration_hook(self, i, is_train, loss, batch) -> None:
        with ch.no_grad():
            # Clone current parameters
            T_prev = self.T.clone()
            v_prev = self.v.clone()

            T = T_prev.clone()
            v = v_prev.clone()

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
            T = self.project_to_psd(T, eps=1e-8)

            # Symmetrize again after PSD projection
            T = 0.5 * (T + T.T)

            # --- 4) Damping / stabilization ---
            # Prevent projection oscillation and shrink toward feasible region
            damping = 0.05
            T = (1 - damping) * T + damping * self.emp_T
            v = (1 - damping) * v + damping * self.emp_v

            # --- 5) Condition check before assigning ---
            cond = ch.linalg.cond(T)
            if cond > 1e12:
                print(f"[iter {i}] Warning: ill-conditioned T (cond={cond:.2e}) → re-PSD")
                T = self.project_to_psd(T, eps=1e-6)

            # --- 6) Assign safely ---
            self.T.copy_(T)
            self.v.copy_(v)

            # Debug: ensure PSD (no inverse!)
            if ch.any(ch.linalg.eigvalsh(T) < -1e-10):
                print("PSD violation — applying repair")
                self.T.copy_(self.project_to_psd(T, eps=1e-6))
                    
        if not is_psd(self.T.inverse()):
            raise PSDError("Covariance matrix is no longer positive-definite. Try reducing covariance_matrix_lr.")


    def post_training_hook(self): 
        self.args.r *= self.args.rate
        # reparameterize distribution

        self.best_covariance_matrix, self.best_loc, self.best_theta_loss = *self._reparameterize(self.trainer.best_params), self.trainer.best_loss
        self.final_covariance_matrix, self.final_loc, self.final_theta_loss = *self._reparameterize(self.trainer.final_params), self.trainer.final_loss


    def parameters_(self):
        if self.args.covariance_matrix_lr is not None: 
            return [
                {'params': self.T, 'lr': self.args.covariance_matrix_lr},   
                {'params': self.v, 'lr': self.args.lr},
            ]
        
        return self.parameters()

    def _reparameterize(self, theta): 
        T = theta[:,:self.dims**2].resize(self.dims, self.dims)
        v = theta[:,self.dims**2:] 

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

    def calculate_score(self, S): 
        return self.trunc_multi_norm_score(S)
    
    def calculate_loss(self, S): 
        data = ch.cat([S, self.calculate_score(S)], dim=1)
        return self.criterion(self.theta, data, *self.criterion_params)
