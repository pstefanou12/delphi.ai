
"""
Truncated multivariate normal distribution without oracle access (ie. unknown truncation set)
"""
import torch as ch
from torch import Tensor
import torch.nn as nn
from typing import Optional
from functools import partial

from .truncated_multivariate_normal import TruncatedMultivariateNormalUnknownCovariance, TruncatedMultivariateNormalKnownCovariance
from ..oracle import UnknownGaussian
from ..trainer import Trainer
from ..grad import UnknownTruncationMultivariateNormalNLL 
from ..utils.datasets import UnknownTruncationNormalDataset, make_train_and_val_distr
from ..utils.helpers import Parameters, cov 
from ..utils.defaults import check_and_fill_args, UNKNOWN_TRUNC_MULTI_NORM_DEFAULTS


class UnknownTruncationMultivariateNormalKnownCovariance(TruncatedMultivariateNormalKnownCovariance):
    """
    Truncated multivariate normal distribution class.
    """
    def __init__(self,
                 args: Parameters,
                 k: int, 
                 alpha: float,
                 dims: int, 
                 covariance_matrix: Optional[ch.Tensor]):
        # instance variables 
        assert isinstance(args, Parameters), "args is type {}. expecting type delphi.utils.helper.Parameters.".format(type(args))
        # algorithm hyperparameters
        self.k = k
        self.args = check_and_fill_args(args, UNKNOWN_TRUNC_MULTI_NORM_DEFAULTS)
        super().__init__(args, partial(UnknownGaussian, k), alpha, dims, covariance_matrix=covariance_matrix)

        self.emp_loc, self.emp_covariance_matrix = None, None
        self.criterion = UnknownTruncationMultivariateNormalNLL.apply
        
    def fit(self, 
            S: Tensor): 
        
        assert isinstance(S, Tensor), "S is type: {}. expected type torch.Tensor.".format(type(S))
        assert S.size(0) > S.size(1), "input expected to be shape num samples by dimenions, current input is size {}.".format(S.size()) 
        self.train_loader_, self.val_loader_ = make_train_and_val_distr(self.args, S, UnknownTruncationNormalDataset)

        # verify that the S is whitened to N(0, I)
        emp_loc = S.mean(0)
        emp_cov = cov(S)
        if ch.norm(emp_loc - ch.zeros(S.size(1))) >= 1e-3 or ch.norm(emp_cov - ch.eye(S.size(1))) >= 1e-3:
            raise Exception(f"input dataset must be whitened (eg. N(O, I)). \n dataset mean: {emp_loc}, covariance matrix: {emp_cov}")
        self.phi = self.phi(S)
        # self.phi = UnknownGaussian(self.emp_loc, self.emp_covariance_matrix, S, self.k)
        self.exp_h = Exp_h(emp_loc, emp_cov)
        self.criterion_params = [self.phi, self.exp_h, self.dims]

        # Initialize tracking
        self.prev_best_loss = None
        self.radius_history = []
        self.loss_history = []
        # Initialize radius and parameters
        self._calc_emp_model()
        self.radius = self.args.min_radius

        phase = 0
        while phase < self.args.max_phases: 
            phase += 1
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"phase {phase}: training with radius={self.radius:.4f}")
            self.logger.info(f"\n{'='*60}")

            self.trainer = Trainer(
                self,
                self.args, 
                self.logger
            )
            self.trainer.train_model(self.train_loader_, self.val_loader_)

            # Update tracking
            current_loss = self.trainer.best_loss
            self.radius_history.append(self.radius)
            self.loss_history.append(current_loss)

            self.best_params, self.best_loss = self._reparameterize_canon_form(self.trainer.best_params), self.trainer.best_loss
            self.prev_theta, self.prev_loss = self.best_params.clone(), self.best_loss
            self.final_params, self.final_loss = self._reparameterize_canon_form(self.trainer.final_params), self.trainer.final_loss 
            self.ema_params = self._reparameterize_canon_form(self.trainer.ema_params)
            self.avg_params = self._reparameterize_canon_form(self.trainer.avg_params)

            should_stop, reason = self._check_convergence()
            
            if should_stop:
                self.logger.info(f"\n{'='*60}")
                self.logger.info(f"procedure converged: {reason}")
                self.logger.info(f"final radius: {self.radius:.4f}, final loss: {current_loss:.6f}")
                self.logger.info(f"total phases: {phase}")
                self.logger.info(f"\n{'='*60}")
                break

            # Expand radius for next phase
            self.prev_best_loss = current_loss
            old_radius = self.radius
            self.radius = min(self.radius * self.args.rate, self.args.max_radius)
            
            self.logger.info(
                f"expanding radius: {old_radius:.4f} -> {self.radius:.4f}, "
                f"loss improved by: {self.prev_best_loss - current_loss:.6e}"
            )

        return self
            
    def forward(self, x): 
        return ch.cat([self.covariance_matrix.inverse().flatten(), self.theta])

    
class UnknownTruncationMultivariateNormalUnknownCovariance(TruncatedMultivariateNormalUnknownCovariance):
    """
    Truncated multivariate normal distribution class.
    """
    def __init__(self,
                 args: Parameters,
                 k: int, 
                 alpha: float,
                 dims: int):
        # instance variables 
        assert isinstance(args, Parameters), "args is type {}. expecting type delphi.utils.helper.Parameters.".format(type(args))
        # algorithm hyperparameters
        self.k = k
        self.args = check_and_fill_args(args, UNKNOWN_TRUNC_MULTI_NORM_DEFAULTS)
        super().__init__(args, partial(UnknownGaussian, k), alpha, dims)

        self.emp_loc, self.emp_covariance_matrix = None, None
        self.criterion = UnknownTruncationMultivariateNormalNLL.apply
        
    def fit(self, 
            S: Tensor): 
        
        assert isinstance(S, Tensor), "S is type: {}. expected type torch.Tensor.".format(type(S))
        assert S.size(0) > S.size(1), "input expected to be shape num samples by dimenions, current input is size {}.".format(S.size()) 
        self.train_loader_, self.val_loader_ = make_train_and_val_distr(self.args, S, UnknownTruncationNormalDataset)

        # verify that the S is whitened to N(0, I)
        emp_loc = S.mean(0)
        emp_cov = cov(S)
        if ch.norm(emp_loc - ch.zeros(S.size(1))) >= 1e-3 or ch.norm(emp_cov - ch.eye(S.size(1))) >= 1e-3:
            raise Exception(f"input dataset must be whitened (eg. N(O, I)). \n dataset mean: {emp_loc}, covariance matrix: {emp_cov}")
        self.phi = self.phi(S)
        # self.phi = UnknownGaussian(self.emp_loc, self.emp_covariance_matrix, S, self.k)
        self.exp_h = Exp_h(emp_loc, emp_cov)
        self.criterion_params = [self.phi, self.exp_h, self.dims]

        # Initialize tracking
        self.prev_best_loss = None
        self.radius_history = []
        self.loss_history = []
        # Initialize radius and parameters
        self._calc_emp_model()
        self.radius = self.args.min_radius

        phase = 0
        while phase < self.args.max_phases: 
            phase += 1
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"phase {phase}: training with radius={self.radius:.4f}")
            self.logger.info(f"\n{'='*60}")

            self.trainer = Trainer(
                self,
                self.args, 
                self.logger
            )
            self.trainer.train_model(self.train_loader_, self.val_loader_)

            # Update tracking
            current_loss = self.trainer.best_loss
            self.radius_history.append(self.radius)
            self.loss_history.append(current_loss)

            self.best_params, self.best_loss = self._reparameterize_canon_form(self.trainer.best_params), self.trainer.best_loss
            self.prev_theta, self.prev_loss = self.best_params.clone(), self.best_loss
            self.final_params, self.final_loss = self._reparameterize_canon_form(self.trainer.final_params), self.trainer.final_loss 
            self.ema_params = self._reparameterize_canon_form(self.trainer.ema_params)
            self.avg_params = self._reparameterize_canon_form(self.trainer.avg_params)

            should_stop, reason = self._check_convergence()
            
            if should_stop:
                self.logger.info(f"\n{'='*60}")
                self.logger.info(f"procedure converged: {reason}")
                self.logger.info(f"final radius: {self.radius:.4f}, final loss: {current_loss:.6f}")
                self.logger.info(f"total phases: {phase}")
                self.logger.info(f"\n{'='*60}")
                break

            # Expand radius for next phase
            self.prev_best_loss = current_loss
            old_radius = self.radius
            self.radius = min(self.radius * self.args.rate, self.args.max_radius)
            
            self.logger.info(
                f"expanding radius: {old_radius:.4f} -> {self.radius:.4f}, "
                f"loss improved by: {self.prev_best_loss - current_loss:.6e}"
            )

        return self
    
    def _calc_emp_model(self):
        S = self.train_loader_.dataset.S 

        self.emp_T = cov(S).inverse()
        self.emp_v = self.emp_T@S.mean(0)
        self.register_parameter('T', nn.Parameter(self.emp_T.clone()))
        self.register_parameter('v', nn.Parameter(self.emp_v.clone()))

    def forward(self, x): 
        return ch.cat([self.T.flatten(), self.v])
    
    def project_to_pos_definite(self, M, eps=1e-6):
        """Projects a symmetric matrix M onto the Positive Semi-Definite (PSD) cone."""
        L, Q = ch.linalg.eigh(M)
        L_clipped = ch.clamp(L, min=eps)
        return Q @ ch.diag_embed(L_clipped) @ Q.T

    def step_post_hook(self, 
                       optimizer, 
                       args, 
                       kwargs):
        if self.T > 10: import ipdb; ipdb.set_trace()
        print(f'pre T: {self.T}')
        print(f'pre v: {self.v}')
        with ch.no_grad():
            T = self.T.clone().view(self.dims, self.dims)
            v = self.v.clone()

            # --- 1) Project mean into L2 ball ---
            loc_diff = self.emp_v - v
            dist = ch.norm(loc_diff)

            if dist > self.radius:
                v = self.emp_v + loc_diff / dist * self.radius

            # --- 2) Frobenius ball projection for T ---
            cov_diff = T - self.emp_T
            frob_norm = ch.linalg.norm(cov_diff, ord='fro')

            if frob_norm > self.radius:
                T = self.emp_T + cov_diff * (self.radius / frob_norm)

            # Symmetrize after projection (important!)
            T = 0.5 * (T + T.T)
            # --- 3) Final PSD projection ---
            T = self.project_to_pos_definite(T, eps=self.eigenvalue_lower_bound)
            # Symmetrize again after PSD projection
            T = 0.5 * (T + T.T)
            
            self.T.copy_(T)
            self.v.copy_(v)
        print(f'post T: {self.T}')
        print(f'post v: {self.v}')

    def _reparameterize_canon_form(self, 
                        theta): 
        T = theta[:self.dims**2].resize(self.dims, self.dims)
        v = theta[self.dims**2:] 

        covariance_matrix = T.inverse()
        loc = v @ covariance_matrix

        return ch.cat([covariance_matrix.flatten(), loc.flatten()])
            
def UnknownTruncationMultivariateNormal(
        args: Parameters, 
        k: int,
        alpha: float, 
        dims: int, 
        covariance_matrix: Optional[ch.Tensor] = None, 
): 
    assert isinstance(args, Parameters), "args is type: {}. expecting args to be type delphi.utils.helpers.Parameters"
    args = check_and_fill_args(args, UNKNOWN_TRUNC_MULTI_NORM_DEFAULTS) 

    if covariance_matrix is not None: 
        return UnknownTruncationMultivariateNormalKnownCovariance(args, k, alpha, dims, covariance_matrix=covariance_matrix)
    return UnknownTruncationMultivariateNormal(args, k, alpha, dims)
    
# HELPER FUNCTIONS
class Exp_h:
    def __init__(self, emp_loc, emp_cov):
        self.emp_loc = emp_loc
        self.emp_cov = emp_cov
        self.pi_const = (self.emp_loc.size(0) / 2.0) * ch.log(ch.Tensor([2.0 * ch.pi])).unsqueeze(0)

    def __call__(self, u, B, x):
        """returns: evaluates exponential function"""
        quad_term = 0.5 * ch.sum(x @ B * x, dim=1, keepdim=True)
        trace_term = ch.trace((B - ch.eye(u.size(0))) @ (self.emp_cov + self.emp_loc[...,None] @ self.emp_loc[None,...])).unsqueeze(0)/2.0
        lin_term = ((x - self.emp_loc) @ u)[..., None]
        h = quad_term - trace_term - lin_term + self.pi_const
        return ch.exp(h).double()