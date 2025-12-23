
"""
Truncated multivariate normal distribution without oracle access (ie. unknown truncation set)
"""
import torch as ch
from torch import Tensor
from typing import Optional
from functools import partial

from .truncated_multivariate_normal import TruncatedMultivariateNormal, TruncatedMultivariateNormalKnownCovariance
from ..oracle import UnknownGaussian
from ..trainer import Trainer
from ..grad import UnknownTruncationMultivariateNormalNLL 
from ..utils.datasets import UnknownTruncationNormalDataset, make_train_and_val_distr
from ..utils.helpers import Parameters, PSDError
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
                 covariance_matrix: Optional[ch.Tensor] = None):
        # instance variables 
        assert isinstance(args, Parameters), "args is type {}. expecting type delphi.utils.helper.Parameters.".format(type(args))
        # algorithm hyperparameters
        self.k = k
        self.args = check_and_fill_args(args, UNKNOWN_TRUNC_MULTI_NORM_DEFAULTS)
        super().__init__(args, partial(UnknownGaussian, k), alpha, dims, covariance_matrix=covariance_matrix)

        # initialiaze pseudo oracle for gaussians with unknown truncation 
        self.emp_loc, self.emp_covariance_matrix = None, None
        self.criterion = UnknownTruncationMultivariateNormalNLL.apply
        
    # def fit(self, S: Tensor):
        
    #     assert isinstance(S, Tensor), f"S is type: {type(S)}. expected type torch.Tensor."
    #     assert S.size(0) > S.size(1), f"input expected to be num samples by dimensions, current input is size {S.size()}." 
    #     self.train_loader_, self.val_loader_ = make_train_and_val_distr(self.args, S, UnknownTruncationNormalDataset)
    #     self._calc_emp_model()
    #     self.phi = self.phi(S)
    #     # self.phi = UnknownGaussian(self.emp_loc, self.emp_covariance_matrix, S, self.k)
    #     self.exp_h = Exp_h(self._reparameterize_canon_form(self.theta), self.covariance_matrix)
    #     self.criterion_params = [self.phi, self.exp_h, self.dims]
    #     while True:
    #         try:
    #             self.trainer = Trainer(self, self.args, self.logger) 
    #             self.trainer.train_model(self.train_loader_, self.val_loader_)
    #             return self
    #         except PSDError as psd:
    #             print(psd.message) 
    #             continue
    #         except Exception as e: 
    #                 raise e
            
    def fit(self, 
            S: Tensor): 
        
        assert isinstance(S, Tensor), "S is type: {}. expected type torch.Tensor.".format(type(S))
        assert S.size(0) > S.size(1), "input expected to be shape num samples by dimenions, current input is size {}.".format(S.size()) 
        # assert self.args.batch_size <= self.args.num_samples, "batch size must be smaller than or equal to the number of samples being sampled"
        
        self.train_loader_, self.val_loader_ = make_train_and_val_distr(self.args, S, UnknownTruncationNormalDataset)
        self._calc_emp_model()
        self.phi = self.phi(S)
        # self.phi = UnknownGaussian(self.emp_loc, self.emp_covariance_matrix, S, self.k)
        self.exp_h = Exp_h(self._reparameterize_canon_form(self.theta), self.covariance_matrix)
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
        return ch.cat([self.covariance_matrix.flatten(), self.theta])
            
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
        return TruncatedMultivariateNormalKnownCovariance(args, k, alpha, dims, covariance_matrix=covariance_matrix)
    
# HELPER FUNCTIONS
class Exp_h:
    def __init__(self, emp_loc, emp_cov):
        self.emp_loc = emp_loc
        self.emp_cov = emp_cov
        self.pi_const = (self.emp_loc.size(0) / 2.0) * ch.log(ch.Tensor([2.0 * ch.pi])).unsqueeze(0)

    def __call__(self, u, B, x):
        """returns: evaluates exponential function"""
        cov_term = ch.sum(x @ B * x, dim=1)[...,None] / 2.0
        trace_term = ch.trace((B - ch.eye(u.size(0))) @ (self.emp_cov + self.emp_loc[...,None]@self.emp_loc[None,...])).unsqueeze(0) / 2.0
        loc_term = ((x - self.emp_loc)@u)[...,None]
        return ch.exp((cov_term - trace_term - loc_term + self.pi_const).double())
