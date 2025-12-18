
"""
Truncated multivariate normal distribution without oracle access (ie. unknown truncation set)
"""
import torch as ch
from torch import Tensor
from typing import Optional

from .truncated_multivariate_normal import TruncatedMultivariateNormal 
from ..oracle import UnknownGaussian
from ..trainer import Trainer
from ..grad import UnknownTruncationMultivariateNormalNLL 
from ..utils.datasets import UnknownTruncationNormalDataset, make_train_and_val_distr
from ..utils.helpers import Parameters, PSDError
from ..utils.defaults import check_and_fill_args, UNKNOWN_TRUNC_MULTI_NORM_DEFAULTS


class UnknownTruncationMultivariateNormal:
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
        super().__init__(args, lambda x: True, alpha, dims, covariance_matrix=covariance_matrix)
        # algorithm hyperparameters
        self.args = check_and_fill_args(args, UNKNOWN_TRUNC_MULTI_NORM_DEFAULTS)
        self.k = k
        # initialiaze pseudo oracle for gaussians with unknown truncation 
        self.emp_loc, self.emp_covariance_matrix = None, None
        del self.criterion
        self.criterion = UnknownTruncationMultivariateNormalNLL.apply
        
    def fit(self, S: Tensor):
        
        assert isinstance(S, Tensor), f"S is type: {type(S)}. expected type torch.Tensor."
        assert S.size(0) > S.size(1), f"input expected to be num samples by dimensions, current input is size {S.size()}." 
        self.train_loader_, self.val_loader_ = make_train_and_val_distr(self.args, S, UnknownTruncationNormalDataset)
        self._calc_emp_model()
        self.phi = UnknownGaussian(self.emp_loc, self.emp_covariance_matrix, S, self.k)
        self.exp_h = Exp_h(self.emp_loc, self.emp_covariance_matrix)
        self.criterion_params = [self.phi, self.exp_h, self.dims]
        while True:
            try:
                self.trainer = Trainer(self, self.args, self.logger) 
                self.trainer.train_model(self.train_loader_, self.val_loader_)
                return self
            except PSDError as psd:
                print(psd.message) 
                continue
            except Exception as e: 
                    raise e
            
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
        loc_term = (x - self.emp_loc)@u
        return ch.exp((cov_term - trace_term - loc_term + self.pi_const).double())
