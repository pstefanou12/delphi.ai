
"""
Truncated multivariate normal distribution without oracle access (ie. unknown truncation set)
"""

from re import I
import torch as ch
from torch import Tensor
from torch.distributions.multivariate_normal import MultivariateNormal
import cox 

from .censored_multivariate_normal import CensoredMultivariateNormalModel
from ..oracle import UnknownGaussian
from .distributions import distributions
from ..trainer import Trainer
from ..grad import TruncatedMultivariateNormalNLL
from ..utils.datasets import TruncatedNormalDataset, make_train_and_val_distr
from ..utils.helpers import Parameters, PSDError
from ..utils.defaults import check_and_fill_args, TRAINER_DEFAULTS, DELPHI_DEFAULTS, TRUNC_MULTI_NORM_DEFAULTS


class TruncatedMultivariateNormal(distributions):
    """
    Truncated multivariate normal distribution class.
    """
    def __init__(self,
            args: Parameters, 
            store: cox.store.Store=None):
        super(TruncatedMultivariateNormal).__init__()
        # instance variables 
        assert isinstance(args, Parameters), "args is type {}. expecting type delphi.utils.helper.Parameters.".format(type(args))
        assert store is None or isinstance(store, cox.store.Store), "store is type: {}. expecting cox.store.Store.".format(type(store))
        self.store = store 
        self.truncated = None
        # algorithm hyperparameters
        TRUNC_MULTI_NORM_DEFAULTS.update(DELPHI_DEFAULTS)
        TRUNC_MULTI_NORM_DEFAULTS.update(TRAINER_DEFAULTS)
        self.args = check_and_fill_args(args, TRUNC_MULTI_NORM_DEFAULTS)

    def fit(self, S: Tensor):
        
        assert isinstance(S, Tensor), "S is type: {}. expected type torch.Tensor.".format(type(S))
        assert S.size(0) > S.size(1), "input expected to be num samples by dimensions, current input is size {}.".format(S.size()) 
        while True:
            try:
                self.train_loader_, self.val_loader_ = make_train_and_val_distr(self.args, S, TruncatedNormalDataset)
                self.truncated = TruncatedMultivariateNormalModel(self.args, self.train_loader_.dataset)
                self.trainer = Trainer(self.truncated, self.args, store=self.store) 
        
                # run PGD for parameter estimation 
                self.trainer.train_model((self.train_loader_, self.val_loader_))
                return self
            except PSDError as psd:
                print(psd.message) 
                continue
            except Exception as e: 
                    raise e

    @property 
    def loc_(self): 
        """
        Returns the mean of the normal disribution.
        """
        return self.truncated.model.loc.clone()
  
    @property 
    def covariance_matrix_(self): 
        """
        Returns the standard deviation for the normal distribution.
        """
        return self.truncated.model.covariance_matrix.clone()


class TruncatedMultivariateNormalModel(CensoredMultivariateNormalModel):
    '''
    Model for truncated normal distributions to be passed into trainer.
    '''
    def __init__(self, args, train_ds):
        '''
        Args: 
            args (delphi.utils.Parameters) : parameter object holding hyperparameters
        '''
        super().__init__(args, train_ds)
        # initialiaze pseudo oracle for gaussians with unknown truncation 
        self.emp_loc, self.emp_covariance_matrix = None, None
        # initialize empirical estimates
        self.calc_emp_model()
        self.args.__setattr__('phi', UnknownGaussian(self.emp_loc, self.emp_covariance_matrix, self.train_ds.S, self.args.d))

        # exponent class
        self.exp_h = Exp_h(self.emp_loc, self.emp_covariance_matrix)

    def __call__(self, batch):
        '''
        Training step for defined model.
        Args: 
            batch (Iterable) : iterable of inputs
        '''
        loss = TruncatedMultivariateNormalNLL.apply(self.model.loc, self.model.covariance_matrix, *batch, self.args.phi, self.exp_h)
        return loss, None, None

    def calc_emp_model(self): 
        # initialize projection set
        self.emp_covariance_matrix = self.train_ds.covariance_matrix
        self.emp_loc = self.train_ds.loc
        self.model = MultivariateNormal(self.emp_loc, self.emp_covariance_matrix)

    def post_training_hook(self):
        if self.model.loc.isnan() or self.model.covariance_matrix.isnan(): 
            import pdb; pdb.set_trace()
        # reparamterize distribution
        self.model.covariance_matrix.requires_grad, self.model.loc.requires_grad = False, False
        self.model.covariance_matrix.data = self.model.covariance_matrix.inverse()
        self.model.loc.data = (self.model.loc[None,...]  @ self.model.covariance_matrix).flatten()
        # set estimated distribution in membership oracle
        self.args.phi.dist = self.model


# HELPER FUNCTIONS
class Exp_h:
    def __init__(self, emp_loc, emp_cov):
        self.emp_loc = emp_loc
        self.emp_cov = emp_cov
        self.pi_const = (self.emp_loc.size(0) / 2.0) * ch.log(2.0 * Tensor([ch.acos(ch.zeros(1)).item() * 2]).unsqueeze(0))

    def __call__(self, u, B, x):
        """
        returns: evaluates exponential function
        """
        cov_term = ch.bmm(x.unsqueeze(1)@B, x.unsqueeze(2)).squeeze(1) / 2.0
        # trace_term = ch.trace((B - self.emp_cov) * (self.emp_cov + self.emp_loc[...,None]@self.emp_loc[None,...])).unsqueeze(0) / 2.0
        trace_term = ch.trace((B - ch.eye(u.size(0))) * (self.emp_cov + self.emp_loc[...,None]@self.emp_loc[None,...])).unsqueeze(0) / 2.0
        loc_term = (x - self.emp_loc)@u.unsqueeze(1)
        return ch.exp((cov_term - trace_term - loc_term + self.pi_const).double())
