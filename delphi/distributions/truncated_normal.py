"""
Truncated normal distribution without oracle access (ie. unknown truncation set)
"""

import torch as ch
from torch import Tensor
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data import DataLoader
from scipy.linalg import sqrtm
import copy

from .. import delphi
from .distributions import distributions
from ..oracle import oracle, UnknownGaussian
from ..trainer import Trainer
from ..utils.helpers import Bounds, check_and_fill_args, Parameters
from ..utils.datasets import TruncatedNormalDataset, make_train_and_val_distr
from ..grad import TruncatedMultivariateNormalNLL

# CONSTANTS 
DEFAULTS = {
        'epochs': (int, 1),
        'num_trials': (int, 3),
        'clamp': (bool, True), 
        'val': (float, .2),
        'lr': (float, 1e-1), 
        'step_lr': (int, 100),
        'step_lr_gamma': (float, .9), 
        'custom_lr_multiplier': (str, None), 
        'momentum': (float, 0.0), 
        'weight_decay': (float, 0.0), 
        'l1': (float, 0.0), 
        'eps': (float, 1e-5),
        'r': (float, 1.0), 
        'rate': (float, 1.5), 
        'batch_size': (int, 10),
        'tol': (float, 1e-1),
        'workers': (int, 0),
        'num_samples': (int, 10),
        'variance': (ch.Tensor, None), 
        'd': (int, 100),
}


class TruncatedNormal(distributions):
    """
    Truncated normal distribution class.
    """
    def __init__(self,
            alpha: float,
            kwargs: dict=None):
        super(TruncatedNormal).__init__()
        assert isinstance(alpha, float), "alpha is type: {}. expected type float.".format(type(alpha))
        # algorithm hyperparameters
        self.args = check_and_fill_args(Parameters({**{'alpha': Tensor([alpha])}, **kwargs}), DEFAULTS)

    def fit(self, S: Tensor):
        """
        Runs PSGD to minimize the negative population log likelihood of the truncated distribution.
        Args: 
            S (torch.Tensor): censored samples dataset, expecting n by d matrix (num_samples by dimensions)
        """
        assert isinstance(S, Tensor), "S is type: {}. expected type torch.Tensor.".format(type(S))
        assert S.size(1) == 1, "truncated normal class only accepts 1 dimensional distributions." 
        self.train_loader_, self.val_loader_ = make_train_and_val_distr(self.args, S, TruncatedNormalDataset)
        self.truncated = TruncatedNormalModel(self.args, self.train_loader_.dataset)
        # run PGD to predict actual estimates
        self.trainer = Trainer(self.truncated, self.args.epochs, self.args.num_trials, self.args.tol)

        # run PGD for parameter estimation 
        self.trainer.train_model((self.train_loader_, self.val_loader_))

    #    # rescale/standardize
    #    self.truncated.model.covariance_matrix.data = self.truncated.model.covariance_matrix @ self.emp_covariance_matrix
    #    self.truncated.model.loc.data = (self.truncated.model.loc[None,...] @ Tensor(sqrtm(self.emp_covariance_matrix.numpy()))).flatten() + self.emp_loc
    #
    @property 
    def loc(self): 
        """
        Returns the mean of the normal disribution.
        """
        return self.truncated.model.loc.clone()

    @property 
    def variance(self): 
        """
        Returns the standard deviation for the normal distribution.
        """
        return self.truncated.model.covariance_matrix.clone()

    def phi_(self, x: Tensor): 
        """
        After running procedure and learning truncation set, can call function 
        to see if samples fall within S.
        """
        return self.truncated.phi_(x)

        
class TruncatedNormalModel(delphi.delphi):
    '''
    Model for truncated normal distributions to be passed into trainer.
    '''
    def __init__(self, args, train_ds): 
        '''
        Args: 
            args (cox.utils.Parameters) : parameter object holding hyperparameters
        '''
        super().__init__(args) 
        self.train_ds = train_ds        
        # initialiaze pseudo oracle for gaussians with unknown truncation 
        self.phi = UnknownGaussian(self.train_ds.loc, self.train_ds.covariance_matrix, self.train_ds.S, self.args.d)

        # exponent class
        self.exp_h = Exp_h(self.train_ds.loc, self.train_ds.covariance_matrix)

    def pretrain_hook(self):
        # parameterize projection set
        if self.args.variance is not None:
            B = self.args.variance.clone()
        else:
            B = self.train_ds.covariance_matrix.inverse() 
        u = (self.train_ds.loc[None,...] @ B).flatten()
        # initialize projection set
        self.radius = self.args.r * ch.sqrt(ch.log(1.0 / self.args.alpha))
        if self.args.clamp:
            self.loc_bounds  = Bounds(self.train_ds.loc - self.radius, self.train_ds.loc + self.radius)             
            if self.args.variance is None: 
                self.scale_bounds = Bounds(self.args.alpha.pow(2) / 12,
                                                         B + self.radius)
        else:
            pass

        self.model = MultivariateNormal(u, B)
        self.model.loc.requires_grad, self.model.covariance_matrix.requires_grad = True, True
        # if distribution with known variance, remove from computation graph
        if self.args.variance is not None: self.model.covariance_matrix.requires_grad = False
        self.params = [self.model.loc, self.model.covariance_matrix]

    def __call__(self, batch):
        '''
        Training step for defined model.
        Args: 
            batch (Iterable) : iterable of inputs that 
        '''
        loss = TruncatedMultivariateNormalNLL.apply(self.model.loc, self.model.covariance_matrix, *batch, self.phi, self.exp_h)
        return loss, None, None

    def iteration_hook(self, i, loop_type, loss, prec1, prec5, batch):
        '''
        Iteration hook for defined model. Method is called after each 
        training update.
        Args:
            loop_type (str) : 'train' or 'val'; indicating type of loop
            loss (ch.Tensor) : loss for that iteration
            prec1 (float) : accuracy for top prediction
            prec5 (float) : accuracy for top-5 predictions
        '''
        if self.args.clamp:
            self.model.loc.data = ch.clamp(self.model.loc.data, float(self.loc_bounds.lower), float(self.loc_bounds.upper))
            if self.args.variance is None:
                self.model.covariance_matrix.data = ch.clamp(self.model.covariance_matrix.data, float(self.scale_bounds.lower), float(self.scale_bounds.upper))
        else:
            pass

    def post_training_hook(self):
        # reparamterize distribution
        self.model.covariance_matrix.requires_grad, self.model.loc.requires_grad = False, False
        self.model.covariance_matrix.data = self.model.covariance_matrix.inverse()
        self.model.loc.data = (self.model.loc[None,...]  @ self.model.covariance_matrix).flatten()
        # set estimated distribution in membership oracle
        self.phi.dist = self.model

    def phi_(self, x): 
        x_norm = (x - self.train_ds.loc) @ Tensor(sqrtm(self.train_ds.covariance_matrix.numpy())).inverse() 
        return self.phi(x_norm)


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
        trace_term = ch.trace((B - ch.eye(u.size(0))) * (self.emp_cov + self.emp_loc[...,None]@self.emp_loc[None,...])).unsqueeze(0) / 2.0
        loc_term = (x - self.emp_loc)@u.unsqueeze(1)
        return ch.exp((cov_term - trace_term - loc_term + self.pi_const).double())
