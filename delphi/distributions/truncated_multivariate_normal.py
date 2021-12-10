
"""
Truncated multivariate normal distribution without oracle access (ie. unknown truncation set)
"""

import torch as ch
from torch import Tensor
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.linalg as LA
from scipy.linalg import sqrtm
import cox 

from .. import delphi
from ..oracle import UnknownGaussian
from .distributions import distributions
from ..trainer import Trainer
from ..grad import TruncatedMultivariateNormalNLL
from ..utils.datasets import TruncatedNormalDataset, make_train_and_val_distr
from ..utils.helpers import Bounds, check_and_fill_args, cov, Parameters, PSDError

# CONSTANTS 
DEFAULTS = {
        'alpha': (float, 'required'), 
        'epochs': (int, 1),
        'num_trials': (int, 1),
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
        'covariance_matrix': (ch.Tensor, None), 
        'd': (int, 100),
        'early_stopping': (bool, False), 
        'n_iter_no_change': (int, 5),
        'verbose': (bool, False),
}


class TruncatedMultivariateNormal(distributions):
    """
    Truncated multivariate normal distribution class.
    """
    def __init__(self,
            args: dict, 
            store: cox.store.Store=None):
        super(TruncatedMultivariateNormal).__init__()
        # instance variables 
        assert store is None or isinstance(store, cox.store.Store), "store is type: {}. expecting cox.store.Store.".format(type(store))
        self.store = store 
        self.truncated = None
        # algorithm hyperparameters
        self.args = check_and_fill_args(Parameters(args), DEFAULTS)

    def fit(self, S: Tensor):
        
        assert isinstance(S, Tensor), "S is type: {}. expected type torch.Tensor.".format(type(S))
        assert S.size(0) > S.size(1), "input expected to bee num samples by dimenions, current input is size {}.".format(S.size()) 
        # run PGD for parameter estimation 
        self.trainer.train_model((self.train_loader_, self.val_loader_))
        while True:
            try:
                self.train_loader_, self.val_loader_ = make_train_and_val_distr(self.args, S, TruncatedNormalDataset)
                self.truncated = TruncatedMultivariateNormalModel(self.args, self.train_loader_.dataset)
                # run PGD to predict actual estimates
                self.trainer = Trainer(self.truncated, max_iter=self.args.epochs, trials=self.args.num_trials, tol=self.args.tol, 
                                    store=self.store, verbose=self.args.verbose, early_stopping=self.args.early_stopping)
        
                # run PGD for parameter estimation 
                self.trainer.train_model((self.train_loader_, self.val_loader_))
                return self
            except PSDError as psd:
                print(psd.message) 
                continue
            except Exception as e: 
                    raise e

#        # rescale/standardize
#        self.truncated.model.covariance_matrix.data = self.truncated.model.covariance_matrix @ self.emp_covariance_matrix
#        self.truncated.model.loc.data = (self.truncated.model.loc[None,...] @ Tensor(sqrtm(self.emp_covariance_matrix.numpy()))).flatten() + self.emp_loc
    
    @property 
    def covariance_matrix(self): 
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


class TruncatedMultivariateNormalModel(delphi.delphi):
    '''
    Model for truncated normal distributions to be passed into trainer.
    '''
    def __init__(self, args, train_ds):
        '''
        Args: 
            args (delphi.utils.Parameters) : parameter object holding hyperparameters
        '''
        super().__init__(args)
        self.args = args
        self.train_ds = train_ds        
        # initialiaze pseudo oracle for gaussians with unknown truncation 
        self.phi = UnknownGaussian(self.train_ds.loc, self.train_ds.covariance_matrix, self.train_ds.S, self.args.d)
        # exponent class
        self.exp_h = Exp_h(self.train_ds.loc, self.train_ds.covariance_matrix)
        self.emp_loc, self.emp_covariance_matrix = None, None
        # initialize empirical estimates
        self.calc_emp_model()
       
    def pretrain_hook(self):
        # parameterize projection set
        if self.args.covariance_matrix is not None:
            B = self.args.covariance_matrix.clone().inverse()
        else:
            B = self.train_ds.covariance_matrix.inverse() 
        u = (self.train_ds.loc[None,...] @ B).flatten()
        # initialize projection set
        self.radius = self.args.r * ch.sqrt(ch.log(1.0 / Tensor([self.args.alpha])))
        # upper and lower bounds
        if self.args.clamp:
            self.loc_bounds = Bounds(self.train_ds.loc - self.radius, self.train_ds.loc + self.radius)
            if self.args.covariance_matrix is None:  
                eig_decomp = LA.eig(self.train_ds.covariance_matrix)
                self.scale_bounds = Bounds(ch.full((self.train_ds.S.size(1),), float((Tensor([self.args.alpha]) / 12.0).pow(2))), eig_decomp.eigenvalues.float() + self.radius)
        else:
            pass

        self.model = MultivariateNormal(u, B)
        self.model.loc.requires_grad, self.model.covariance_matrix.requires_grad = True, True
        # if distribution with known variance, remove from computation graph
        if self.args.covariance_matrix is not None: self.model.covariance_matrix.requires_grad = False
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
            self.model.loc.data = ch.cat([ch.clamp(self.model.loc[i], self.loc_bounds.lower[i], self.loc_bounds.upper[i]).unsqueeze(0) for i in range(self.model.loc.shape[0])])
            if self.args.covariance_matrix is None:
                u, s, v = self.model.covariance_matrix.svd()  # decompose covariance estimate
                self.model.covariance_matrix.data = u.matmul(ch.diag(ch.cat([ch.clamp(s[i], self.scale_bounds.lower[i], self.scale_bounds.upper[i]).unsqueeze(0) for i in range(s.shape[0])]))).matmul(v.t())
        else:
            pass

        # check that the covariance matrix is PSD
        if (LA.eig(self.model.covariance_matrix).eigenvalues.float() < 0).any(): 
            raise PSDError('covariance matrix is not PSD, rerunning procedure')

    def calc_emp_model(self): 
        # initialize projection set
        self.emp_covariance_matrix = self.train_ds.covariance_matrix
        self.emp_loc = self.train_ds.loc
        self.model = MultivariateNormal(self.emp_loc, self.emp_covariance_matrix)

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
