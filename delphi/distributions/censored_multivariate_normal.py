"""
Censored multivariate normal distribution with oracle access (ie. known truncation set).
"""

import torch as ch
from torch import Tensor
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.linalg as LA
import cox
from cox.utils import Parameters
import math
from typing import Callable

from .. import delphi
from .distributions import distributions
from ..utils.datasets import CensoredNormalDataset, make_train_and_val_distr
from ..grad import CensoredMultivariateNormalNLL
from ..trainer import Trainer
from ..utils.helpers import check_and_fill_args, PSDError
from ..utils.datasets import CensoredNormalDataset

# CONSTANTS 
DEFAULTS = {
        'phi': (Callable, 'required'),
        'alpha': (float, 'required'), 
        'epochs': (int, 1),
        'trials': (int, 3),
        'val': (float, .2),
        'lr': (float, 1e-1), 
        'step_lr': (int, 100),
        'step_lr_gamma': (float, .9), 
        'adam': (bool, False),
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
        'early_stopping': (bool, False), 
        'n_iter_no_change': (int, 5),
        'verbose': (bool, False),
}


class CensoredMultivariateNormal(distributions):
    """
    Censored multivariate distribution class.
    """
    def __init__(self,
            args: dict,
            store: cox.store.Store=None):
        """
        """
        super(CensoredMultivariateNormal).__init__()
        # instance variables
        assert isinstance(args, Parameters), "args is type: {}. expecting args to be type delphi.utils.helpers.Parameters"
        assert store is None or isinstance(store, cox.store.Store), "store is type: {}. expecting cox.store.Store.".format(type(store))
        self.store = store 
        self.censored = None
        # algorithm hyperparameters
        self.args = check_and_fill_args(Parameters(args), DEFAULTS)

    def fit(self, S: Tensor):
        """
        """
        assert isinstance(S, Tensor), "S is type: {}. expected type torch.Tensor.".format(type(S))
        assert S.size(0) > S.size(1), "input expected to bee num samples by dimenions, current input is size {}.".format(S.size()) 
        
        while True: 
            try: 
                self.train_loader_, self.val_loader_ = make_train_and_val_distr(self.args, S, CensoredNormalDataset)
                self.censored = CensoredMultivariateNormalModel(self.args, self.train_loader_.dataset)
                # run PGD to predict actual estimates
                trainer = Trainer(self.censored, self.args, store=self.store)
                trainer.train_model((self.train_loader_, self.val_loader_))
                return self
            except PSDError as psd:
                print(psd.message) 
                continue
            except Exception as e: 
                raise e
    
    @property 
    def loc(self): 
        """
        Returns the mean of the normal disribution.
        """
        return self.censored.model.loc.clone()
    
    @property
    def covariance_matrix(self): 
        '''
        Returns the covariance matrix of the distribution.
        '''
        return self.censored.model.covariance_matrix.clone()


class CensoredMultivariateNormalModel(delphi.delphi):
    '''
    Model for censored normal distributions to be passed into trainer.
    '''
    def __init__(self, args, train_ds): 
        '''
        Args: 
            args (cox.utils.Parameters) : parameter object holding hyperparameters
        '''
        super().__init__(args)
        self.train_ds = train_ds
        self.model = None
        self.emp_loc, self.emp_covariance_matrix = None, None
        # initialize empirical estimates
        self.calc_emp_model()

    def pretrain_hook(self):
        self.radius = self.args.r * (math.log(1.0 / self.args.alpha) / (self.args.alpha ** 2))
        # parameterize projection set
        if self.args.covariance_matrix is not None:
            self.T = self.args.covariance_matrix.clone().inverse()
        else:
            self.T = self.emp_covariance_matrix.clone().inverse()
        self.v = self.emp_loc.clone() @ self.T

        # initialize empirical model 
        self.model = MultivariateNormal(self.v, self.T)
        self.model.loc.requires_grad, self.model.covariance_matrix.requires_grad = True, True
        # if distribution with known variance, remove from computation graph
        if self.args.covariance_matrix is not None: self.model.covariance_matrix.requires_grad = False
        self.params = [self.model.loc, self.model.covariance_matrix]
    
    def calc_emp_model(self): 
        # initialize projection set
        self.emp_covariance_matrix = self.train_ds.covariance_matrix
        self.emp_loc = self.train_ds.loc
        self.model = MultivariateNormal(self.emp_loc, self.emp_covariance_matrix)

    def __call__(self, batch):
        '''
        Training step for defined model.
        Args: 
            i (int) : gradient step or epoch number
            batch (Iterable) : iterable of inputs that 
        '''
        loss = CensoredMultivariateNormalNLL.apply(self.model.loc, self.model.covariance_matrix, *batch, self.args.phi, self.args.num_samples, self.args.eps)
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
        loc_diff = self.model.loc - self.v
        loc_diff = loc_diff[...,None].renorm(p=2, dim=0, maxnorm=self.radius).flatten()
        self.model.loc.data = self.v + loc_diff
        cov_diff = self.model.covariance_matrix - self.T
        cov_diff = cov_diff.renorm(p=2, dim=0, maxnorm=self.radius)
        self.model.covariance_matrix.data = self.T + cov_diff 
        
        cov_inv = self.model.covariance_matrix.inverse()
        cov_inv = cov_inv.renorm(p=2, dim=0, maxnorm=self.radius)
        self.model.covariance_matrix.data = cov_inv.inverse()

        # check that the covariance matrix is PSD
        if (LA.eig(self.model.covariance_matrix).eigenvalues.float() < 0).any(): 
            raise PSDError('covariance matrix is not PSD, rerunning procedure')

    def post_training_hook(self): 
        self.args.r *= self.args.rate
        # reparamterize distribution
        self.model.covariance_matrix.requires_grad, self.model.loc.requires_grad = False, False
        self.model.covariance_matrix.data = self.model.covariance_matrix.inverse()
        self.model.loc.data = self.model.loc  @ self.model.covariance_matrix
