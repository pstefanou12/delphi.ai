"""
Censored multivariate normal distribution with oracle access (ie. known truncation set).
"""

import torch as ch
from torch import Tensor
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data import DataLoader
import torch.linalg as LA
from cox.utils import Parameters

from .. import delphi
from .normal import CensoredNormal, CensoredNormalModel
from ..utils.datasets import CensoredNormalDataset, make_train_and_val_distr
from ..oracle import oracle
from ..trainer import Trainer
from ..utils.helpers import Bounds, check_and_fill_args
from ..utils.datasets import CensoredNormalDataset

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
        'covariance_matrix': (ch.Tensor, None),
}

class CensoredMultivariateNormal(CensoredNormal):
    """
    Censored multivariate distribution class.
    """
    def __init__(self,
            phi: oracle,
            alpha: float,
            kwargs: dict={}):
        """
        """
        super().__init__(phi, alpha, kwargs)

    def fit(self, S: Tensor):
        """
        """
        assert isinstance(S, Tensor), "S is type: {}. expected type torch.Tensor.".format(type(S))
        assert S.size(0) > S.size(1), "input expected to bee num samples by dimenions, current input is size {}.".format(S.size()) 
        self.train_loader_, self.val_loader_ = make_train_and_val_distr(self.args, S, CensoredNormalDataset)
        self.censored = CensoredMultivariateNormalModel(self.args, self.train_loader_.dataset, self.phi)
        # run PGD to predict actual estimates
        self.trainer = Trainer(self.censored, self.args.epochs, self.args.num_trials, self.args.tol)

        # run PGD for parameter estimation 
        self.trainer.train_model((self.train_loader_, self.val_loader_))
        
    @property
    def covariance_matrix(self): 
        '''
        Returns the covariance matrix of the distribution.
        '''
        return self.censored.model.covariance_matrix.clone()


class CensoredMultivariateNormalModel(CensoredNormalModel):
    '''
    Model for censored normal distributions to be passed into trainer.
    '''
    def __init__(self, args, train_ds, phi): 
        '''
        Args: 
            args (cox.utils.Parameters) : parameter object holding hyperparameters
        '''
        super().__init__(args, train_ds, phi) 

    def pretrain_hook(self):
        self.radius = self.args.r * (ch.log(1.0 / self.args.alpha) / self.args.alpha.pow(2))
        # parameterize projection set
        if self.args.covariance_matrix is not None:
            T = self.args.covariance_matrix.clone().inverse()
        else:
            T = self.emp_covariance_matrix.clone().inverse()
        v = self.emp_loc.clone() @ T

        # upper and lower bounds
        if self.args.clamp:
            self.loc_bounds = Bounds(v - self.radius, self.train_ds.loc + self.radius)
            if self.args.covariance_matrix is None:
                # initialize covaraince matrix projection set around its empirical eigenvalues
                eig_decomp = LA.eig(T)
                self.scale_bounds = Bounds(ch.full((self.train_ds.S.size(1),), float((self.args.alpha / 12.0).pow(2))), eig_decomp.eigenvalues.float() + self.radius)
        else:
            pass

        # initialize empirical model 
        self.model = MultivariateNormal(v, T)
        self.model.loc.requires_grad, self.model.covariance_matrix.requires_grad = True, True
        # if distribution with known variance, remove from computation graph
        if self.args.covariance_matrix is not None: self.model.covariance_matrix.requires_grad = False
        self.params = [self.model.loc, self.model.covariance_matrix]
   
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
            self.model.loc.data = ch.cat(
                [ch.clamp(self.model.loc[i], float(self.loc_bounds.lower[i]), float(self.loc_bounds.upper[i])).unsqueeze(0) for i in
                 range(self.model.loc.size(0))])
            if self.args.covariance_matrix is None:
                eig_decomp = LA.eig(self.model.covariance_matrix) 
                self.model.covariance_matrix.data = eig_decomp.eigenvectors.float()@ch.diag(ch.cat(
                [ch.clamp(eig_decomp.eigenvalues[i].float(), float(self.scale_bounds.lower[i]), float(self.scale_bounds.upper[i])).unsqueeze(0) for i in
                 range(self.model.loc.size(0))]))@eig_decomp.eigenvectors.T.float()
        else:
            pass

