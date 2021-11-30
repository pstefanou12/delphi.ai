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
from ..oracle import oracle
from ..trainer import Trainer
from ..utils.helpers import Bounds
from ..utils.datasets import CensoredNormalDataset


class CensoredMultivariateNormal(CensoredNormal):
    """
    Censored multivariate distribution class.
    """
    def __init__(self,
            phi: oracle,
            alpha: float,
            epochs: int=1,
            clamp: bool=True,
            val: int=50,
            tol: float=1e-2,
            r: float=2.0,
            num_samples: int=100,
            bs: int=10,
            lr: float=1e-1,
            step_lr: int=100, 
            custom_lr_multiplier: str=None,
            lr_interpolation: str=None,
            step_lr_gamma: float=.9,
            momentum: float=0.0, 
            weight_decay: float=0.0,
            eps: float=1e-5, 
            **kwargs):
        """
        """
        super().__init__(phi, alpha, epochs, clamp, val, tol, r, num_samples, bs, lr, step_lr, custom_lr_multiplier, lr_interpolation, step_lr_gamma, momentum, weight_decay, eps, **kwargs)

    def fit(self, S: Tensor):
        """
        """
        # separate into training and validation set
        rand_indices = ch.randperm(S.size(0))
        train_indices, val_indices = rand_indices[self.args.val:], rand_indices[:self.args.val]
        self.X_train = S[train_indices]
        self.X_val = S[val_indices]
        self.train_ds = CensoredNormalDataset(self.X_train)
        self.val_ds = CensoredNormalDataset(self.X_val)
        self.train_loader_ = DataLoader(self.train_ds, batch_size=self.args.bs)
        self.val_loader_ = DataLoader(self.val_ds, batch_size=len(self.val_ds))

        self.censored = CensoredMultivariateNormalModel(self.args, self.train_ds, self.phi, self.custom_lr_multiplier, self.lr_interpolation, self.step_lr, self.step_lr_gamma)
        # run PGD to predict actual estimates
        self.trainer = Trainer(self.censored)

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
    def __init__(self, args, train_ds, phi, custom_lr_multiplier, lr_interpolation, step_lr, step_lr_gamma): 
        '''
        Args: 
            args (cox.utils.Parameters) : parameter object holding hyperparameters
        '''
        super().__init__(args, train_ds, phi, custom_lr_multiplier, lr_interpolation, step_lr, step_lr_gamma)

    def pretrain_hook(self):
        # initialize projection set
        self.radius = self.args.r * ch.sqrt(ch.log(1.0 / self.args.alpha))
        eig_decomp = LA.eig(self.train_ds.covariance_matrix)
        # upper and lower bounds
        if self.args.clamp:
            self.loc_bounds, self.scale_bounds = Bounds(self.train_ds.loc - self.radius, self.train_ds.loc + self.radius), Bounds(ch.full((self.train_ds.S.size(1),), float((self.args.alpha / 12.0).pow(2))), eig_decomp.eigenvalues.float() + self.radius)
        else:
            pass
   
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
            eig_decomp = LA.eig(self.model.covariance_matrix) 
            self.model.loc.data = ch.cat(
                [ch.clamp(self.model.loc[i], float(self.loc_bounds.lower[i]), float(self.loc_bounds.upper[i])).unsqueeze(0) for i in
                 range(self.model.loc.size(0))])
            self.model.covariance_matrix.data = eig_decomp.eigenvectors.float()@ch.diag(ch.cat(
                [ch.clamp(eig_decomp.eigenvalues[i].float(), float(self.scale_bounds.lower[i]), float(self.scale_bounds.upper[i])).unsqueeze(0) for i in
                 range(self.model.loc.size(0))]))@eig_decomp.eigenvectors.T.float()
        else:
            pass

