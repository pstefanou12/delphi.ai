"""
Censored multivariate normal distribution with oracle access (ie. known truncation set).
"""

import torch as ch
from torch import Tensor
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data import DataLoader
from cox.utils import Parameters
import config

from .. import delphi
from .normal import CensoredNormal, CensoredNormalModel
from ..oracle import oracle
from ..trainer import Trainer
from ..utils.datasets import CensoredMultivariateNormalDataset 
from ..utils.helpers import Bounds
from ..utils import defaults


class CensoredMultivariateNormal(CensoredNormal):
    """
    Censored multivariate distribution class.
    """
    def __init__(self,
            phi: oracle,
            alpha: float,
            steps: int=1000,
            clamp: bool=True,
            n: int=10, 
            val: int=50,
            tol: float=1e-2,
            workers: int=0,
            r: float=2.0,
            num_samples: int=100,
            bs: int=10,
            lr: float=1e-1,
            step_lr: int=100, 
            custom_lr_multiplier: str=None,
            step_lr_gamma: float=.9,
            eps: float=1e-5, 
            **kwargs):
        """
        """
        super().__init__(phi, alpha, steps, clamp, n, val, tol, workers, r, num_samples, bs, lr, step_lr, custom_lr_multiplier, step_lr_gamma, eps, **kwargs)

    def fit(self, S: Tensor):
        """
        """
         # separate into training and validation set
        rand_indices = ch.randperm(S.size(0))
        train_indices, val_indices = rand_indices[self.val:], rand_indices[:self.val]
        self.X_train = S[train_indices]
        self.X_val = S[val_indices]

        self.train_ds = CensoredMultivariateNormalDataset(self.X_train)
        self.val_ds = CensoredMultivariateNormalDataset(self.X_val)
        train_loader = DataLoader(self.train_ds, batch_size=self.bs, num_workers=self.workers)
        val_loader = DataLoader(self.val_ds, batch_size=len(self.val_ds), num_workers=self.workers)

        self.censored_multi_normal = CensoredMultivariateNormalModel(config.args, self.train_ds, self.val_ds, self.phi, self.tol, self.r, self.alpha, self.clamp, n=self.n)
        # run PGD to predict actual estimates
        self.trainer = Trainer(self.censored_multi_normal)

        # run PGD for parameter estimation 
        self.trainer.train_model((train_loader, None))


class CensoredMultivariateNormalModel(CensoredNormalModel):
    '''
    Model for censored normal distributions to be passed into trainer.
    '''
    def __init__(self, args,  X_train, X_val, phi, tol, r, alpha, clamp, n=100, store=None, table=None, schema=None): 
        '''
        Args: 
            args (cox.utils.Parameters) : parameter object holding hyperparameters
        '''
        super().__init__(args, X_train, X_val, phi, tol, r, alpha, clamp, store=store, table=table, schema=schema)

        u, s, v = ch.linalg.svd(self.emp_covariance_matrix)

        # parameterize projection set
        if self.clamp:
            self.loc_bounds, self.scale_bounds = Bounds(self.emp_loc-self.radius, self.emp_loc+self.radius), \
             Bounds(ch.max(ch.square(self.alpha / 12.0), s - self.radius), s + self.radius)
        else:
            pass
        '''
        self.model = MultivariateNormal(self.emp_loc.clone(), self.emp_covariance_matrix.clone())
        self.model.loc.requires_grad, self.model.covariance_matrix.requires_grad = True, True
        self.params = [self.model.loc, self.model.covariance_matrix]
        '''

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
        # increase number of steps taken
        self.steps += 1

        if self.clamp:
            u, s, v = ch.linalg.svd(self.model.covariance_matrix) # decompose covariance estimate
            self.model.loc.data = ch.cat([ch.clamp(self.model.loc[i], self.loc_bounds.lower[i], self.loc_bounds.upper[i]).unsqueeze(0) for i in range(self.model.loc.shape[0])])
            self.model.covariance_matrix.data = u.matmul(ch.diag(ch.cat([ch.clamp(s[i], self.scale_bounds.lower[i], self.scale_bounds.upper[i]).unsqueeze(0) for i in range(s.shape[0])]))).matmul(v.t())
        else:
            pass

        # check for convergence every n steps
        if self.steps % self.n == 0: 
            grad = self.check_grad()

