
"""
Truncated multivariate normal distribution without oracle access (ie. unknown truncation set)
"""

import torch as ch
from torch import Tensor
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data import DataLoader
from cox.utils import Parameters
import config

from .unknown_truncation_normal import TruncatedNormal, TruncatedNormalModel, Exp_h
from ..trainer import Trainer
from ..grad import TruncatedMultivariateNormalNLL
from ..utils.datasets import TruncatedMultivariateNormalDataset
from ..utils.helpers import Bounds


class TruncatedMultivariateNormal(TruncatedNormal):
    """
    Truncated multivariate normal distribution class.
    """
    def __init__(
            self,
            alpha: float,
            d: int=100,
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
        super().__init__(alpha, d, steps, clamp, n, val, tol, workers, r, num_samples, bs, lr, step_lr, custom_lr_multiplier, step_lr_gamma, eps)

    def fit(self, S: Tensor):
        # separate into training and validation set
        rand_indices = ch.randperm(S.size(0))
        train_indices, val_indices = rand_indices[self.val:], rand_indices[:self.val]
        self.X_train = S[train_indices]
        self.X_val = S[val_indices]

        self.train_ds = TruncatedMultivariateNormalDataset(self.X_train)
        self.val_ds = TruncatedMultivariateNormalDataset(self.X_val)
        train_loader = DataLoader(self.train_ds, batch_size=self.bs, num_workers=self.workers)
        val_loader = DataLoader(self.val_ds, batch_size=len(self.val_ds), num_workers=self.workers)

        self.truncated_normal = TruncatedMultivariateNormalModel(config.args, self.d, self.train_ds, self.val_ds, self.tol, self.r, self.alpha, self.clamp, n=self.n)
        # run PGD to predict actual estimates
        self.trainer = Trainer(self.truncated_normal)

        # run PGD for parameter estimation 
        self.trainer.train_model((train_loader, None))


class TruncatedMultivariateNormalModel(TruncatedNormalModel):
    '''
    Model for truncated normal distributions to be passed into trainer.
    '''
    def __init__(self, args, d,  X_train, X_val, tol, r, alpha, clamp, n=100, store=None, table=None, schema=None): 
        '''
        Args: 
            args (cox.utils.Parameters) : parameter object holding hyperparameters
        '''
        super().__init__(args, d, X_train, X_val, tol, r, alpha, clamp, n, store=store, table=table, schema=schema)

        u, s, v = ch.linalg.svd(self.emp_covariance_matrix)

        # upper and lower bounds
        if self.clamp:
            self.loc_bounds, self.scale_bounds = Bounds(self.emp_loc - self.radius, self.emp_loc + self.radius), Bounds(ch.max(self.alpha.pow(2) / 12, \
                                                               s - self.radius),
                                                        s + self.radius)
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
        # increase number of steps taken
        self.steps += 1
        if self.clamp:
            u, s, v = self.model.covariance_matrix.svd()  # decompose covariance estimate
            self.model.loc.data = ch.cat(
                [ch.clamp(self.model.loc[i], float(self.loc_bounds.lower[i]), float(self.loc_bounds.upper[i])).unsqueeze(0) for i in
                 range(self.model.loc.shape[0])])
            self.model.covariance_matrix.data = u.matmul(ch.diag(ch.cat(
                [ch.clamp(s[i], float(self.scale_bounds.lower[i]), float(self.scale_bounds.upper[i])).unsqueeze(0) for i in
                 range(s.shape[0])]))).matmul(v.t())
        else:
            pass

        # check for convergence every n steps
        if self.steps % self.n == 0: 
            grad = self.check_grad()

