
"""
Truncated multivariate normal distribution without oracle access (ie. unknown truncation set)
"""

import torch as ch
from torch import Tensor
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data import DataLoader
import torch.linalg as LA
from scipy.linalg import sqrtm

from .truncated_normal import TruncatedNormal, TruncatedNormalModel, Exp_h
from ..trainer import Trainer
from ..grad import TruncatedMultivariateNormalNLL
from ..utils.datasets import TruncatedNormalDataset
from ..utils.helpers import Bounds, cov, Parameters


class TruncatedMultivariateNormal(TruncatedNormal):
    """
    Truncated multivariate normal distribution class.
    """
    def __init__(
            self,
            alpha: float,
            d: int=100,
            epochs: int=1,
            clamp: bool=True,
            val: int=50,
            tol: float=1e-2,
            r: float=2.0,
            bs: int=1,
            lr: float=1e-1,
            step_lr: int=100, 
            custom_lr_multiplier: str=None,
            lr_interpolation: str=None,
            step_lr_gamma: float=.9,
            momentum: float=0.0, 
            eps: float=1e-5, 
            **kwargs):
        super().__init__(alpha, d, epochs, clamp, val, tol, r, bs, lr, step_lr, custom_lr_multiplier, step_lr_gamma, eps)

    def fit(self, S: Tensor):
        self.emp_loc, self.emp_covariance_matrix = S.mean(0), cov(S)
        self.S_norm = (S - self.emp_loc) @ Tensor(sqrtm(self.emp_covariance_matrix.numpy())).inverse()
        rand_indices = ch.randperm(S.size(0))
        train_indices, val_indices = rand_indices[self.args.val:], rand_indices[:self.args.val]
        self.X_train = self.S_norm[train_indices]
        self.X_val = self.S_norm[val_indices]
        self.train_ds = TruncatedNormalDataset(self.X_train)
        self.val_ds = TruncatedNormalDataset(self.X_val)
        self.train_loader_ = DataLoader(self.train_ds, batch_size=self.args.bs)
        self.val_loader_ = DataLoader(self.val_ds, batch_size=len(self.val_ds))

        self.truncated = TruncatedMultivariateNormalModel(self.args, self.train_ds, self.custom_lr_multiplier, self.lr_interpolation, self.step_lr, self.step_lr_gamma)
        # run PGD to predict actual estimates
        self.trainer = Trainer(self.truncated)
        # run PGD for parameter estimation 
        self.trainer.train_model((self.train_loader_, self.val_loader_))

        # rescale/standardize
        self.truncated.model.covariance_matrix.data = self.truncated.model.covariance_matrix @ self.emp_covariance_matrix
        self.truncated.model.loc.data = (self.truncated.model.loc[None,...] @ Tensor(sqrtm(self.emp_covariance_matrix.numpy()))).flatten() + self.emp_loc
    
    @property 
    def covariance_matrix(self): 
        """
        Returns the standard deviation for the normal distribution.
        """
        return self.truncated.model.covariance_matrix.clone()


class TruncatedMultivariateNormalModel(TruncatedNormalModel):
    '''
    Model for truncated normal distributions to be passed into trainer.
    '''
    def __init__(self, args, train_ds, custom_lr_multiplier, lr_interpolation, step_lr, step_lr_gamma): 
        '''
        Args: 
            args (cox.utils.Parameters) : parameter object holding hyperparameters
        '''
        super().__init__(args, train_ds, custom_lr_multiplier, lr_interpolation, step_lr, step_lr_gamma)
       
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
        
