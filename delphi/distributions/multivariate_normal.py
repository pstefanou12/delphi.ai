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
            iter_: int=1,
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
        super().__init__(phi, alpha, iter_, clamp, val, tol, r, num_samples, bs, lr, step_lr, custom_lr_multiplier, lr_interpolation, step_lr_gamma, momentum, weight_decay, eps, **kwargs)

    def fit(self, S: Tensor):
        """
        """
        self.censored = CensoredMultivariateNormalModel(config.args, S, self.phi, self.custom_lr_multiplier, self.lr_interpolation, self.step_lr, self.step_lr_gamma)
        # run PGD to predict actual estimates
        self.trainer = Trainer(self.censored)

        # run PGD for parameter estimation 
        self.trainer.train_model()
        
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
    def __init__(self, args, S, phi, custom_lr_multiplier, lr_interpolation, step_lr, step_lr_gamma): 
        '''
        Args: 
            args (cox.utils.Parameters) : parameter object holding hyperparameters
        '''
        super().__init__(args, S, phi, custom_lr_multiplier, lr_interpolation, step_lr, step_lr_gamma)

        u, s, v = ch.linalg.svd(self.emp_covariance_matrix)

        # parameterize projection set
        if config.args.clamp:
            self.loc_bounds, self.scale_bounds = Bounds(self.emp_loc-self.radius, self.emp_loc+self.radius), \
             Bounds(ch.full((S.size(1),), float(ch.square(config.args.alpha / 12.0))), s + self.radius)
        else:
            pass

        self.loc_est = self.emp_loc.clone()[None,...]
        self.cov_est = self.emp_covariance_matrix.clone()[None,...]
       
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
        print("cov matrix before projection: {}".format(self.model.covariance_matrix))
        print("cov eigenvalues: {}".format(ch.linalg.eigvals(self.model.covariance_matrix)))

        if config.args.clamp:
            u, s, v = ch.linalg.svd(self.model.covariance_matrix) # decompose covariance estimate
            eigvals = ch.linalg.eigvals(self.model.covariance_matrix).float()
            self.model.loc.data = ch.cat([ch.clamp(self.model.loc[i], self.loc_bounds.lower[i], self.loc_bounds.upper[i]).unsqueeze(0) for i in range(self.model.loc.shape[0])])

            project = Tensor([])
            for i in range(s.size(0)):
                project = ch.cat([project, ch.clamp(eigvals[i], self.scale_bounds.lower[i], self.scale_bounds.upper[i])[None,...]])
            
            self.model.covariance_matrix.data = u@ch.diag(project)@v

            '''
            self.model.covariance_matrix.data = u@ch.diag(ch.cat([ch.clamp(s[i], self.scale_bounds.lower[i], self.scale_bounds.upper[i]).unsqueeze(0) for i in range(s.shape[0])]))@v.T
            '''
        else:
            pass
        self.loc_est = ch.cat([self.loc_est, self.model.loc[None,...]])
        self.cov_est = ch.cat([self.cov_est, self.model.covariance_matrix[None,...]])

        print("cov matrix after projection: {}".format(self.model.covariance_matrix))
        if not (ch.all(ch.linalg.eigvals(self.model.covariance_matrix).float() >= 0)):
            print("u: ", u)
            print("s: ", s) 
            print("v: ", v)
            import pdb; pdb.set_trace()

