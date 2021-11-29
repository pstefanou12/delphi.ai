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
from ..utils.helpers import Bounds, cov, Parameters
from ..utils.datasets import TruncatedNormalDataset
from ..grad import TruncatedMultivariateNormalNLL


class TruncatedNormal(distributions):
    """
    Truncated normal distribution class.
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
        super(TruncatedNormal).__init__()
        # instance variables
        self.custom_lr_multiplier = custom_lr_multiplier
        self.step_lr = step_lr 
        self.step_lr_gamma = step_lr_gamma
        self.lr_interpolation = lr_interpolation

        self.args = Parameters({ 
            'alpha': Tensor([alpha]),
            'd': d,
            'bs': bs, 
            'epochs': epochs,
            'momentum': momentum, 
            'weight_decay': 0,
            'lr': lr,  
            'eps': eps,
            'tol': tol,
            'val': val,
            'clamp': clamp,
            'r': r,
            'verbose': False,
        })

    def fit(self, S: Tensor):
        """
        :param S:
        :return:
        """
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

        self.truncated = TruncatedNormalModel(self.args, self.train_ds, self.custom_lr_multiplier, self.lr_interpolation, self.step_lr, self.step_lr_gamma)
        
        # run PGD to predict actual estimates
        self.trainer = Trainer(self.truncated)
        # run PGD for parameter estimation 
        self.trainer.train_model((self.train_loader_, self.val_loader_))

        # rescale/standardize
        self.truncated.model.covariance_matrix.data = self.truncated.model.covariance_matrix @ self.emp_covariance_matrix
        self.truncated.model.loc.data = (self.truncated.model.loc[None,...] @ Tensor(sqrtm(self.emp_covariance_matrix.numpy()))).flatten() + self.emp_loc

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
    def __init__(self, args, train_ds, custom_lr_multiplier, lr_interpolation, step_lr, step_lr_gamma): 
        '''
        Args: 
            args (cox.utils.Parameters) : parameter object holding hyperparameters
        '''
        super().__init__(args, custom_lr_multiplier, lr_interpolation, step_lr, step_lr_gamma)
        self.train_ds = train_ds        
        # initialiaze pseudo oracle for gaussians with unknown truncation 
        self.phi = UnknownGaussian(self.train_ds.loc, self.train_ds.covariance_matrix, self.train_ds.S, self.args.d)

        # exponent class
        self.exp_h = Exp_h(self.train_ds.loc, self.train_ds.covariance_matrix)

        # establish empirical distribution
        B = self.train_ds.covariance_matrix.inverse() 
        u = (self.train_ds.loc[None,...] @ B).flatten()
        self.model = MultivariateNormal(u, B)
        self.model.loc.requires_grad, self.model.covariance_matrix.requires_grad = True, True
        self.params = [self.model.loc, self.model.covariance_matrix]

    def calc_nll(self, S, pdf, loc_grad, cov_grad): 
        """
        Calculates the log likelihood of the current regression estimates of the validation set.
        """
        return TruncatedMultivariateNormalNLL.apply(self.model.loc, self.model.covariance_matrix, S,  pdf, loc_grad, cov_grad, self.phi, self.exp_h)

    def pretrain_hook(self):
        # initialize projection set
        self.radius = self.args.r * ch.sqrt(ch.log(1.0 / self.args.alpha))
        if self.args.clamp:
            self.loc_bounds, self.scale_bounds = Bounds(self.train_ds.loc - self.radius, self.train_ds.loc + self.radius), Bounds(self.args.alpha.pow(2) / 12,
                                                         self.train_ds.covariance_matrix + self.radius)
        else:
            pass

    def train_step(self, i, batch):
        '''
        Training step for defined model.
        Args: 
            i (int) : gradient step or epoch number
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
            self.model.covariance_matrix.data = ch.clamp(self.model.covariance_matrix.data, float(self.scale_bounds.lower), float(self.scale_bounds.upper))
        else:
            pass

    def val_step(self, i, batch): 
        # check for convergence every at each epoch
        loss = self.calc_nll(*batch)
        print("Epoch {} | Log Likelihood: {}".format(i, round(float(abs(loss)), 3)))
        return loss, None, None

    def epoch_hook(self, i, loop_type, loss, prec1, prec5, batch):
        # re-randomize data points in training set 
        self.train_ds.randomize() 

    def post_training_hook(self, val_loader):
        # reparamterize distribution
        self.model.covariance_matrix.requires_grad, self.model.loc.requires_grad = False, False
        self.model.covariance_matrix.data = self.model.covariance_matrix.inverse()
        self.model.loc.data = (self.model.loc[None,...]  @ self.model.covariance_matrix).flatten()
        # set estimated distribution in membership oracle
        self.phi.dist = self.model
      
        # rescale/standardize
        self.model.covariance_matrix.data = self.model.covariance_matrix @ self.train_ds.covariance_matrix
        self.model.loc.data = (self.model.loc[None,...] @ Tensor(sqrtm(self.train_ds.covariance_matrix.numpy()))).flatten() + self.train_ds.loc
        return True

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
