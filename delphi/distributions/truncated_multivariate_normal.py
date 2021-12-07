
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
from ..utils.datasets import TruncatedNormalDataset, make_train_and_val_distr
from ..utils.helpers import Bounds, check_and_fill_args, cov, Parameters

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
        'd': (int, 100),
}

class TruncatedMultivariateNormal(TruncatedNormal):
    """
    Truncated multivariate normal distribution class.
    """
    def __init__(self,
            alpha: float,
            kwargs: dict={}):
        super().__init__(alpha, kwargs)

    def fit(self, S: Tensor):
        
        assert isinstance(S, Tensor), "S is type: {}. expected type torch.Tensor.".format(type(S))
        assert S.size(0) > S.size(1), "input expected to bee num samples by dimenions, current input is size {}.".format(S.size()) 
        self.train_loader_, self.val_loader_ = make_train_and_val_distr(self.args, S, TruncatedNormalDataset)
        self.truncated = TruncatedMultivariateNormalModel(self.args, self.train_loader_.dataset)
        # run PGD to predict actual estimates
        self.trainer = Trainer(self.truncated, self.args.epochs, self.args.num_trials, self.args.tol)
        # run PGD for parameter estimation 
        self.trainer.train_model((self.train_loader_, self.val_loader_))

#        # rescale/standardize
#        self.truncated.model.covariance_matrix.data = self.truncated.model.covariance_matrix @ self.emp_covariance_matrix
#        self.truncated.model.loc.data = (self.truncated.model.loc[None,...] @ Tensor(sqrtm(self.emp_covariance_matrix.numpy()))).flatten() + self.emp_loc
    
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
    def __init__(self, args, train_ds):
        '''
        Args: 
            args (delphi.utils.Parameters) : parameter object holding hyperparameters
        '''
        super().__init__(args, train_ds)
       
    def pretrain_hook(self):
        # parameterize projection set
        if self.args.covariance_matrix is not None:
            B = self.args.covariance_matrix.clone().inverse()
        else:
            B = self.train_ds.covariance_matrix.inverse() 
        u = (self.train_ds.loc[None,...] @ B).flatten()
        # initialize projection set
        self.radius = self.args.r * ch.sqrt(ch.log(1.0 / self.args.alpha))
        # upper and lower bounds
        if self.args.clamp:
            self.loc_bounds = Bounds(self.train_ds.loc - self.radius, self.train_ds.loc + self.radius)
            if self.args.covariance_matrix is None:  
                eig_decomp = LA.eig(self.train_ds.covariance_matrix)
                self.scale_bounds = Bounds(ch.full((self.train_ds.S.size(1),), float((self.args.alpha / 12.0).pow(2))), eig_decomp.eigenvalues.float() + self.radius)

        else:
            pass

        self.model = MultivariateNormal(u, B)
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
        
