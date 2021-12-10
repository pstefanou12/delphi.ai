"""
Truncated normal distribution without oracle access (ie. unknown truncation set)
"""

import torch as ch
from torch import Tensor
from torch.distributions.multivariate_normal import MultivariateNormal
import cox

from .truncated_multivariate_normal import TruncatedMultivariateNormal, TruncatedMultivariateNormalModel
from .. import delphi
from ..trainer import Trainer
from ..utils.helpers import Bounds, PSDError
from ..utils.datasets import TruncatedNormalDataset, make_train_and_val_distr

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
        'variance': (ch.Tensor, None), 
        'd': (int, 100),
        'early_stopping': (bool, False), 
        'n_iter_no_change': (int, 5),
        'verbose': (bool, False),
}


class TruncatedNormal(TruncatedMultivariateNormal):
    """
    Truncated normal distribution class.
    """
    def __init__(self,
            args: dict, 
            store: cox.store.Store=None):
        super().__init__(args, store=store)

    def fit(self, S: Tensor):
        """
        Runs PSGD to minimize the negative population log likelihood of the truncated distribution.
        Args: 
            S (torch.Tensor): censored samples dataset, expecting n by d matrix (num_samples by dimensions)
        """
        assert isinstance(S, Tensor), "S is type: {}. expected type torch.Tensor.".format(type(S))
        assert S.size(1) == 1, "truncated normal class only accepts 1 dimensional distributions." 
        while True:
            try:
                self.train_loader_, self.val_loader_ = make_train_and_val_distr(self.args, S, TruncatedNormalDataset)
                self.truncated = TruncatedNormalModel(self.args, self.train_loader_.dataset)
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
     #    # rescale/standardize
    #    self.truncated.model.covariance_matrix.data = self.truncated.model.covariance_matrix @ self.emp_covariance_matrix
    #    self.truncated.model.loc.data = (self.truncated.model.loc[None,...] @ Tensor(sqrtm(self.emp_covariance_matrix.numpy()))).flatten() + self.emp_loc
    #
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

        
class TruncatedNormalModel(TruncatedMultivariateNormalModel):
    '''
    Model for truncated normal distributions to be passed into trainer.
    '''
    def __init__(self, args, train_ds): 
        '''
        Args: 
            args (cox.utils.Parameters) : parameter object holding hyperparameters
        '''
        super().__init__(args, train_ds)

    def pretrain_hook(self):
        # parameterize projection set
        if self.args.variance is not None:
            B = self.args.variance.clone()
        else:
            B = self.train_ds.covariance_matrix.inverse() 
        u = (self.train_ds.loc[None,...] @ B).flatten()
        # initialize projection set
        self.radius = self.args.r * ch.sqrt(ch.log(1.0 / Tensor([self.args.alpha])))
        if self.args.clamp:
            self.loc_bounds  = Bounds(self.train_ds.loc - self.radius, self.train_ds.loc + self.radius)             
            if self.args.variance is None: 
                self.scale_bounds = Bounds(Tensor([self.args.alpha]).pow(2) / 12,
                                                         B + self.radius)
        else:
            pass

        self.model = MultivariateNormal(u, B)
        self.model.loc.requires_grad, self.model.covariance_matrix.requires_grad = True, True
        # if distribution with known variance, remove from computation graph
        if self.args.variance is not None: self.model.covariance_matrix.requires_grad = False
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
            self.model.loc.data = ch.clamp(self.model.loc.data, float(self.loc_bounds.lower), float(self.loc_bounds.upper))
            if self.args.variance is None:
                self.model.covariance_matrix.data = ch.clamp(self.model.covariance_matrix.data, float(self.scale_bounds.lower), float(self.scale_bounds.upper))
        else:
            pass
