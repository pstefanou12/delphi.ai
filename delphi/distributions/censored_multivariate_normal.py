"""
Censored multivariate normal distribution with oracle access (ie. known truncation set).
"""

import torch as ch
from torch import Tensor
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.linalg as LA
import cox
import math

from .. import delphi
from .distributions import distributions
from ..utils.datasets import CensoredNormalDataset, make_train_and_val_distr
from ..grad import CensoredMultivariateNormalNLL
from ..trainer import Trainer
from ..utils.helpers import PSDError, Parameters
from ..utils.datasets import CensoredNormalDataset
from ..utils.defaults import check_and_fill_args, TRAINER_DEFAULTS, DELPHI_DEFAULTS, CENSOR_MULTI_NORM_DEFAULTS


class CensoredMultivariateNormal(distributions):
    """
    Censored multivariate distribution class.
    """
    def __init__(self,
            args: Parameters,
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
        CENSOR_MULTI_NORM_DEFAULTS.update(TRAINER_DEFAULTS)
        CENSOR_MULTI_NORM_DEFAULTS.update(DELPHI_DEFAULTS)
        self.args = check_and_fill_args(args, CENSOR_MULTI_NORM_DEFAULTS)

    def fit(self, S: Tensor):
        """
        """
        assert isinstance(S, Tensor), "S is type: {}. expected type torch.Tensor.".format(type(S))
        assert S.size(0) > S.size(1), "input expected to be shape num samples by dimenions, current input is size {}.".format(S.size()) 
        
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
    def loc_(self): 
        """
        Returns the mean of the normal disribution.
        """
        return self.censored.model.loc.clone()
    
    @property
    def covariance_matrix_(self): 
        """
        Returns the covariance matrix of the distribution.
        """
        return self.censored.model.covariance_matrix.clone()


class CensoredMultivariateNormalModel(delphi.delphi):
    """
    Model for censored normal distributions to be passed into trainer.
    """
    def __init__(self, args, train_ds): 
        """
        Args: 
            args (cox.utils.Parameters) : parameter object holding hyperparameters
        """
        super().__init__(args)
        self.train_ds = train_ds
        self.model = None
        self.emp_loc, self.emp_covariance_matrix = None, None
        # initialize empirical estimates
        self.calc_emp_model()

    def pretrain_hook(self):
        self.radius = self.args.r * (math.log(1.0 / self.args.alpha) / (self.args.alpha ** 2)) + 12
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
        """
        Training step for defined model.
        Args: 
            i (int) : gradient step or epoch number
            batch (Iterable) : iterable of inputs that 
        """
        loss = CensoredMultivariateNormalNLL.apply(self.model.loc, self.model.covariance_matrix, *batch, self.args.phi, self.args.num_samples, self.args.eps)
        return loss, None, None

    def iteration_hook(self, i, loop_type, loss, prec1, prec5, batch):
        """
        Iteration hook for defined model. Method is called after each 
        training update.
        Args:
            loop_type (str) : "train" or "val"; indicating type of loop
            loss (ch.Tensor) : loss for that iteration
            prec1 (float) : accuracy for top prediction
            prec5 (float) : accuracy for top-5 predictions
        """
        loc_diff = self.model.loc - self.v
        loc_diff = loc_diff[...,None].renorm(p=2, dim=0, maxnorm=self.radius).flatten()
        self.model.loc.data = self.v + loc_diff
        cov_diff = self.model.covariance_matrix - self.T
        cov_diff = cov_diff.renorm(p=2, dim=0, maxnorm=self.radius)
        self.model.covariance_matrix.data = self.T + cov_diff 
        
        # check that the covariance matrix is PSD
        eig_vals = ch.view_as_real(LA.eig(self.model.covariance_matrix).eigenvalues)[:,0]
        # print("real eig vals: ", ch.view_as_real(eig_vals))
        if (eig_vals < 0).any(): 
            raise PSDError("covariance matrix is not PSD, rerunning procedure")

    def post_training_hook(self): 
        self.args.r *= self.args.rate
        # reparamterize distribution
        self.model.covariance_matrix.requires_grad, self.model.loc.requires_grad = False, False
        self.model.covariance_matrix.data = self.model.covariance_matrix.inverse()
        self.model.loc.data = self.model.loc  @ self.model.covariance_matrix
