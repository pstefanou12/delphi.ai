"""
Truncated multivariate normal distribution with oracle access (ie. known truncation set).
"""

import torch as ch
from torch import Tensor
from torch.distributions.multivariate_normal import MultivariateNormal
import cox
import math
import torch.nn as nn

from .. import delphi
from .distributions import distributions
from ..utils.datasets import TruncatedNormalDataset, make_train_and_val_distr
from ..grad import TruncatedMultivariateNormalNLL 
from ..trainer import Trainer
from ..utils.helpers import PSDError, Parameters, is_psd
from ..utils.defaults import check_and_fill_args, TRAINER_DEFAULTS, DELPHI_DEFAULTS, TRUNC_MULTI_NORM_DEFAULTS


class TruncatedMultivariateNormal(distributions):
    """
    Truncated multivariate normal distribution class with known truncation set.
    """
    def __init__(self,
            args: Parameters,
            store: cox.store.Store=None):
        """
        """
        # instance variables
        assert isinstance(args, Parameters), "args is type: {}. expecting args to be type delphi.utils.helpers.Parameters"
        assert store is None or isinstance(store, cox.store.Store), "store is type: {}. expecting cox.store.Store.".format(type(store))
        self.store = store 
        self.censored = None
        # algorithm hyperparameters
        TRUNC_MULTI_NORM_DEFAULTS.update(TRAINER_DEFAULTS)
        TRUNC_MULTI_NORM_DEFAULTS.update(DELPHI_DEFAULTS)
        self.args = check_and_fill_args(args, TRUNC_MULTI_NORM_DEFAULTS)

    def fit(self, S: Tensor):
        """
        """
        assert isinstance(S, Tensor), "S is type: {}. expected type torch.Tensor.".format(type(S))
        assert S.size(0) > S.size(1), "input expected to be shape num samples by dimenions, current input is size {}.".format(S.size()) 
        
        while True: 
            try: 
                self.train_loader_, self.val_loader_ = make_train_and_val_distr(self.args, S, TruncatedNormalDataset)
                self.truncated = TruncatedMultivariateNormalModel(self.args, self.train_loader_.dataset)

                # run PGD to predict actual estimates
                trainer = Trainer(self.truncated, self.args, store=self.store)
                trainer.train_model(self.train_loader_, self.val_loader_)
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
        return self.truncated.model.loc.clone()
    
    @property
    def covariance_matrix_(self): 
        """
        Returns the covariance matrix of the distribution.
        """
        return self.truncated.model.covariance_matrix.clone()


class TruncatedMultivariateNormalModel(delphi.delphi):
    """
    Model for truncated multivariate normal distributions with known truncation to be passed into trainer.
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
        self._criterion = TruncatedMultivariateNormalNLL 
        self.criterion_params = [self.args.phi, self.args.num_samples, self.args.eps]
        # initialize empirical estimates
        self.calc_emp_model()

    def pretrain_hook(self, train_loader):
        # parameterize projection set
        self.radius = self.args.r * (math.log(1.0 / self.args.alpha) / (self.args.alpha ** 2)) + 12
        if self.args.covariance_matrix is not None:
            self.T = self.args.covariance_matrix.clone().inverse()
        else:
            self.T = self.emp_covariance_matrix.clone().inverse()
        self.v = self.emp_loc.clone() @ self.T

        # Initialize empirical model 
        self.model = MultivariateNormal(self.v, self.T)
        self.model.loc.requires_grad, self.model.covariance_matrix.requires_grad = True, True
        # Register parameters with Pytorch
        self.register_parameter('loc', nn.Parameter(self.v))
        self.register_parameter('covariance_matrix', nn.Parameter(self.T)) 
        # if distribution with known variance, remove from computation graph
        if self.args.covariance_matrix is not None: 
            for name, param in self.named_parameters():
                if name == 'covariance_matrix':
                    param.requires_grad = False
    
    def calc_emp_model(self): 
        # initialize projection set
        if 'covariance_matrix' in self.args: 
            self.emp_covariance_matrix = self.args.covariance_matrix
        else:   
            self.emp_covariance_matrix = self.train_ds.covariance_matrix
        self.emp_loc = self.train_ds.loc
        self.model = MultivariateNormal(self.emp_loc, self.emp_covariance_matrix)

    def __call__(self, batch, targ):
        """
        Training step for defined model.
        Args: 
            i (int) : gradient step or epoch number
            batch (Iterable) : iterable of inputs that 
        """
        loss = TruncatedMultivariateNormalNLL.apply(self.model.loc, self.model.covariance_matrix, batch, targ, self.args.phi, self.args.num_samples, self.args.eps)
        return loss, None, None

    def iteration_hook(self, i, is_train, loss, batch) -> None:
        # Project location to ball around v
        loc_diff = self.optimizer.param_groups[0]['params'][0] - self.v
        loc_norm = ch.norm(loc_diff)
        if loc_norm > self.radius:
            self.optimizer.param_groups[0]['params'][0] = self.v + (loc_diff / loc_norm) * self.radius
            
        # Project covariance to PSD cone with bounded deviation from T
        cov = self.optimizer.param_groups[0]['params'][1] 
    
        # Ensure PSD via eigenvalue clipping
        L, Q = ch.linalg.eigh(cov)
        L_clipped = ch.clamp(L, min=1e-6)  # Ensure positive eigenvalues
        cov_psd = Q @ ch.diag_embed(L_clipped) @ Q.T
    
        # Project to Frobenius ball around T if needed
        cov_diff = cov_psd - self.T
        frob_norm = ch.linalg.norm(cov_diff, ord='fro')
        if frob_norm > self.radius:
            cov_projected = self.T + (cov_diff / frob_norm) * self.radius
            # Re-ensure PSD after projection
            L, Q = ch.linalg.eigh(cov_projected)
            L_clipped = ch.clamp(L, min=1e-6)
            self.optimizer.param_groups[0]['params'][1] = Q @ ch.diag_embed(L_clipped) @ Q.T
        else:
            self.optimizer.param_groups[0]['params'][1] = cov_psd

    def post_training_hook(self): 
        self.args.r *= self.args.rate
        # reparamterize distribution
        self.model.covariance_matrix.requires_grad, self.model.loc.requires_grad = False, False

        for name, param in self.named_parameters(): 
            if name == 'loc': 
                v = param
            else: 
                T = param

        self.model.covariance_matrix.data = T.inverse()
        self.model.loc.data = v @ self.model.covariance_matrix
    