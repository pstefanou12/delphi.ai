"""
Censored multivariate normal distribution with oracle access (ie. known truncation set).
"""

import torch as ch
from torch import Tensor
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.linalg as LA
import cox
from cox.utils import Parameters

from .. import delphi
from .distributions import distributions
from ..utils.datasets import CensoredNormalDataset, make_train_and_val_distr
from ..oracle import oracle
from ..grad import CensoredMultivariateNormalNLL
from ..trainer import Trainer
from ..utils.helpers import Bounds, check_and_fill_args, PSDError
from ..utils.datasets import CensoredNormalDataset

# CONSTANTS 
DEFAULTS = {
        'phi': (oracle, 'required'),
        'alpha': (float, 'required'), 
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
        'early_stopping': (bool, False), 
        'n_iter_no_change': (int, 5),
        'verbose': (bool, False),
}


class CensoredMultivariateNormal(distributions):
    """
    Censored multivariate distribution class.
    """
    def __init__(self,
            args: dict,
            store: cox.store.Store=None):
        """
        """
        super(CensoredMultivariateNormal).__init__()
        # instance variables
        assert store is None or isinstance(store, cox.store.Store), "store is type: {}. expecting cox.store.Store.".format(type(store))
        self.store = store 
        self.censored = None
        # algorithm hyperparameters
        self.args = check_and_fill_args(Parameters(args), DEFAULTS)

    def fit(self, S: Tensor):
        """
        """
        assert isinstance(S, Tensor), "S is type: {}. expected type torch.Tensor.".format(type(S))
        assert S.size(0) > S.size(1), "input expected to bee num samples by dimenions, current input is size {}.".format(S.size()) 
        
        while True: 
            try: 
                self.train_loader_, self.val_loader_ = make_train_and_val_distr(self.args, S, CensoredNormalDataset)
                self.censored = CensoredMultivariateNormalModel(self.args, self.train_loader_.dataset)
                # run PGD to predict actual estimates
                trainer = Trainer(self.censored, max_iter=self.args.epochs, trials=self.args.num_trials, 
                                        tol=self.args.tol, store=self.store, verbose=self.args.verbose, 
                                        early_stopping=self.args.early_stopping)
                trainer.train_model((self.train_loader_, self.val_loader_))
                return self
            except PSDError as psd:
                print(psd.message) 
                continue
            except Exception as e: 
                raise e
    @property
    def covariance_matrix(self): 
        '''
        Returns the covariance matrix of the distribution.
        '''
        return self.censored.model.covariance_matrix.clone()


class CensoredMultivariateNormalModel(delphi.delphi):
    '''
    Model for censored normal distributions to be passed into trainer.
    '''
    def __init__(self, args, train_ds): 
        '''
        Args: 
            args (cox.utils.Parameters) : parameter object holding hyperparameters
        '''
        super().__init__(args)
        self.train_ds = train_ds
        self.model = None
        self.emp_loc, self.emp_covariance_matrix = None, None
        # initialize empirical estimates
        self.calc_emp_model()

    def pretrain_hook(self):
        self.radius = self.args.r * (ch.log(1.0 / Tensor([self.args.alpha])) / Tensor([self.args.alpha]).pow(2))
        # parameterize projection set
        if self.args.covariance_matrix is not None:
            T = self.args.covariance_matrix.clone().inverse()
        else:
            T = self.emp_covariance_matrix.clone().inverse()
        v = self.emp_loc.clone() @ T

        # upper and lower bounds
        if self.args.clamp:
            self.loc_bounds = Bounds(v - self.radius, self.train_ds.loc + self.radius)
            if self.args.covariance_matrix is None:
                # initialize covaraince matrix projection set around its empirical eigenvalues
                eig_decomp = LA.eig(T)
                self.scale_bounds = Bounds(ch.full((self.train_ds.S.size(1),), float((self.args.alpha / 12.0).pow(2))), eig_decomp.eigenvalues.float() + self.radius)
        else:
            pass

        # initialize empirical model 
        self.model = MultivariateNormal(v, T)
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
        '''
        Training step for defined model.
        Args: 
            i (int) : gradient step or epoch number
            batch (Iterable) : iterable of inputs that 
        '''
        loss = CensoredMultivariateNormalNLL.apply(self.model.loc, self.model.covariance_matrix, *batch, self.args.phi, self.args.num_samples, self.args.eps)
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
            self.model.loc.data = ch.cat([ch.clamp(self.model.loc[i], self.loc_bounds.lower[i], self.loc_bounds.upper[i]).unsqueeze(0) for i in range(self.model.loc.shape[0])])
           # self.model.loc.data = ch.cat(
            #    [ch.clamp(self.model.loc[i], float(self.loc_bounds.lower[i]), float(self.loc_bounds.upper[i])).unsqueeze(0) for i in
            #     range(self.model.loc.size(0))])
            if self.args.covariance_matrix is None:
            ##    eig_decomp = LA.eig(self.model.covariance_matrix) 
            #    print("eig vals: ", eig_decomp.eigenvalues)
            #    print("eig vectors: ", eig_decomp.eigenvectors)
            #    project = ch.diag(ch.cat([ch.clamp(eig_decomp.eigenvalues[i].float(), float(self.scale_bounds.lower[i]), float(self.scale_bounds.upper[i])).unsqueeze(0) for i in
            #     range(self.model.loc.size(0))]))
            #    print("project: ", project)
            #    self.model.covariance_matrix.data = eig_decomp.eigenvectors.float()@ch.diag(ch.cat(
            #    [ch.clamp(eig_decomp.eigenvalues[i].float(), float(self.scale_bounds.lower[i]), float(self.scale_bounds.upper[i])).unsqueeze(0) for i in
            #     range(self.model.loc.size(0))]))@eig_decomp.eigenvectors.T.float()
                u, s, v = self.model.covariance_matrix.svd()  # decompose covariance estimate
            #    self.model.loc.data = ch.cat([ch.clamp(self.model.loc[i], self.loc_bounds.lower[i], self.loc_bounds.upper[i]).unsqueeze(0) for i in range(self.model.loc.shape[0])])
                self.model.covariance_matrix.data = u.matmul(ch.diag(ch.cat([ch.clamp(s[i], self.scale_bounds.lower[i], self.scale_bounds.upper[i]).unsqueeze(0) for i in range(s.shape[0])]))).matmul(v.t())
        else:
            pass

        # check that the covariance matrix is PSD
        if (LA.eig(self.model.covariance_matrix).eigenvalues.float() < 0).any(): 
            raise PSDError('covariance matrix is not PSD, rerunning procedure')

    def post_training_hook(self): 
        self.args.r *= self.args.rate
        # reparamterize distribution
        self.model.covariance_matrix.requires_grad, self.model.loc.requires_grad = False, False
        self.model.covariance_matrix.data = self.model.covariance_matrix.inverse()
        self.model.loc.data = self.model.loc  @ self.model.covariance_matrix
