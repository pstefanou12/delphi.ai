"""
Censored normal distribution with oracle access (ie. known truncation set)
"""

import torch as ch
from torch import Tensor
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data import DataLoader
from cox.utils import Parameters
import config
import copy

from .. import delphi
from .distributions import distributions
from ..oracle import oracle
from ..trainer import Trainer
from ..utils.datasets import CensoredNormalDataset
from ..grad import CensoredMultivariateNormalNLL
from ..utils import defaults
from ..utils.helpers import Bounds, censored_sample_nll


class CensoredNormal(distributions):
    """
    Censored normal distribution class.
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
        Args:
            
        """
        super(CensoredNormal).__init__()
        # instance variables
        self.phi = phi 
        self.alpha = Tensor([alpha]) 
        self.clamp = clamp 
        self.n = n 
        self.val = val 
        self.tol = tol 
        self.workers = workers 
        self.r = r 
        self.bs = bs 
        self.lr = lr
        self.ds = None

        config.args = Parameters({ 
            'steps': steps,
            'momentum': 0.0, 
            'weight_decay': 0.0,   
            'num_samples': num_samples,
            'lr': lr,  
            'eps': eps,
        })

        # set attribute for learning rate scheduler
        if custom_lr_multiplier: 
            config.args.__setattr__('custom_lr_multiplier', custom_lr_multiplier)
        else: 
            config.args.__setattr__('step_lr', step_lr)
            config.args.__setattr__('step_lr_gamma', step_lr_gamma)

        # create instance variables for empirical estimates
        self.emp_loc, self.emp_covariance_matrix = None, None

    def fit(self, S: Tensor):
        """
        """
        # separate into training and validation set
        rand_indices = ch.randperm(S.size(0))
        train_indices, val_indices = rand_indices[self.val:], rand_indices[:self.val]
        self.X_train = S[train_indices]
        self.X_val = S[val_indices]

        self.train_ds = CensoredNormalDataset(self.X_train)
        self.val_ds = CensoredNormalDataset(self.X_val)
        train_loader = DataLoader(self.train_ds, batch_size=self.bs, num_workers=self.workers)
        val_loader = DataLoader(self.val_ds, batch_size=len(self.val_ds), num_workers=self.workers)

        self.censored_normal = CensoredNormalModel(config.args, self.train_ds, self.val_ds, self.phi, self.tol, self.r, self.alpha, self.clamp, n=self.n)
        # run PGD to predict actual estimates
        self.trainer = Trainer(self.censored_normal)

        # run PGD for parameter estimation 
        self.trainer.train_model((train_loader, None))


class CensoredNormalModel(delphi.delphi):
    '''
    Model for censored normal distributions to be passed into trainer.
    '''
    def __init__(self, args,  X_train, X_val, phi, tol, r, alpha, clamp, n=100, store=None, table=None, schema=None): 
        '''
        Args: 
            args (cox.utils.Parameters) : parameter object holding hyperparameters
        '''
        super().__init__(args, store=store, table=table, schema=schema)
        self.r = r 
        self.alpha = alpha 
        self.phi = phi 

        # initialize projection set 
        self.clamp = clamp 
        self.emp_loc = X_train._loc
        self.emp_covariance_matrix = X_train._covariance_matrix

        self.radius = self.r * (ch.log(1.0 / self.alpha)/ch.square(self.alpha))
        # parameterize projection set
        if self.clamp:
            self.loc_bounds, self.scale_bounds = Bounds(self.emp_loc-self.radius, self.emp_loc+self.radius), \
             Bounds(ch.max(ch.square(self.alpha / 12.0), self.emp_covariance_matrix - self.radius), self.emp_covariance_matrix + self.radius)
        else:
            pass

        # validation set
        # use steps counter to keep track of steps taken
        self.n, self.steps = n, 0
        self.X_val  = X_val
        self.X_train  = X_train 
        self.tol = tol
        # track best estimates based off of gradient norm
        self.best_grad_norm = None
        self.best_state_dict = None
        self.best_opt = None

        # establish empirical distribution
        self.model = MultivariateNormal(self.emp_loc.clone(), self.emp_covariance_matrix.clone())
        self.model.loc.requires_grad, self.model.covariance_matrix.requires_grad = True, True
        self.params = [self.model.loc, self.model.covariance_matrix]

    def check_grad(self): 
        """
        Calculates the check_grad of the current regression estimates of the validation set. It 
        then updates the best estimates accordingly based off of the check_grad's norm.
        """
        loss = CensoredMultivariateNormalNLL.apply(self.model.loc, self.model.covariance_matrix, self.X_val.S, self.phi)
        loc_grad, cov_grad = ch.autograd.grad(loss, [self.model.loc, self.model.covariance_matrix])
        grad = ch.cat([loc_grad, cov_grad.flatten()])

        grad_norm = grad.norm(dim=-1)
        # check that gradient magnitude is less than tolerance
        if self.steps != 0 and grad_norm < self.tol: 
            print("Final Gradient Estimate: {}".format(grad_norm))
            raise ProcedureComplete()
        
        print("{} Steps | Gradient Estimate: {}".format(self.steps, grad_norm))
        # if smaller gradient norm, update best
        if self.best_grad_norm is None or grad_norm < self.best_grad_norm: 
            self.best_grad_norm = grad_norm
            # keep track of state dict
            self.best_loc = copy.deepcopy(self.model.loc)
            self.best_covariance_matrix = copy.deepcopy(self.model.covariance_matrix)
            self.best_opt = copy.deepcopy(self.optimizer.state_dict())
        elif 1e-1 <= grad_norm - self.best_grad_norm: 
            # load in the best model state and optimizer dictionaries
            self.model.loc = self.best_loc
            self.model.covariance_matrix = self.best_covariance_matrix
            self.optimizer.load_state_dict(self.best_opt)

    def train_step(self, i, batch):
        '''
        Training step for defined model.
        Args: 
            i (int) : gradient step or epoch number
            batch (Iterable) : iterable of inputs that 
        '''
        self.optimizer.zero_grad()
        loss = CensoredMultivariateNormalNLL.apply(self.model.loc, self.model.covariance_matrix, batch[0], self.phi)
        loss.backward()
        self.optimizer.step()
        self.schedule.step()

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
        # increase number of steps taken
        self.steps += 1
        
        if self.clamp:
            self.model.loc.data = ch.clamp(self.model.loc.data, float(self.loc_bounds.lower), float(self.loc_bounds.upper))
            self.model.covariance_matrix.data = ch.clamp(self.model.covariance_matrix.data, float(self.scale_bounds.lower), float(self.scale_bounds.upper))
        else:
            pass

        # check for convergence every n steps
        if self.steps % self.n == 0: 
            grad = self.check_grad()


