"""
Truncated normal distribution without oracle access (ie. unknown truncation set)
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
from ..oracle import oracle, UnknownGaussian
from ..trainer import Trainer
from ..utils.helpers import Bounds
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
        super(TruncatedNormal).__init__()
        # instance variables
        self.d = d
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
        :param S:
        :return:
        """
        # separate into training and validation set
        rand_indices = ch.randperm(S.size(0))
        train_indices, val_indices = rand_indices[self.val:], rand_indices[:self.val]
        self.X_train = S[train_indices]
        self.X_val = S[val_indices]

        self.train_ds = TruncatedNormalDataset(self.X_train)
        self.val_ds = TruncatedNormalDataset(self.X_val)
        train_loader = DataLoader(self.train_ds, batch_size=self.bs, num_workers=self.workers)
        val_loader = DataLoader(self.val_ds, batch_size=len(self.val_ds), num_workers=self.workers)

        self.truncated_normal = TruncatedNormalModel(config.args, self.d, self.train_ds, self.val_ds, self.tol, self.r, self.alpha, self.clamp, n=self.n)
        # run PGD to predict actual estimates
        self.trainer = Trainer(self.truncated_normal)

        # run PGD for parameter estimation 
        self.trainer.train_model((train_loader, None))

        
class TruncatedNormalModel(delphi.delphi):
    '''
    Model for truncated normal distributions to be passed into trainer.
    '''
    def __init__(self, args, d,  X_train, X_val, tol, r, alpha, clamp, n=100, store=None, table=None, schema=None): 
        '''
        Args: 
            args (cox.utils.Parameters) : parameter object holding hyperparameters
        '''
        super().__init__(args, store=store, table=table, schema=schema)
        self.r = r 
        self.alpha = alpha 
        
        # initialize projection set 
        self.clamp = clamp 
        self.emp_loc = X_train._loc
        self.emp_covariance_matrix = X_train._covariance_matrix

        self.radius = r * ch.sqrt(ch.log(1.0 / self.alpha))

        # upper and lower bounds
        if self.clamp:
            self.loc_bounds, self.scale_bounds = Bounds(self.emp_loc - self.radius, self.emp_loc + self.radius), Bounds(ch.max(self.alpha.pow(2) / 12, \
                                                               self.emp_covariance_matrix - self.radius),
                                                        self.emp_covariance_matrix + self.radius)
        else:
            pass


        # initialiaze pseudo oracle for gaussians with unknown truncation 
        self.phi = UnknownGaussian(self.emp_loc, self.emp_covariance_matrix, X_train.S, d)

        # exponent class
        self.exp_h = Exp_h(self.emp_loc, self.emp_covariance_matrix)

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
        loss = TruncatedMultivariateNormalNLL.apply(self.model.loc, self.model.covariance_matrix, self.X_val.S, self.X_val.loc_grad, self.X_val.cov_grad, self.phi, self.exp_h)
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

    def pretrain_hook(self):
        '''
        Assign OLS estimates as original empirical estimates 
        for PGD procedure.
        '''
        pass 

    def train_step(self, i, batch):
        '''
        Training step for defined model.
        Args: 
            i (int) : gradient step or epoch number
            batch (Iterable) : iterable of inputs that 
        '''
        self.optimizer.zero_grad()
        loss = TruncatedMultivariateNormalNLL.apply(self.model.loc, self.model.covariance_matrix, *batch, self.phi, self.exp_h)
        loss.backward()
        self.optimizer.step()
        return loss, None, None

    def val_step(self, i, batch):
        '''
        Valdation step for defined model. 
        '''
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
            self.model.loc.data = ch.clamp(self.model.loc.data, float(self.loc_bounds.lower), float(self.loc_bounds.upper))
            self.model.covariance_matrix.data = ch.clamp(self.model.covariance_matrix.data, float(self.scale_bounds.lower), float(self.scale_bounds.upper))
        else:
            pass

        # check for convergence every n steps
        if self.steps % self.n == 0: 
            grad = self.check_grad()

    def epoch_hook(self, i, loop_type, loss, prec1, prec5, batch):
        '''
        Epoch hook for defined model. Method is called after each 
        complete iteration through dataset.
        '''
        pass 

    def post_train_hook(self):
        '''
        Post training hook, called after sgd procedures completes. 
        '''
        pass
 

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
        cov_term = ch.bmm(x.unsqueeze(1).matmul(B), x.unsqueeze(2)).flatten(1) / 2.0
        trace_term = ch.trace((B - ch.eye(u.size(0))) * (self.emp_cov + self.emp_loc.matmul(self.emp_loc))).unsqueeze(0)
        loc_term = (x - self.emp_loc).matmul(u.unsqueeze(1))
        return ch.exp(cov_term - trace_term - loc_term + self.pi_const)
