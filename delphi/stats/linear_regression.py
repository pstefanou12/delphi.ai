"""
Truncated Linear Regression.
"""

import torch as ch
from torch import Tensor
import torch.nn as nn
from torch.nn import Linear
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import LinearRegression
from cox.utils import Parameters
from cox.store import Store
import config
import copy
import warnings
from abc import abstractmethod

from .. import delphi
from .stats import stats
from ..oracle import oracle
from ..trainer import Trainer
from ..grad import TruncatedMSE, TruncatedUnknownVarianceMSE
from ..utils import constants as consts
from ..utils.helpers import Bounds, LinearUnknownVariance, setup_store_with_metadata, ProcedureComplete


class TruncatedRegression(stats):
    '''
    Truncated linear regression class. Supports truncated linear regression
    with known noise, unknown noise, and confidence intervals. Module uses 
    delphi.trainer.Trainer to train truncated linear regression by performing 
    projected stochastic gradient descent on the truncated population log likelihood. 
    Module requires the user to specify an oracle from the delphi.oracle.oracle class, 
    and the survival probability. 
    '''
    def __init__(
            self,
            phi: oracle,
            alpha: float,
            steps: int=1000,
            unknown: bool=True,
            clamp: bool=True,
            n: int=10, 
            val: int=50,
            tol: float=1e-2,
            workers: int=0,
            r: float=2.0,
            num_samples: int=100,
            bs: int=10,
            lr: float=1e-1,
            var_lr: float=1e-1, 
            step_lr: int=100, 
            custom_lr_multiplier: str=None,
            step_lr_gamma: float=.9,
            eps: float=1e-5, 
            **kwargs):
        '''
        Args: 
            phi (delphi.oracle.oracle) : `
        '''
        super(TruncatedRegression).__init__()
        # instance variables
        self.phi = phi 
        self.alpha = alpha 
        self.unknown = unknown 
        self.model = None
        self.clamp = clamp
        self.iter_hook = None
        self.n = n 
        self.val = val
        self.tol = tol
        self.workers = workers
        self.r = r 
        self.num_samples = num_samples
        self.bs = bs
        self.lr = lr 
        self.ds = None

        config.args = Parameters({ 
            'steps': steps,
            'momentum': 0.0, 
            'weight_decay': 0.0,   
            'num_samples': num_samples,
            'lr': lr,  
            'var_lr': var_lr,
            'eps': eps,
        })

        # set attribute for learning rate scheduler
        if custom_lr_multiplier: 
            config.args.__setattr__('custom_lr_multiplier', custom_lr_multiplier)
        else: 
            config.args.__setattr__('step_lr', step_lr)
            config.args.__setattr__('step_lr_gamma', step_lr_gamma)

    def fit(self, X: Tensor, y: Tensor):
        """
        """
        # separate into training and validation set
        rand_indices = ch.randperm(X.size(0))
        train_indices, val_indices = rand_indices[self.val:], rand_indices[:self.val]
        self.X_train, self.y_train = X[train_indices], y[train_indices]
        self.X_val, self.y_val = X[val_indices], y[val_indices]

        self.ds = TensorDataset(self.X_train, self.y_train)
        loader = DataLoader(self.ds, batch_size=self.bs, num_workers=self.workers)
        
        self.trunc_reg = TruncatedRegressionModel(config.args, self.X_train, self.y_train, self.X_val, self.y_val, self.phi, self.tol, self.r, self.alpha, self.clamp, self.unknown)

        trainer = Trainer(self.trunc_reg)

        # run PGD for parameter estimation
        trainer.train_model((loader, None))

    def __call__(self, x: Tensor): 
        """
        """
        return self.trunc_reg.model(x)

    @property
    def weight(self): 
        """
        Regression weight.
        """
        if self.unknown: 
            return self.trunc_reg.model.weight.detach().clone().T * self.trunc_reg.model.lambda_.detach().inverse().clone()
        return self.trunc_reg.model.weight.detach().clone().T

    @property
    def intercept(self): 
        """
        Regression intercept.
        """
        if self.unknown: 
            return self.trunc_reg.model.bias.detach().clone().T * self.trunc_reg.model.lambda_.detach().inverse().clone()   
        return self.trunc_reg.model.bias.detach().clone()

    @property
    def variance(self): 
        """
        Noise variance prediction for linear regression with
        unknown noise variance algorithm.
        """
        if self.unknown: 
            return self.trunc_reg.model.lambda_.detach().inverse().clone()
        else: 
            warnings.warn("no variance prediction because regression with known variance was run")


class TruncatedRegressionModel(delphi.delphi):
    '''
    Parent/abstract class for models to be passed into trainer.  
    '''
    def __init__(self, args,  X_train, y_train, X_val, y_val, phi, tol, r, alpha, clamp, unknown, n=100, store=None, table=None, schema=None): 
        '''
        Args: 
            args (cox.utils.Parameters) : parameter object holding hyperparameters
        '''
        super().__init__(args, store=store, table=table, schema=schema)
        self.unknown = unknown
        # use OLS as empirical estimate to define projection set
        self.r = r
        self.alpha = alpha
        self.unknown = unknown
        self.phi = phi

        # initialize projection set
        self.clamp = clamp
        self.emp_model = LinearRegression().fit(X_train, y_train)
        self.emp_weight = Tensor(self.emp_model.coef_) 
        self.emp_bias = Tensor(self.emp_model.intercept_)
        self.emp_var = ch.var(Tensor(self.emp_model.predict(X_train)) - y_train, dim=0)[..., None]
        self.radius = self.r * (12.0 + 4.0 * ch.log(2.0 / self.alpha)) if self.unknown else self.r * (4.0 * ch.log(2.0 / self.alpha) + 7.0)

        if self.clamp:
            self.weight_bounds = Bounds(self.emp_weight.flatten() - self.radius,
                                        self.emp_weight.flatten() + self.radius)
            # generate noise variance radius bounds if unknown 
            self.var_bounds = Bounds(float(self.emp_var.flatten() / self.r), float(self.emp_var.flatten() / self.alpha.pow(2))) if self.unknown else None
            self.bias_bounds = Bounds(float(self.emp_bias.flatten() - self.radius),
                                      float(self.emp_bias.flatten() + self.radius))
        else:
            pass

        # validation set
        # use steps counter to keep track of steps taken
        self.n, self.steps = n, 0
        self.X_val, self.y_val = X_val, y_val
        self.X_train, self.y_train = X_train, y_train
        self.tol = tol
        # track best estimates based off of gradient norm
        self.best_grad_norm = None
        self.best_state_dict = None
        self.best_opt = None
        
        if self.unknown: # unknown variance
            self.model = LinearUnknownVariance(in_features=self.X_train.size(1), out_features=1, bias=True)
            # assign empirical estimates
            self.model.lambda_.data = self.emp_var.inverse()
            self.model.weight.data = self.emp_weight * self.model.lambda_ 
            self.model.bias.data = (self.emp_bias * self.model.lambda_).flatten()
            update_params = [{'params': [self.model.weight, self.model.bias]},
                {'params': self.model.lambda_, 'lr': args.var_lr}]
        else:  # unknown variance
            self.model = Linear(in_features=self.X_train.size(1), out_features=1, bias=True)
            # assign empirical estimates
            self.model.weight.data = self.emp_weight
            self.model.bias.data = self.emp_bias
            update_params = None

    def check_grad(self): 
        """
        Calculates the check_grad of the current regression estimates of the validation set. It 
        then updates the best estimates accordingly based off of the check_grad's norm.
        """
        pred = self.model(self.X_val)
        if self.unknown:
            loss = TruncatedUnknownVarianceMSE.apply(pred, self.y_val, self.model.lambda_, self.phi)
            grad, lambda_grad = ch.autograd.grad(loss, [pred, self.model.lambda_])
            grad = ch.cat([(grad.sum(0) / self.model.lambda_).flatten(), lambda_grad.flatten()])
        else: 
            loss = TruncatedMSE.apply(pred, self.y_val, self.phi)
            grad, = ch.autograd.grad(loss, [pred])
            grad = grad.sum(0)

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
            self.best_state_dict, self.best_opt = copy.deepcopy(self.model.state_dict()), copy.deepcopy(self.optimizer.state_dict())
        elif 1e-1 <= grad_norm - self.best_grad_norm: 
            # load in the best model state and optimizer dictionaries
            self.model.load_state_dict(self.best_state_dict)
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
        inp, targ = batch

        pred = self.model(inp)
        if self.unknown: 
            loss = TruncatedUnknownVarianceMSE.apply(pred, targ, self.model.lambda_, self.phi)
        else: 
            loss = TruncatedMSE.apply(pred, targ, self.phi)

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
        # project model parameters back to domain 
        if self.clamp: 
            if self.unknown: 
                var = self.model.lambda_.inverse()
                weight = self.model.weight * var

                self.model.lambda_.data = ch.clamp(var, self.var_bounds.lower, self.var_bounds.upper).inverse()
                # project weights
                self.model.weight.data = ch.cat([ch.clamp(weight[:,i], self.weight_bounds.lower[i], self.weight_bounds.upper[i])
                    for i in range(weight.size(1))])[None,...] * self.model.lambda_
                # project bias
                bias = self.model.bias * var
                self.model.bias.data = (ch.clamp(bias, self.bias_bounds.lower, self.bias_bounds.upper) * self.model.lambda_).reshape(self.model.bias.size())
            else: 
                self.model.weight.data = ch.cat([ch.clamp(self.model.weight[:,i], self.weight_bounds.lower[i], self.weight_bounds.upper[i]) 
                    for i in range(self.model.weight.size(1))])[None,...]
                # project bias
                self.model.bias.data = ch.clamp(self.model.bias, self.bias_bounds.lower, self.bias_bounds.upper).reshape(self.model.bias.size())
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
    
