"""
Truncated Linear Regression.
"""

import torch as ch
import torch.linalg as LA
from torch import Tensor
import torch.nn as nn
from torch.nn import Linear
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import LinearRegression
import math
import cox
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
from ..utils.helpers import Parameters, Bounds, LinearUnknownVariance, setup_store_with_metadata, ProcedureComplete


class TruncatedLinearRegression(stats):
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
            noise_var: float=None,
            fit_intercept: bool=True,
            normalize: bool=True,
            epochs: int=1,
            num_trials: int=3,
            clamp: bool=True,
            val: int=50,
            tol: float=1e-2,
            r: float=2.0,
            rate: float=.5, 
            num_samples: int=10,
            bs: int=10,
            lr: float=1e-1,
            var_lr: float=1e-1, 
            step_lr: int=100, 
            custom_lr_multiplier: str=None,
            lr_interpolation: str=None,
            step_lr_gamma: float=.9,
            momentum: float=0.0, 
            weight_decay: float=0.0,
            eps: float=1e-5,
            store: cox.store.Store=None, 
            **kwargs):
        '''
        Args: 
            phi (delphi.oracle.oracle) : oracle object for truncated regression model 
            alpha (float) : survival probability for truncated regression model
            unknown (bool) : boolean indicating whether the noise variance is known or not - specifying algorithm to use 
            normalize (bool) : normalize the input features before running procedure
            fit_intercept (bool) : boolean indicating whether to fit a intercept or not 
            steps (int) : number of gradient steps to take
            clamp (bool) : boolean indicating whether to clamp the projection set 
            n (int) : number of gradient steps to take before checking gradient 
            val (int) : number of samples to use for validation set 
            tol (float) : gradient tolerance threshold 
            workers (int) : number of workers to spawn 
            r (float) : size for projection set radius 
            rate (float): rate at which to increase the size of the projection set, when procedure does not converge - input as a decimal percentage
            num_samples (int) : number of samples to sample in gradient 
            bs (int) : batch size
            lr (float) : initial learning rate for regression parameters 
            var_lr (float) : initial learning rate to use for variance parameter in the settign where the variance is unknown 
            step_lr (int) : number of gradient steps to take before decaying learning rate for step learning rate 
            custom_lr_multiplier (str) : 'cosine' (cosine annealing), 'adam' (adam optimizer) - different learning rate schedulers available
            lr_interpolation (str) : 'linear' linear interpolation
            step_lr_gamma (float) : amount to decay learning rate when running step learning rate
            momentum (float) : momentum for SGD optimizer 
            weight_decay (float) : weight decay for SGD optimizer 
            eps (float) :  epsilon value for gradient to prevent zero in denominator
            store (cox.store.Store) : cox store object for logging 
            
        '''
        super(TruncatedLinearRegression).__init__()
        # instance variables
        self.phi = phi 
        self.model = None
        self.store = store 

        self.args = Parameters({
            'alpha': alpha,
            'num_trials': num_trials,
            'bs': bs, 
            'workers': 0,
            'epochs': epochs,
            'momentum': momentum, 
            'weight_decay': weight_decay,   
            'num_samples': num_samples,
            'lr': lr,  
            'var_lr': var_lr,
            'eps': eps,
            'tol': tol,
            'val': val,
            'clamp': clamp,
            'fit_intercept': fit_intercept,
            'r': r,
            'rate': rate,
            'noise_var': noise_var,
            'normalize': normalize,
            'verbose': False,
        })

        self.custom_lr_multiplier = custom_lr_multiplier
        self.step_lr = step_lr 
        self.step_lr_gamma = step_lr_gamma
        self.lr_interpolation = lr_interpolation

    def fit(self, X: Tensor, y: Tensor):
        """
        """
        # separate into training and validation set
        rand_indices = ch.randperm(X.size(0))
        train_indices, val_indices = rand_indices[self.args.val:], rand_indices[:self.args.val]
        self.X_train, self.y_train = X[train_indices], y[train_indices]
        self.X_val, self.y_val = X[val_indices], y[val_indices]

        if self.args.normalize: 
            # normalize x features so that ||x_{i}||_{2} <= 1
            l_inf = LA.norm(self.X_train, dim=-1, ord=float('inf')).max() # find max l_inf
            # calculate normalizing constant
            self.beta = l_inf * math.sqrt(self.X_train.size(1))

            self.X_val /= self.beta
            self.X_train /= self.beta
        self.train_ds = TensorDataset(self.X_train, self.y_train)
        self.train_loader_ = DataLoader(self.train_ds, batch_size=self.args.bs, num_workers=self.args.workers)
        self.val_ds = TensorDataset(self.X_val, self.y_val)
        self.val_loader_ = DataLoader(self.val_ds, batch_size=len(self.val_ds), num_workers=self.args.workers)

        self.trunc_reg = TruncatedLinearRegressionModel(self.args, self.X_train, self.y_train, self.phi, self.store, self.custom_lr_multiplier, self.lr_interpolation, self.step_lr, self.step_lr_gamma)

        # run PGD for parameter estimation
        trainer = Trainer(self.trunc_reg)
        trainer.train_model((self.train_loader_, self.val_loader_))

        with ch.no_grad():
            # renormalize weights and biases
            if self.args.normalize:
                # unnormalize coefficients
                # self.trunc_reg.best_model.weight /= self.beta
                pass
            # assign results from procedure to instance variables
            if self.args.noise_var is None: 
                self.variance = self.trunc_reg.best_model.lambda_.clone()
            self.coef = self.trunc_reg.best_model.weight.clone()
            if self.args.fit_intercept: self.intercept = self.trunc_reg.best_model.bias.clone()

    def __call__(self, x: Tensor): 
        """
        Make predictions with regression estimates.
        """
        with ch.no_grad():
            return self.trunc_reg.model(x)

    @property
    def coef_(self): 
        """
        Regression weight.
        """
        return self.coef

    @property
    def intercept_(self): 
        """
        Regression intercept.
        """
        return self.intercept

    @property
    def variance_(self): 
        """
        Noise variance prediction for linear regression with
        unknown noise variance algorithm.
        """
        if self.args.noise_var is None: 
            return self.variance
        else: 
            warnings.warn("no variance prediction because regression with known variance was run")

    @property 
    def beta_(self): 
        """
        Beta normalizing constant calculated on the training set.  
        """
        return self.trunc_reg.beta

    @property
    def nll_(self): 
        """
        Gradient of the negative log likelihood on validation set 
        for model's current estimates.
        """
        return self.trunc_reg.best_grad_norm.clone()
    
    @property
    def train_loader(self): 
        '''
        Get the train loader for model.
        '''
        return self.train_loader_

    @property
    def val_loader(self): 
        '''
        Get the train loader for model.
        '''
        return self.val_loader_


class TruncatedLinearRegressionModel(delphi.delphi):
    '''
    Parent/abstract class for models to be passed into trainer.  
    '''
    def __init__(self, args, X_train, y_train, phi, store, custom_lr_multiplier, lr_interpolation, step_lr, step_lr_gamma): 
        '''
        Args: 
            args (cox.utils.Parameters) : parameter object holding hyperparameters
        '''
        super().__init__(args, custom_lr_multiplier, lr_interpolation, step_lr, step_lr_gamma, store=store)
        self.X_train, self.y_train = X_train, y_train
        self.trials = 0
        self.phi = phi
        # track best estimates based off of likelihood
        self.best_nll, self.best_model = None, None

    def pretrain_hook(self):
        # use OLS as empirical estimate to define projection set
        self.emp_model = LinearRegression(fit_intercept=self.args.fit_intercept).fit(self.X_train, self.y_train)
        self.emp_weight = Tensor(self.emp_model.coef_)
        if self.args.fit_intercept:
            self.emp_bias = Tensor(self.emp_model.intercept_)
        self.emp_var = ch.var(Tensor(self.emp_model.predict(self.X_train)) - self.y_train, dim=0)[..., None]
        self.radius = self.args.r * (12.0 + 4.0 * ch.log(2.0 / self.args.alpha)) if self.args.noise_var is None else self.args.r * (7.0 + 4.0 * ch.log(2.0 / self.args.alpha))

        if self.args.clamp:
            self.weight_bounds = Bounds(self.emp_weight.flatten() - self.radius,
                                        self.emp_weight.flatten() + self.radius)
            # generate noise variance radius bounds if unknown 
            self.var_bounds = Bounds(float(self.emp_var.flatten() / self.args.r), float(self.emp_var.flatten() / self.args.alpha.pow(2))) if self.args.noise_var is None else None
            if self.args.fit_intercept:
                self.bias_bounds = Bounds(float(self.emp_bias.flatten() - self.radius),
                                      float(self.emp_bias.flatten() + self.radius))
        else:
            pass

        if self.args.noise_var is None: # unknown variance
            self.model = LinearUnknownVariance(in_features=self.X_train.size(1), out_features=1, bias=self.args.fit_intercept)
            # assign empirical estimates
            self.model.lambda_.data = self.emp_var.inverse()
            self.model.weight.data = self.emp_weight * self.model.lambda_
            if self.args.fit_intercept:
                self.model.bias.data = (self.emp_bias * self.model.lambda_).flatten()
            self.params = [{'params': [self.model.weight, self.model.bias]},
                {'params': self.model.lambda_, 'lr': self.args.var_lr}]
        else:  # unknown variance
            self.model = Linear(in_features=self.X_train.size(1), out_features=1, bias=self.args.fit_intercept)
            # assign empirical estimates
            self.model.weight.data = self.emp_weight
            if self.args.fit_intercept:
                self.model.bias.data = self.emp_bias
            self.params = None

    def calc_nll(self, X_val, y_val): 
        """
        Calculates the negative log likelihood of the current regression estimates of the validation set.
        Args: 
            proc (bool) : boolean indicating whether, the function is being called within 
            a stochastic process, or someone is accessing the parent class's property
        """
        pred = self.model(X_val)
        if self.args.noise_var is None:
            loss = TruncatedUnknownVarianceMSE.apply(pred, y_val, self.model.lambda_, self.phi, self.args.num_samples, self.args.eps)
        else: 
            loss = TruncatedMSE.apply(pred,y_val, self.phi, self.args.noise_var, self.args.num_samples, self.args.eps)
        return loss
        
    def calc_grad(self, X, y): 
        '''
        Calculates the gradient of the validation set.
        '''
        pred = self.model(X)
        if self.args.noise_var is None:
            loss = TruncatedUnknownVarianceMSE.apply(pred, y, self.model.lambda_, self.phi, self.args.num_samples, self.args.eps)
            grad, lambda_grad = ch.autograd.grad(loss, [pred, self.model.lambda_])
            grad = ch.cat([(grad.sum(0) / self.model.lambda_).flatten(), lambda_grad.flatten()])
        else: 
            loss = TruncatedMSE.apply(pred, y, self.phi, self.args.noise_var, self.args.num_samples, self.args.eps)
            grad, = ch.autograd.grad(loss, [pred])
            grad = grad.sum(0)
        return grad

    def train_step(self, i, batch):
        '''
        Training step for defined model.
        Args: 
            i (int) : gradient step or epoch number
            batch (Iterable) : iterable of inputs that 
        '''
        inp, targ = batch

        pred = self.model(inp)
        if self.args.noise_var is None: 
            loss = TruncatedUnknownVarianceMSE.apply(pred, targ, self.model.lambda_, self.phi, self.args.num_samples, self.args.eps)
        else: 
            loss = TruncatedMSE.apply(pred, targ, self.phi, self.args.noise_var, self.args.num_samples, self.args.eps)

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
        # project model parameters back to domain 
        if self.args.clamp: 
            if self.args.noise_var is None: 
                var = self.model.lambda_.inverse()
                weight = self.model.weight * var
                self.model.lambda_.data = ch.clamp(var, self.var_bounds.lower, self.var_bounds.upper).inverse()
                # project weights
                self.model.weight.data = ch.cat([ch.clamp(weight[:,i], self.weight_bounds.lower[i], self.weight_bounds.upper[i])
                    for i in range(weight.size(1))])[None,...] * self.model.lambda_
                if self.args.fit_intercept: 
                    # project bias
                    bias = self.model.bias * var
                    self.model.bias.data = (ch.clamp(bias, self.bias_bounds.lower, self.bias_bounds.upper) * self.model.lambda_).reshape(self.model.bias.size())
            else: 
                self.model.weight.data = ch.cat([ch.clamp(self.model.weight[:,i], self.weight_bounds.lower[i], self.weight_bounds.upper[i]) 
                    for i in range(self.model.weight.size(1))])[None,...]
                if self.args.fit_intercept:
                    # project bias
                    self.model.bias.data = ch.clamp(self.model.bias, self.bias_bounds.lower, self.bias_bounds.upper).reshape(self.model.bias.size())
        else: 
            pass

    def val_step(self, i, batch):
        # check for convergence every at each epoch
        loss = self.calc_nll(*batch)
        print("Epoch {} | Log Likelihood: {}".format(i, round(float(abs(loss)), 3)))
        return loss, None, None

    def post_training_hook(self, val_loader): 
        self.trials += 1
        # check gradient
        X, y = val_loader.dataset.tensors[0], val_loader.dataset.tensors[1]
        grad = self.calc_grad(X, y)
        loss = self.calc_nll(X, y)
        
        if self.best_nll is None or loss < self.best_nll:
            self.best_nll = loss 
            self.best_model = copy.copy(self.model)

        # terminate procedure
        converge = (grad.abs() < self.args.tol).all()
        if converge or self.trials == self.args.num_trials: 
            # assign results from procedure to instance variables
            if self.args.noise_var is None: 
                with ch.no_grad():
                    self.best_model.lambda_ = ch.nn.Parameter(self.best_model.lambda_.inverse())
                    self.best_model.weight =  ch.nn.Parameter(self.best_model.weight * self.best_model.lambda_)
                    if self.args.fit_intercept: self.best_model.bias = ch.nn.Parameter(self.best_model.bias * self.model.lambda_)
            if not converge: warnings.warn("Procedure did not converge, increase batch size of number of epochs.")
            return True

        self.args.r *= self.args.rate
        return False

    
