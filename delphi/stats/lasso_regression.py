"""
Truncated Lasso Regression.
"""

import torch as ch
import torch.linalg as LA
from torch import Tensor
import torch.nn as nn
from torch.nn import Linear
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import LassoCV
import math
import cox
from cox.store import Store
import copy
import warnings
from abc import abstractmethod

from .. import delphi
from .stats import stats
from ..oracle import oracle
from ..trainer import Trainer
from .linear_regression import TruncatedLinearRegression, KnownVariance, UnknownVariance
from ..utils.helpers import Parameters, Bounds, make_train_and_val, check_and_fill_args


# CONSTANTS 
DEFAULTS = {
        'epochs': (int, 1),
        'noise_var': (float, None), 
        'fit_intercept': (bool, True), 
        'num_trials': (int, 3),
        'clamp': (bool, True), 
        'val': (float, .2),
        'lr': (float, 1e-1), 
        'var_lr': (float, 1e-2), 
        'step_lr': (int, 100),
        'step_lr_gamma': (float, .9), 
        'custom_lr_multiplier': (str, None), 
        'momentum': (float, 0.0), 
        'weight_decay': (float, 0.0), 
        'l1': (float, 0.0), 
        'eps': (float, 1e-5),
        'r': (float, 1.0), 
        'rate': (float, 1.5), 
        'normalize': (bool, True), 
        'batch_size': (int, 10),
        'tol': (float, 1e-1),
        'workers': (int, 0),
        'num_samples': (int, 10),
}


class TruncatedLassoRegression(TruncatedLinearRegression):
    '''
    Truncated LASSO regression class. Supports truncated LASSO regression
    with known noise, unknown noise, and confidence intervals. Module uses 
    delphi.trainer.Trainer to train truncated linear regression by performing 
    projected stochastic gradient descent on the truncated population log likelihood. 
    Module requires the user to specify an oracle from the delphi.oracle.oracle class, 
    and the survival probability. 
    '''
    def __init__(self,
            phi: oracle,
            alpha: float,
            kwargs: dict={}):
        '''
        Args: 
            phi (delphi.oracle.oracle) : oracle object for truncated regression model 
            alpha (float) : survival probability for truncated regression model
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
            l1 (float) : weight decay for SGD optimizer 
            eps (float) :  epsilon value for gradient to prevent zero in denominator
            store (cox.store.Store) : cox store object for logging 
        '''
        super().__init__(phi, alpha, kwargs)
        
    def fit(self, X: Tensor, y: Tensor):
        """
        """
        self.train_loader_, self.val_loader_ = make_train_and_val(self.args, X, y) 

        if self.args.noise_var is None:
            self.trunc_reg = LassoUnknownVariance(self.args, self.train_loader_, self.phi) 
        else: 
            self.trunc_reg = LassoKnownVariance(self.args, self.train_loader_, self.phi) 
        
        # run PGD for parameter estimation
        trainer = Trainer(self.trunc_reg, self.args.epochs, self.args.num_trials, self.args.tol)
        trainer.train_model((self.train_loader_, self.val_loader_))

        with ch.no_grad():
            # assign results from procedure to instance variables
            self.coef = self.trunc_reg.model.weight.clone()
            if self.args.fit_intercept: self.intercept = self.trunc_reg.model.bias.clone()
            if self.args.noise_var is None: 
                self.variance = self.trunc_reg.scale.clone().inverse()
                self.coef *= self.variance
                if self.args.fit_intercept: self.intercept *= self.variance.flatten()

    def predict(self, x: Tensor): 
        """
        Make predictions with regression estimates.
        """
        with ch.no_grad():
            return self.trunc_reg.model(x)


class LassoKnownVariance(KnownVariance):
    '''
    Truncated linear regression with known noise variance model.
    '''
    def __init__(self, args, train_loader, phi): 
        '''
        Args: 
            args (cox.utils.Parameters) : parameter object holding hyperparameters
        '''
        super().__init__(args, train_loader, phi)
        
    def calc_emp_model(self):
        # calculate empirical estimates
        self.emp_model = LassoCV(fit_intercept=self.args.fit_intercept, alphas=[self.args.l1]).fit(self.X, self.y.flatten())
        self.emp_weight = Tensor(self.emp_model.coef_)[None,...]
        if self.args.fit_intercept:
            self.emp_bias = Tensor([self.emp_model.intercept_])
        self.emp_var = ch.var(Tensor(self.emp_model.predict(self.X))[...,None] - self.y, dim=0)[..., None]

    def pretrain_hook(self):
        # use OLS as empirical estimate to define projection set
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

        # assign empirical estimates
        self.model.weight.data = self.emp_weight
        if self.args.fit_intercept:
            self.model.bias.data = self.emp_bias
        self.params = None


class LassoUnknownVariance(UnknownVariance):
    '''
    Parent/abstract class for models to be passed into trainer.  
    '''
    def __init__(self, args, train_loader, phi): 
        '''
        Args: 
            args (cox.utils.Parameters) : parameter object holding hyperparameters
        '''
        super().__init__(args, train_loader, phi)

    def calc_emp_model(self):
        # calculate empirical estimates
        self.emp_model = LassoCV(fit_intercept=self.args.fit_intercept, alphas=[self.args.l1]).fit(self.X, self.y.flatten())
        self.emp_weight = Tensor(self.emp_model.coef_)[None,...]
        if self.args.fit_intercept:
            self.emp_bias = Tensor([self.emp_model.intercept_])
        self.emp_var = ch.var(Tensor(self.emp_model.predict(self.X))[...,None] - self.y, dim=0)[..., None]

    def pretrain_hook(self):
        # use OLS as empirical estimate to define projection set
        self.radius = self.args.r * (12.0 + 4.0 * ch.log(2.0 / self.args.alpha)) if self.args.noise_var is None else self.args.r * (7.0 + 4.0 * ch.log(2.0 / self.args.alpha))

        if self.args.clamp:
            self.weight_bounds = Bounds(self.emp_weight.flatten() - self.radius,
                                        self.emp_weight.flatten() + self.radius)
            # generate noise variance radius bounds if unknown 
            self.var_bounds = Bounds(float(self.emp_var.flatten() / self.args.r), float(self.emp_var.flatten() / self.args.alpha.pow(2))) 
            if self.args.fit_intercept:
                self.bias_bounds = Bounds(float(self.emp_bias.flatten() - self.radius),
                                      float(self.emp_bias.flatten() + self.radius))
        else:
            pass

        # assign empirical estimates
        self.scale.data = self.emp_var.inverse()
        self.model.weight.data = self.emp_weight * self.scale
        if self.args.fit_intercept:
            self.model.bias.data = (self.emp_bias * self.scale).flatten()
        self.params = [{'params': [self.model.weight, self.model.bias]},
            {'params': self.scale, 'lr': self.args.var_lr}]
        

