"""
Truncated Lasso Regression.
"""

import torch as ch
from torch import Tensor
import torch.nn as nn
import numpy as np
from sklearn.linear_model import LassoCV
from typing import Callable
import warnings

from .linear_model import LinearModel
from ..trainer import Trainer
from ..utils.datasets import make_train_and_val
from ..grad import TruncatedMSE 
from ..utils.defaults import check_and_fill_args, TRUNC_LASSO_DEFAULTS


class TruncatedLassoRegression(LinearModel):
    """
    Truncated LASSO regression class. Supports truncated LASSO regression
    with known noise, unknown noise, and confidence intervals. Module uses 
    delphi.trainer.Trainer to train truncated linear regression by performing 
    projected stochastic gradient descent on the truncated population log likelihood. 
    Module requires the user to specify an oracle from the delphi.oracle.oracle class, 
    and the survival probability. 
    """
    def __init__(self,
                args: dict, 
                phi: Callable, 
                alpha: float,
                l1: float=1.0, 
                fit_intercept: bool=True, 
                noise_var: float = 1.0, 
                emp_weight: ch.Tensor=None,
                rand_seed: int=0):
        """
        Args: 
            phi (delphi.oracle.oracle) : oracle object for truncated regression model 
            alpha (float) : survival probability for truncated regression model
            fit_intercept (bool) : boolean indicating whether to fit a intercept or not 
            steps (int) : number of gradient steps to take
            val (int) : number of samples to use for validation set 
            tol (float) : gradient tolerance threshold 
            workers (int) : number of workers to spawn 
            r (float) : size for projection set radius 
            rate (float): rate at which to increase the size of the projection set, when procedure does not converge - input as a decimal percentage
            num_samples (int) : number of samples to sample in gradient 
            batch_size (int) : batch size
            lr (float) : initial learning rate for regression parameters 
            var_lr (float) : initial learning rate to use for variance parameter in the settign where the variance is unknown 
            step_lr (int) : number of gradient steps to take before decaying learning rate for step learning rate 
            custom_lr_multiplier (str) : "cosine" (cosine annealing), "adam" (adam optimizer) - different learning rate schedulers available
            lr_interpolation (str) : "linear" linear interpolation
            step_lr_gamma (float) : amount to decay learning rate when running step learning rate
            momentum (float) : momentum for SGD optimizer 
            l1 (float) : weight decay for SGD optimizer 
            eps (float) :  epsilon value for gradient to prevent zero in denominator
            store (cox.store.Store) : cox store object for logging 
        """
        args = check_and_fill_args(args, TRUNC_LASSO_DEFAULTS)
        super().__init__(args, False, emp_weight=emp_weight)
        self.phi = phi
        self.alpha = alpha
        self.l1 = l1
        self.fit_intercept = fit_intercept
        self.noise_var = noise_var
        self.rand_seed = rand_seed

        del self.criterion 
        def self.criterion_params
    
        self.criterion = TruncatedMSE.apply
        self.criterion_params = [
            self.phi, self.noise_var, 
            self.arg.num_samples, self.args.eps
        ]

    def fit(self, 
            X: Tensor, 
            y: Tensor):
        """
        Train truncated lasso regression model by running PSGD on the truncated negative 
        population log likelihood.
        Args: 
            X (torch.Tensor): input feature covariates num_samples by dims
            y (torch.Tensor): dependent variable predictions num_samples by 1
        """
        assert isinstance(X, Tensor), "X is type: {}. expected type torch.Tensor.".format(type(X))
        assert isinstance(y, Tensor), "y is type: {}. expected type torch.Tensor.".format(type(y))
        assert X.size(0) >  X.size(1), "number of dimensions, larger than number of samples. procedure expects matrix with size num samples by num feature dimensions." 
        assert y.dim() == 2 and y.size(1) == 1, "y is size: {}. expecting y tensor with size num_samples by 1.".format(y.size()) 
        # add one feature to x when fitting intercept
        if self.args.fit_intercept:
            X = ch.cat([X, ch.ones(X.size(0), 1)], axis=1)

        if self.fit_intercept: 
            k = X.size(1) - 1
        else: 
            k = X.size(1)

        # Normalization factor: B * √k
        # Compute B = maximum L∞ norm across all samples
        B = X.norm(dim=1, p=float('inf')).max()  # L∞ norm for each sample, then max
        self.beta = B * (k ** .5)
    
        # Normalize all features except intercept column
        if self.fit_intercept:
            X_normalized = X[:, :-1] / self.beta
            X = ch.cat([X_normalized, X[:, -1:]], dim=1)  # Keep intercept as 1
        else:
            X = X / self.beta

        self.train_loader, self.val_loader = make_train_and_val(self.args, X, y) 
        
        self.trainer = Trainer(self, self.args) 
        self.trainer.train_model(self.train_loader, 
                                 self.val_loader, 
                                 rand_seed=self.rand_seed)
        return self

    def pretrain_hook(self, 
                      train_loader: ch.utils.data.DataLoader):
        self.calc_emp_model(train_loader)
        # use OLS as empirical estimate to define projection set
        self.radius = self.args.r * self.base_radius
        self.register_parameter("weight", nn.Parameter(self.emp_weight.clone()))

    def calc_emp_model(self, 
                       train_loader: ch.utils.data.DataLoader) -> None: 
        X, y = train_loader.dataset.tensors
        emp_lasso = LassoCV(fit_intercept=self.fit_intercept, alphas=[self.l1]) 
        emp_lasso.fit(X, y)
        lasso_coef_ = ch.from_numpy(np.concatenate([emp_lasso.coef_flatten(), emp_lasso.intercept_]))
        self.register_buffer('emp_weight', lasso_coef_)

    def __call__(self, 
                 X: ch.Tensor, 
                 y: ch.Tensor): 
        return X@self.weight

    def pre_step_hook(self, 
                      inp: ch.Tensor) -> None:
        if self.noise_var is not None and not self.dependent:
            # only regulaize the weight coefficients and not intercept
            self.weight.grad += (self.l1 * ch.sign(inp)).mean(0)[...,None]

    def post_training_hook(self): 
        best_params = self.trainer.best_params
        final_params = self.trainer.final_params
        if self.args.r is not None: self.args.r *= self.args.rate
        if self.fit_intercept: 
            self.best_coef = best_params[:,:-1] / self.beta
            self.best_intercept = best_params[:,-1]
            self.final_coef = final_params[:,:-1] / self.beta
            self.final_intercept = final_params[:,-1]
            self.emp_weight /= self.beta
        else: 
            self.best_coef = best_params[:] / self.beta
            self.final_coef = final_params[:] / self.beta
            self.emp_weight /= self.beta

    def predict(self, 
                X: Tensor): 
        """
        Make predictions with regression estimates.
        """
        assert self.coef is not None, "must fit model before using predict method"
        if self.fit_intercept: 
            return X@self.best_coef + self.best_intercept
        return X@self.best_coef

    @property
    def best_coef_(self): 
        return self.best_coef

    @property
    def best_intercept_(self): 
        if self.fit_intercept:
            return self.best_intercept
        warnings.warn("intercept not fit, check inputs.") 
    
    @property
    def final_coef_(self): 
        return self.final_coef
    
    @property
    def final_intercept_(self): 
        if self.fit_intercept:
            return self.final_intercept
        warnings.warn("intercept not fit, check inputs.") 

    @property
    def coef_(self): 
        """
        Regression coefficient weights.
        """

        return self.best_coef.clone()

    @property
    def intercept_(self): 
        """
        Regression intercept.
        """
        if self.best_intercept:
            return self.best_intercept.clone()
        warnings.warn("intercept not fit, check args input.")
