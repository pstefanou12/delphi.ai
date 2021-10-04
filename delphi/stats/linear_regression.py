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

from .stats import stats
from ..oracle import oracle
from ..train import train_model
from ..grad import TruncatedMSE, TruncatedUnknownVarianceMSE
from ..utils import constants as consts
from ..utils.helpers import Bounds, LinearUnknownVariance, setup_store_with_metadata, ProcedureComplete


class TruncatedRegression(stats):
    """
    """
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
        """
        """
        super(TruncatedRegression).__init__()
        # instance variables
        self.phi = phi 
        self.alpha = alpha 
        self.steps = steps
        self.unknown = unknown 
        self._lin_reg = None
        self.criterion = TruncatedUnknownVarianceMSE.apply if self.unknown else TruncatedMSE.apply
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
        self.var_lr = var_lr
        self.step_lr = step_lr
        self.custom_lr_multiplier = custom_lr_multiplier
        self.step_lr_gamma = step_lr_gamma
        self.eps = eps 
        self.ds = None

        config.args = Parameters({ 
            'steps': self.steps,
            'momentum': 0.0, 
            'weight_decay': 0.0,   
            'num_samples': self.num_samples,
            'lr': self.lr,  
            'var_lr': self.var_lr,
            'eps': self.eps,
        })

        # ste attribute for learning rate scheduler
        if self.custom_lr_multiplier: 
            config.args.__setattr__('custom_lr_multiplier', self.custom_lr_multiplier)
        else: 
            config.args.__setattr__('step_lr', self.step_lr)
            config.args.__setattr__('step_lr_gamma', self.step_lr_gamma)


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
        self.emp_lin_reg = LinearRegression().fit(X, y)
        self.emp_weight = Tensor(self.emp_lin_reg.coef_)
        self.emp_bias = Tensor(self.emp_lin_reg.intercept_)
        self.emp_var = ch.var(Tensor(self.emp_lin_reg.predict(X)) - y, dim=0)[..., None]
        
        if self.unknown: # known variance
            self._lin_reg = LinearUnknownVariance(in_features=X.size(1), out_features=y.size(1), bias=True)
            # assign empirical estimates
            self._lin_reg.lambda_.data = self.emp_var.inverse()
            self._lin_reg.weight.data = self.emp_weight * self._lin_reg.lambda_ 
            self._lin_reg.bias.data = (self.emp_bias * self._lin_reg.lambda_).flatten()
            update_params = [{'params': [self._lin_reg.weight, self._lin_reg.bias]},
                {'params': self._lin_reg.lambda_, 'lr': self.var_lr}]
        else:  # unknown variance
            self._lin_reg = Linear(in_features=X.size(1), out_features=y.size(1), bias=True)
            # assign empirical estimates
            self._lin_reg.weight.data = self.emp_weight
            self._lin_reg.bias.data = self.emp_bias
            update_params = None

        self.iter_hook = TruncatedRegressionIterationHook(self.X_train, self.y_train, self.X_val, self.y_val, self.phi, self.tol, self.r, self.alpha, self.clamp, self.unknown, self.n, self.criterion)
        config.args.__setattr__('iteration_hook', self.iter_hook)
        # run PGD for parameter estimation
        if self.score() > self.tol: # first check regression's empirical score
            self._lin_reg = train_model(config.args, self._lin_reg, (loader, None), phi=self.phi, criterion=self.criterion, update_params=update_params)
        # remove linear regression from computation graph

        with ch.no_grad():
            return self._lin_reg

    def __call__(self, x: Tensor): 
        """
        """
        return self._lin_reg(x)

    def score(self): 
        """
        Check the score of the validation set. Passes validation 
        set through regression and then returns the gradient with 
        respect to y and in the unknown setting with respect to lambda.
        """
        pred = self._lin_reg(self.X_val)
        if self.unknown:
            loss = self.criterion(pred, self.y_val, self._lin_reg.lambda_, self.phi)
            grad, lambda_grad = ch.autograd.grad(loss, [pred, self._lin_reg.lambda_])
            grad = ch.cat([(grad.sum(0) / self._lin_reg.lambda_).flatten(), lambda_grad.flatten()])
        else: 
            loss = self.criterion(pred, self.y_val, self.phi)
            grad, = ch.autograd.grad(loss, [pred])
            grad = grad.sum(0)
        return grad.norm(dim=-1)

    @property
    def weight(self): 
        """
        Regression weight.
        """
        if self.unknown: 
            return self._lin_reg.weight.detach().clone().T * self._lin_reg.lambda_.detach().inverse().clone()
        return self._lin_reg.weight.detach().clone().T

    @property
    def intercept(self): 
        """
        Regression intercept.
        """
        if self.unknown: 
            return self._lin_reg.bias.detach().clone().T * self._lin_reg.lambda_.detach().inverse().clone()   
        return self._lin_reg.bias.detach().clone()

    @property
    def variance(self): 
        """
        Noise variance prediction for linear regression with
        unknown noise variance algorithm.
        """
        if self.unknown: 
            return self._lin_reg.lambda_.detach().inverse().clone()
        else: 
            warnings.warn("no variance prediction because regression with known variance was run")


class TruncatedRegressionIterationHook: 
    """
    Iteration for truncated regression algorithm for the known and unknown cases. 
    Hook does two things. First it projects the current model estimates back into the 
    projection set. For the known case, we only project the regression parameters into 
    the domain. For the unknown case, we project both the model parameter and variance 
    estimates. Further, every n steps we check to check the gradient against our validation 
    set of samples. If the gradient for the samples is less than our tolerance, then 
    we terminate the procedure.
    """
    def __init__(self, X_train, y_train, X_val, y_val, phi, tol, r, alpha, clamp, unknown, n, criterion):
        """
        :param X_train: train covariates - torch.Tensor
        :param y_train: train dependent variable - torch.Tensor
        :param X_val: val covariates - torch.Tensor
        :param y_val: val dependent variable - torch.Tensor
        :param phi: membership oracle - delphi.oracle
        :param tol: gradient tolerance to end procedure - float
        :param alpha: survival probability - torch.Tensor
        :param clamp: boolean to use clamp projection set - bool
        :param unknown: boolean for known or unknown noise variance - bool
        :param n: number of steps to check gradient - int 
        :param criterion: criterion to determine convergence - torch.autograd.Function 
        """
        # use OLS as empirical estimate to define projection set
        self.r = r
        self.alpha = alpha
        self.unknown = unknown
        self.phi = phi

        # initialize projection set
        self.clamp = clamp
        self.emp_lin_reg = LinearRegression().fit(X_train, y_train)
        self.emp_weight = Tensor(self.emp_lin_reg.coef_) 
        self.emp_bias = Tensor(self.emp_lin_reg.intercept_)
        self.emp_var = ch.var(Tensor(self.emp_lin_reg.predict(X_train)) - y_train, dim=0)[..., None]
        self.radius = r * (12.0 + 4.0 * ch.log(2.0 / self.alpha)) if self.unknown else r * (4.0 * ch.log(2.0 / self.alpha) + 7.0)

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
        self.criterion = criterion
        self.tol = tol
        # track best estimates based off of gradient norm
        self.best_grad_norm = None
        self.best_state_dict = None
        self.best_opt = None
        # calculate empirical score
        emp = LinearUnknownVariance(in_features=X_train.size(0), out_features=1, bias=True) if self.unknown else ch.nn.Linear(in_features = X_train.size(1), out_features=1, bias=True) 
        if self.unknown: 
            emp.lambda_.data = self.emp_var.inverse()
        emp.weight.data = self.emp_weight * emp.lambda_ if self.unknown else self.emp_weight
        emp.bias.data = (self.emp_bias * emp.lambda_).flatten() if self.unknown else self.emp_bias

    def score(self, M, optimizer): 
        """
        Calculates the score of the current regression estimates of the validation set. It 
        then updates the best estimates accordingly based off of the score's norm.
        """
        pred = M(self.X_val)
        if self.unknown:
            loss = self.criterion(pred, self.y_val, M.lambda_, self.phi)
            grad, lambda_grad = ch.autograd.grad(loss, [pred, M.lambda_])
            grad = ch.cat([(grad.sum(0) / M.lambda_).flatten(), lambda_grad.flatten()])
        else: 
            loss = self.criterion(pred, self.y_val, self.phi)
            grad, = ch.autograd.grad(loss, [pred])
            grad = grad.sum(0)

        grad_norm = grad.norm(dim=-1)
        # check that gradient magnitude is less than tolerance
        if self.steps != 0 and grad_norm < self.tol: 
            print("Final Score: {}".format(grad_norm))
            raise ProcedureComplete()
        
        print("Iteration {} | Score: {}".format(int(self.steps / self.n), grad_norm))
        # if smaller gradient norm, update best
        if self.best_grad_norm is None or grad_norm < self.best_grad_norm: 
            self.best_grad_norm = grad_norm
            # keep track of state dict
            self.best_state_dict, self.best_opt = copy.deepcopy(M.state_dict()), copy.deepcopy(optimizer.state_dict())
        elif 1e-1 <= grad_norm - self.best_grad_norm: 
            # load in the best model state and optimizer dictionaries
            M.load_state_dict(self.best_state_dict)
            optimizer.load_state_dict(self.best_opt)

    def __call__(self, M, optimizer, i, loop_type, inp, target): 
        # increase number of steps taken
        self.steps += 1
        # project model parameters back to domain 
        if self.clamp: 
            if self.unknown: 
                var = M.lambda_.inverse()
                weight = M.weight * var

                M.lambda_.data = ch.clamp(var, self.var_bounds.lower, self.var_bounds.upper).inverse()
                # project weights
                M.weight.data = ch.cat([ch.clamp(weight[:,i], self.weight_bounds.lower[i], self.weight_bounds.upper[i])
                    for i in range(weight.size(1))])[None,...] * M.lambda_
                # project bias
                bias = M.bias * var
                M.bias.data = (ch.clamp(bias, self.bias_bounds.lower, self.bias_bounds.upper) * M.lambda_).reshape(M.bias.size())
            else: 
                M.weight.data = ch.cat([ch.clamp(M.weight[:,i], self.weight_bounds.lower[i], self.weight_bounds.upper[i]) 
                    for i in range(M.weight.size(1))])[None,...]
                # project bias
                M.bias.data = ch.clamp(M.bias, self.bias_bounds.lower, self.bias_bounds.upper).reshape(M.bias.size())
        else: 
            pass

        # check for convergence every n steps
        if self.steps % self.n == 0: 
            self.score(M, optimizer)

