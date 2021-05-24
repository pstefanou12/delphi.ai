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
            bias: bool=True,
            unknown: bool=True,
            clamp: bool=True,
            n: int=10, 
            val: int=50,
            tol: float=1e-2,
            workers: int=0,
            r: float=2.0,
            num_samples: int=100,
            bs: int=10,
            **kwargs):
        """
        """
        super(TruncatedRegression).__init__()
        # instance variables
        self.phi = phi 
        self.alpha = alpha 
        self.steps = steps
        self.bias = bias 
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
        self.ds = None

        config.args = Parameters({ 
            'steps': steps,
            'momentum': 0.0, 
            'weight_decay': 0.0, 
            'step_lr': 100, 
            'step_lr_gamma': .9,    
            # 'custom_lr_multiplier': consts.CYCLIC,
            'num_samples': self.num_samples,
            'lr': 1e-1,  
            'var_lr': 1e-1,
            'eps': 1e-5,
        })

    def fit(self, X: Tensor, y: Tensor):
        """
        """
        # separate into training and validation set
        rand_indices = ch.randperm(X.size(0))
        train_indices, val_indices = rand_indices[self.val:], rand_indices[:self.val]
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]

        self.ds = TensorDataset(X, y)
        loader = DataLoader(self.ds, batch_size=self.bs, num_workers=self.workers)
        self.emp_lin_reg = LinearRegression(fit_intercept=self.bias).fit(X, y)
        self.emp_weight = Tensor(self.emp_lin_reg.coef_)
        self.emp_bias = Tensor(self.emp_lin_reg.intercept_) if self.bias else None
        self.emp_var = ch.var(Tensor(self.emp_lin_reg.predict(X)) - y, dim=0)[..., None]
        
        if self.unknown: # known variance
            self._lin_reg = LinearUnknownVariance(in_features=X.size(1), out_features=y.size(1), bias=self.bias)
            # assign empirical estimates
            self._lin_reg.lambda_.data = self.emp_var.inverse()
            self._lin_reg.weight.data = self.emp_weight * self._lin_reg.lambda_ 
            if self.bias: 
                self._lin_reg.bias.data = (self.emp_bias * self._lin_reg.lambda_).flatten()
            update_params = [{'params': [self._lin_reg.weight, self._lin_reg.bias if self._lin_reg.bias is not None else None]},
                {'params': self._lin_reg.lambda_, 'lr': config.args.var_lr}]
        else:  # unknown variance
            self._lin_reg = Linear(in_features=X.size(1), out_features=1, bias=self.bias)
            # assign empirical estimates
            self._lin_reg.weight.data = self.emp_weight
            if self.bias: 
                self._lin_reg.bias.data = self.emp_bias
            update_params = None

        self.iter_hook = TruncatedRegressionIterationHook(X_train, y_train, X_val, y_val, self.phi, self.tol, self.r, self.alpha, self.bias, self.clamp, self.unknown, self.n, self.criterion)
        config.args.__setattr__('iteration_hook', self.iter_hook)
        # run PGD for parameter estimation
        return train_model(config.args, self._lin_reg, (loader, None), phi=self.phi, criterion=self.criterion, update_params=update_params)

    def __call__(self, x: Tensor): 
        """
        """
        return self._lin_reg(x)


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
    def __init__(self, X_train, y_train, X_val, y_val, phi, tol, r, alpha, bias, clamp, unknown, n, criterion):
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
        self.bias = bias
        self.r = r
        self.alpha = alpha
        self.unknown = unknown
        self.phi = phi

        # initialize projection set
        self.clamp = clamp
        self.emp_lin_reg = LinearRegression(fit_intercept=self.bias).fit(X_train, y_train)
        self.emp_weight = Tensor(self.emp_lin_reg.coef_) 
        self.emp_bias = Tensor(self.emp_lin_reg.intercept_) if self.bias else None
        self.emp_var = ch.var(Tensor(self.emp_lin_reg.predict(X_train)) - y_train, dim=0)[..., None]
        self.radius = r * (12.0 + 4.0 * ch.log(2.0 / self.alpha)) if self.unknown else r * (4.0 * ch.log(2.0 / self.alpha) + 7.0)

        if self.clamp:
            self.weight_bounds = Bounds(self.emp_weight.flatten() - self.radius,
                                        self.emp_weight.flatten() + self.radius)
            # generate noise variance radius bounds if unknown 
            self.var_bounds = Bounds(self.emp_var.flatten() / self.r, (self.emp_var.flatten()) / self.alpha.pow(2)) if self.unknown else None
            self.bias_bounds = Bounds(self.emp_bias.flatten() - self.radius,
                                      self.emp_bias.flatten() + self.radius) if self.bias else None
        else:
            pass

        # validation set
        # use steps counter to keep track of steps taken
        self.n, self.steps = n, 0
        self.X_val, self.y_val = X_val, y_val
        self.criterion = criterion
        self.tol = tol
        # track best estimates based off of gradient norm
        self.best_w, self.best_w0, self.best_lambda = None, None, None
        self.best_grad_norm = None
        # calculate empirical score
        emp = LinearUnknownVariance(in_features=X_train.size(0), out_features=1, bias=self.bias) if self.unknown else ch.nn.Linear(in_features = X_train.size(1), out_features=1, bias=self.bias) 
        if self.unknown: 
            emp.lambda_.data = self.emp_var.inverse()
        emp.weight.data = self.emp_weight * emp.lambda_ if self.unknown else self.emp_weight
        if self.bias: 
            emp.bias.data = (self.emp_bias * emp.lambda_).flatten() if self.unknown else self.emp_bias

        self.score(emp)

    def score(self, M): 
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

        print("{} steps | score: {}".format(self.steps, grad.tolist()))
        # check that gradient magnitude is less than tolerance
        if self.steps != 0 and ch.all(ch.abs(grad) < self.tol): 
            raise ProcedureComplete()

        # grad_norm = grad.norm(dim=-1)
        # # if smaller gradient, update best
        # if self.best_grad_norm is None or grad_norm < self.best_grad_norm: 
        #     self.best_grad_norm = grad_norm
        #     self.best_w, self.best_w0 = M.weight.data.clone(), M.bias.data.clone().flatten() if self.bias else None
        #     if self.unknown: 
        #         self.best_lambda = M.lambda_.data.clone()
        # else: 
        #     M.weight.data = self.best_w.clone()
        #     M.bias.data = self.best_w0.clone() if self.bias else None
        #     if self.unknown: 
        #         M.lambda_.data = self.best_lambda.clone()
        


    def __call__(self, M, i, loop_type, inp, target): 
        # increase number of steps taken
        self.steps += 1
        # project model parameters back to domain 
        if self.clamp: 
            if self.unknown: 
                var = M.lambda_.inverse()
                weight = M.weight * var

                M.lambda_.data = ch.clamp(var, float(self.var_bounds.lower), float(self.var_bounds.upper)).inverse()
                # project weights
                M.weight.data = ch.cat(
                    [ch.clamp(weight[i].unsqueeze(0), float(self.weight_bounds.lower[i]),
                            float(self.weight_bounds.upper[i]))
                    for i in range(weight.size(0))]) * M.lambda_
                # project bias
                if self.bias:
                    bias = M.bias * var
                    M.bias.data = (ch.clamp(bias, float(self.bias_bounds.lower), float(self.bias_bounds.upper)) * M.lambda_).reshape(M.bias.size())
            else: 
                M.weight.data = ch.stack(
                    [ch.clamp(M.weight[i], self.weight_bounds.lower[i], self.weight_bounds.upper[i]) for i in
                    range(M.weight.size(0))])
                if self.bias:
                    M.bias.data = ch.clamp(M.bias, float(self.bias_bounds.lower), float(self.bias_bounds.upper)).reshape(
                        M.bias.size())
        else: 
            pass

        # check for convergence every n steps
        if self.steps % self.n == 0: 
            self.score(M)

