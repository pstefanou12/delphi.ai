"""
Truncated Linear Regression.
"""

from re import A
import torch as ch
from torch import Tensor
import cox
import warnings
from typing import Callable

from .linear_model import LinearModel
from ..grad import TruncatedMSE, TruncatedUnknownVarianceMSE
from ..utils.datasets import make_train_and_val
from ..utils.helpers import Parameters
from .linear_model import LinearModel

REQ = 'required'

# DEFAULT PARAMETERS
TRUNC_REG_DEFAULTS = {
        'phi': (Callable, REQ),
        'noise_var': (float, None), 
        'fit_intercept': (bool, True), 
        'val': (float, .2),
        'var_lr': (float, 1e-2), 
        'l1': (float, 0.0), 
        'eps': (float, 1e-5),
        'r': (float, 1.0), 
        'rate': (float, 1.5), 
        'batch_size': (int, 50),
        'workers': (int, 0),
        'num_samples': (int, 50),
}


TRUNC_LASSO_DEFAULTS = {
        'phi': (Callable, REQ),
        'noise_var': (float, None), 
        'fit_intercept': (bool, True), 
        'num_trials': (int, 3),
        'val': (float, .2),
        'lr': (float, 1e-1), 
        'var_lr': (float, 1e-2), 
        'l1': (float, REQ), 
        'eps': (float, 1e-5),
        'r': (float, 1.0), 
        'rate': (float, 1.5), 
        'batch_size': (int, 10),
        'workers': (int, 0),
        'num_samples': (int, 10),
}


TRUNC_RIDGE_DEFAULTS = {
        'phi': (Callable, REQ),
        'noise_var': (float, None), 
        'fit_intercept': (bool, True), 
        'num_trials': (int, 3),
        'val': (float, .2),
        'lr': (float, 1e-1), 
        'var_lr': (float, 1e-2), 
        'l1': (float, 0.0), 
        'weight_decay': (float, REQ),
        'eps': (float, 1e-5),
        'r': (float, 1.0), 
        'rate': (float, 1.5), 
        'batch_size': (int, 10),
        'workers': (int, 0),
        'num_samples': (int, 10),
}


TRUNC_ELASTIC_NET_DEFAULTS = {
        'phi': (Callable, REQ),
        'noise_var': (float, None), 
        'fit_intercept': (bool, True), 
        'num_trials': (int, 3),
        'val': (float, .2),
        'lr': (float, 1e-1), 
        'var_lr': (float, 1e-2), 
        'l1': (float, REQ),
        'weight_decay': (float, REQ),
        'eps': (float, 1e-5),
        'r': (float, 1.0), 
        'rate': (float, 1.5), 
        'batch_size': (int, 10),
        'workers': (int, 0),
        'num_samples': (int, 10),
}



class TruncatedLinearRegression(LinearModel):
    """
    Truncated linear regression class. Supports truncated linear regression
    with known noise, unknown noise, and confidence intervals. Module uses 
    delphi.trainer.Trainer to train truncated linear regression by performing 
    projected stochastic gradient descent on the truncated population log likelihood. 
    Module requires the user to specify an oracle from the delphi.oracle.oracle class, 
    and the survival probability. 
    """
    def __init__(self,
                args: Parameters,
                store: cox.store.Store=None):
        """
        Args: 
            phi (delphi.oracle.oracle) : oracle object for truncated regression model 
            alpha (float) : survival probability for truncated regression model
            fit_intercept (bool) : boolean indicating whether to fit a intercept or not 
            val (int) : number of samples to use for validation set 
            tol (float) : gradient tolerance threshold 
            workers (int) : number of workers to spawn 
            r (float) : size for projection set radius 
            rate (float): rate at which to increase the size of the projection set, when procedure does not converge - input as a decimal percentage
            num_samples (int) : number of samples to sample in gradient 
            batch_size (int) : batch size
            lr (float) : initial learning rate for regression weight parameters 
            var_lr (float) : initial learning rate to use for variance parameter in the settign where the variance is unknown 
            step_lr (int) : number of gradient steps to take before decaying learning rate for step learning rate 
            custom_lr_multiplier (str) : "cosine" (cosine annealing), "adam" (adam optimizer) - different learning rate schedulers available
            lr_interpolation (str) : "linear" linear interpolation
            step_lr_gamma (float) : amount to decay learning rate when running step learning rate
            momentum (float) : momentum for SGD optimizer 
            eps (float) :  epsilon value for gradient to prevent zero in denominator
            store (cox.store.Store) : cox store object for logging 
        """
        super().__init__(args, defaults=TRUNC_REG_DEFAULTS, store=store)

        del self.criterion
        if self.args.noise_var is None: 
            self.criterion = TruncatedUnknownVarianceMSE.apply
        else: 
            self.criterion = TruncatedMSE.apply
            self.criterion_params = [ 
                self.args.phi, self.args.noise_var,
                self.args.num_samples, self.args.eps,
            ]
        # property instance variables 
        self.coef, self.intercept = None, None


    def fit(self, 
            X: Tensor, 
            y: Tensor):
        """
        Train truncated linear regression model by running PSGD on the truncated negative 
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
        self.train_loader_, self.val_loader_ = make_train_and_val(self.args, X, y) 
        self.train_model(self.train_loader_, self.val_loader_)

        # reparameterize the regression's parameters
        if self.args.noise_var is None: 
            self.variance = self.lambda_.inverse()
            self.weight *= self.variance

        # assign results from procedure to instance variables
        if self.args.fit_intercept: 
            self.coef = self.weight[:-1]
            self.intercept = self.weight[-1]
        else: 
            self.coef = self.weight[:]
        return self

    def predict(self, 
                X: Tensor): 
        """
        Make predictions with regression estimates.
        """
        assert self.coef is not None, "must fit model before using predict method"
        if self.args.fit_intercept: 
            return X@self.coef + self.intercept
        return X@self.coef

    @property
    def coef_(self): 
        """
        Regression coefficient weights.
        """
        return self.coef

    @property
    def intercept_(self): 
        """
        Regression intercept.
        """
        if self.intercept is not None:
            return self.intercept
        warnings.warn("intercept not fit, check args input.") 

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
    def ols_coef_(self): 
        """
        OLS empirical estimates for coefficients.
        """
        return self.trunc_reg.emp_weight.clone()

    @property
    def ols_intercept_(self):
        """
        OLS empirical estimates for intercept.
        """
        return self.trunc_reg.emp_bias.clone()

    @property
    def ols_variance_(self): 
        """
        OLS empirical estimates for noise variance.
        """
        return self.trunc_reg.emp_var.clone()

    def __call__(self, X: ch.Tensor, y: ch.Tensor):
        if self.args.noise_var is None:
            weight = self._parameters[0]['params'][0]
            lambda_ = self._parameters[1]['params'][0]
            return X@weight * lambda_.inverse() 
        return X@self.weight
       
    def pre_step_hook(self, inp) -> None:
        # l1 regularization
        if self.args.noise_var is not None:
            self.weight.grad += (self.args.l1 * ch.sign(inp)).mean(0)[...,None]

    def iteration_hook(self, i, loop_type, loss, batch) -> None:
        if self.args.noise_var is None:
            # project model parameters back to domain 
            var = self._parameters[1]['params'][0].inverse()
            self._parameters[1]['params'][0].data = ch.clamp(var, self.var_bounds.lower, self.var_bounds.upper).inverse()