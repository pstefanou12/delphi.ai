"""
Truncated Linear Regression.
"""

import torch as ch
from torch import Tensor
import cox
import warnings
import math

from .linear_model import TruncatedLinearModel
from .stats import stats
from ..oracle import oracle
from ..trainer import Trainer
from ..grad import TruncatedMSE, TruncatedUnknownVarianceMSE
from ..utils.datasets import make_train_and_val
from ..utils.helpers import Parameters, Bounds, check_and_fill_args

# CONSTANTS 
DEFAULTS = {
        'phi': (oracle, 'required'),
        'alpha': (float, 'required'), 
        'epochs': (int, 5),
        'noise_var': (float, None), 
        'fit_intercept': (bool, True), 
        'trials': (int, 3),
        'val': (float, .2),
        'lr': (float, 1e-1), 
        'var_lr': (float, 1e-2), 
        'step_lr': (int, 100),
        'step_lr_gamma': (float, .9), 
        'custom_lr_multiplier': (str, None), 
        'momentum': (float, 0.0), 
        'weight_decay': (float, 1e-3), 
        'l1': (float, 0.0), 
        'eps': (float, 1e-5),
        'r': (float, 1.0), 
        'rate': (float, 1.5), 
        'normalize': (bool, True), 
        'batch_size': (int, 50),
        'tol': (float, 1e-3),
        'workers': (int, 0),
        'num_samples': (int, 50),
        'early_stopping': (bool, False), 
        'n_iter_no_change': (int, 5),
        'verbose': (bool, False),
}


class TruncatedLinearRegression(stats):
    '''
    Truncated linear regression class. Supports truncated linear regression
    with known noise, unknown noise, and confidence intervals. Module uses 
    delphi.trainer.Trainer to train truncated linear regression by performing 
    projected stochastic gradient descent on the truncated population log likelihood. 
    Module requires the user to specify an oracle from the delphi.oracle.oracle class, 
    and the survival probability. 
    '''
    def __init__(self,
                args: Parameters, 
                store: cox.store.Store=None):
        '''
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
            custom_lr_multiplier (str) : 'cosine' (cosine annealing), 'adam' (adam optimizer) - different learning rate schedulers available
            lr_interpolation (str) : 'linear' linear interpolation
            step_lr_gamma (float) : amount to decay learning rate when running step learning rate
            momentum (float) : momentum for SGD optimizer 
            eps (float) :  epsilon value for gradient to prevent zero in denominator
            store (cox.store.Store) : cox store object for logging 
        '''
        super(TruncatedLinearRegression).__init__()
        # instance variables
        assert isinstance(args, Parameters), "args is type: {}. expecting args to be type delphi.utils.helpers.Parameters"
        assert store is None or isinstance(store, cox.store.Store), "store is type: {}. expecting cox.store.Store.".format(type(store))
        self.store = store 
        self.trunc_reg = None
        # algorithm hyperparameters
        self.args = check_and_fill_args(args, DEFAULTS)

    def fit(self, X: Tensor, y: Tensor):
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

        self.train_loader_, self.val_loader_ = make_train_and_val(self.args, X, y) 
        if self.args.noise_var is None:
            self.trunc_reg = UnknownVariance(self.args, self.train_loader_) 
        else: 
            self.trunc_reg = KnownVariance(self.args, self.train_loader_) 
        
        # run PGD for parameter estimation
        trainer = Trainer(self.trunc_reg, self.args, store=self.store) 
        trainer.train_model((self.train_loader_, self.val_loader_))

        # assign results from procedure to instance variables
        self.coef = self.trunc_reg.model.weight.clone()
        if self.args.fit_intercept: self.intercept = self.trunc_reg.model.bias.clone()
        if self.args.noise_var is None: 
            self.variance = self.trunc_reg.lambda_.clone().inverse()
            self.coef *= self.variance
            if self.args.fit_intercept: self.intercept *= self.variance.flatten()
        return self

    def predict(self, x: Tensor): 
        """
        Make predictions with regression estimates.
        """
        with ch.no_grad():
            return self.trunc_reg.model(x)

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
        warnings.warn('intercept not fit, check args input.') 

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


class KnownVariance(TruncatedLinearModel):
    '''
    Truncated linear regression with known noise variance model.
    '''
    def __init__(self, args, train_loader): 
        '''
        Args: 
            args (cox.utils.Parameters) : parameter object holding hyperparameters
        '''
        super().__init__(args, train_loader)
        self.base_radius = (7.0 + 4.0 * math.log(2.0 / self.args.alpha))
        
    def pretrain_hook(self):
        # use OLS as empirical estimate to define projection set
        self.radius = self.args.r * self.base_radius

        # empirical estimates for projection set
        self.w = self.emp_weight 
        if self.args.fit_intercept: 
            self.w = ch.cat([self.emp_weight.flatten(), self.emp_bias])
        
        # assign empirical estimates
        self.model.weight.data = self.emp_weight
        if self.args.fit_intercept:
            self.model.bias.data = self.emp_bias
        self.params = None

    def __call__(self, batch): 
        """
        Calculates the negative log likelihood of the current regression estimates of the validation set.
        Args: 
            proc (bool) : boolean indicating whether, the function is being called within 
            a stochastic process, or someone is accessing the parent class's property
        """
        X, y = batch
        pred = self.model(X)
        loss = TruncatedMSE.apply(pred, y, self.args.phi, self.args.noise_var, self.args.num_samples, self.args.eps)
        return [loss, None, None]
        
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
        if self.args.fit_intercept: 
            temp_w = ch.cat([self.model.weight.flatten(), self.model.bias])
            w_diff = temp_w - self.w
            w_diff = w_diff[None, ...].renorm(p=2, dim=0, maxnorm=self.radius)
            self.model.weight.data, self.model.bias.data = self.emp_weight + w_diff[:,:-1], self.emp_bias + w_diff[:,-1]
        else: 
            w_diff = self.model.weight - self.w
            w_diff = w_diff.renorm(p=2, dim=0, maxnorm=self.radius)
            self.model.weight.data = self.emp_weight + w_diff 
    
    def regularize(self, batch):
        '''
        L1 regularizer for LASSO regression.
        '''
        if self.args.l1 == 0.0: return 0.0
        reg_term = 0
        for param in self.model.parameters(): 
            reg_term += param.norm()
        return self.args.l1 * reg_term

    def post_training_hook(self): 
        self.args.r *= self.args.rate
        # remove model from computation graph
        self.model.weight.requires_grad = False
        if self.args.fit_intercept:
            self.model.bias.requires_grad = False


class UnknownVariance(KnownVariance):
    '''
    Parent/abstract class for models to be passed into trainer.  
    '''
    def __init__(self, args, train_loader): 
        '''
        Args: 
            args (cox.utils.Parameters) : parameter object holding hyperparameters
        '''
        super().__init__(args, train_loader)
        
    def pretrain_hook(self):
        # use OLS as empirical estimate to define projection set
        self.radius = self.args.r * (12.0 + 4.0 * math.log(2.0 / self.args.alpha)) 
        
        # empirical estimates for projection set
        self.w = self.emp_weight.clone() 
        if self.args.fit_intercept: 
            self.w = ch.cat([self.emp_weight.flatten(), self.emp_bias])

        # generate noise variance radius bounds if unknown 
        self.var_bounds = Bounds(float(self.emp_var.flatten() / self.args.r), float(self.emp_var.flatten() / Tensor([self.args.alpha]).pow(2))) 
        
        # assign empirical estimates
        self.lambda_.data = self.emp_var.inverse()
        self.model.weight.data = self.emp_weight * self.lambda_
        if self.args.fit_intercept:
            self.model.bias.data = (self.emp_bias * self.lambda_).flatten()
            self.params = [{'params': [self.model.weight, self.model.bias]},
                {'params': self.lambda_, 'lr': self.args.var_lr}]
        else: 
            self.params = [{'params': [self.model.weight]},
                {'params': self.lambda_, 'lr': self.args.var_lr}]

    
    def __call__(self, batch):
        '''
        Training step for defined model.
        Args: 
            batch (Iterable) : iterable of inputs that 
        '''
        X, y = batch
        pred = self.model(X) * self.lambda_.inverse()
        loss = TruncatedUnknownVarianceMSE.apply(pred, y, self.lambda_, self.args.phi, self.args.num_samples, self.args.eps)
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
        var = self.lambda_.inverse()
        weight = self.model.weight * var
        self.lambda_.data = ch.clamp(var, self.var_bounds.lower, self.var_bounds.upper).inverse()
                
        if self.args.fit_intercept: 
            temp_w = ch.cat([self.model.weight.flatten(), self.model.bias])
            w_diff = temp_w - self.w
            w_diff = w_diff[None, ...].renorm(p=2, dim=0, maxnorm=self.radius)
            self.model.weight.data, self.model.bias.data = self.emp_weight + w_diff[:,:-1], self.emp_bias + w_diff[:,-1]
        else: 
            w_diff = self.model.weight - self.w
            w_diff = w_diff.renorm(p=2, dim=0, maxnorm=self.radius)
            self.model.weight.data = self.emp_weight + w_diff 


    def post_training_hook(self): 
        self.args.r *= self.args.rate
        # remove model from computation graph
        self.model.weight.requires_grad = False
        if self.args.fit_intercept:
            self.model.bias.requires_grad = False
        self.lambda_.requires_grad = False