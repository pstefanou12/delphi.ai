"""
Truncated Ridge Regression.
"""

import torch as ch
from torch import Tensor
from sklearn.linear_model import ElasticNet
import cox
import warnings

from .stats import stats
from ..trainer import Trainer
from .truncated_linear_regression import KnownVariance, UnknownVariance
from ..utils.datasets import make_train_and_val
from ..utils.helpers import Parameters
from ..utils.defaults import check_and_fill_args, TRAINER_DEFAULTS, DELPHI_DEFAULTS, TRUNC_ELASTIC_NET_DEFAULTS



class TruncatedElasticNetRegression(stats):
    '''
    Truncated elastic net regression class. Supports truncated elastic net regression
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
        super(TruncatedElasticNetRegression).__init__()
        # instance variables
        assert isinstance(args, Parameters), "args is type: {}. expecting args to be type delphi.utils.helpers.Parameters"
        assert store is None or isinstance(store, cox.store.Store), "store is type: {}. expecting cox.store.Store.".format(type(store))
        self.store = store 
        self.trunc_ridge = None
        # algorithm hyperparameters
        TRUNC_ELASTIC_NET_DEFAULTS.update(TRAINER_DEFAULTS)
        TRUNC_ELASTIC_NET_DEFAULTS.update(DELPHI_DEFAULTS)
        self.args = check_and_fill_args(args, TRUNC_ELASTIC_NET_DEFAULTS)
        assert self.args.weight_decay > 0, "ridge regression requires l2 coefficient to be nonzero"

    def fit(self, X: Tensor, y: Tensor):
        """
        Train truncated elastic net regression model by running PSGD on the truncated negative 
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
            self.trunc_elastic_net = ElasticNetUnknownVariance(self.args, self.train_loader_) 
        else: 
            self.trunc_elastic_net = ElasticNetKnownVariance(self.args, self.train_loader_) 
        
        # run PGD for parameter estimation
        trainer = Trainer(self.trunc_elastic_net, self.args, store=self.store)
        trainer.train_model((self.train_loader_, self.val_loader_))

        # assign results from procedure to instance variables
        self.coef = self.trunc_elastic_net.model.weight.clone()
        if self.args.fit_intercept: self.intercept = self.trunc_elastic_net.model.bias.clone()
        if self.args.noise_var is None: 
            self.variance = self.trunc_elastic_net.lambda_.clone().inverse()
            self.coef *= self.variance
            if self.args.fit_intercept: self.intercept *= self.variance.flatten()
        return self

    def predict(self, x: Tensor): 
        """
        Make predictions with regression estimates.
        """
        return self.trunc_elastic_net.model(x)

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
        if self.args.fit_intercept:
            return self.intercept
        warnings.warn('intercept not fit, check args input.') 


class ElasticNetKnownVariance(KnownVariance):
    '''
    Truncated ridge regression with known noise variance model.
    '''
    def __init__(self, args, train_loader): 
        '''
        Args: 
            args (cox.utils.Parameters) : parameter object holding hyperparameters
        '''
        super().__init__(args, train_loader)
        
    def calc_emp_model(self):
        # calculate empirical estimates
        self.emp_model = ElasticNet(fit_intercept=self.args.fit_intercept, alpha=self.args.weight_decay, l1_ratio=self.args.l1).fit(self.X, self.y.flatten())
        self.emp_weight = Tensor(self.emp_model.coef_)[None,...]
        if self.args.fit_intercept:
            self.emp_bias = Tensor([self.emp_model.intercept_])
        self.emp_var = ch.var(Tensor(self.emp_model.predict(self.X))[...,None] - self.y, dim=0)[..., None]


class ElasticNetUnknownVariance(UnknownVariance):
    '''
    Truncated elastic net regression with unknown noise variance model.
    '''
    def __init__(self, args, train_loader): 
        '''
        Args: 
            args (cox.utils.Parameters) : parameter object holding hyperparameters
        '''
        super().__init__(args, train_loader)

    def calc_emp_model(self):
        # calculate empirical estimates
        self.emp_model = ElasticNet(fit_intercept=self.args.fit_intercept, alpha=self.args.weight_decay, l1_ratio=self.args.l1).fit(self.X, self.y.flatten())
        self.emp_weight = Tensor(self.emp_model.coef_)[None,...]
        if self.args.fit_intercept:
            self.emp_bias = Tensor([self.emp_model.intercept_])
        self.emp_var = ch.var(Tensor(self.emp_model.predict(self.X))[...,None] - self.y, dim=0)[..., None]
