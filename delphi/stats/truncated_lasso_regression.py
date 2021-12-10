"""
Truncated Lasso Regression.
"""
import torch as ch
from torch import Tensor
from sklearn.linear_model import LassoCV
import cox
import warnings

from .. import delphi
from ..oracle import oracle
from ..trainer import Trainer
from .linear_regression import KnownVariance, UnknownVariance
from ..utils.helpers import make_train_and_val


# CONSTANTS 
DEFAULTS = {
        'phi': (oracle, 'required'),
        'alpha': (float, 'required'), 
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
        'l1': (float, 1e-3), 
        'eps': (float, 1e-5),
        'r': (float, 1.0), 
        'rate': (float, 1.5), 
        'normalize': (bool, True), 
        'batch_size': (int, 10),
        'tol': (float, 1e-1),
        'workers': (int, 0),
        'num_samples': (int, 10),
        'early_stopping': (bool, False), 
        'n_iter_no_change': (int, 5),
        'verbose': (bool, False),
}


class TruncatedLassoRegression(delphi.delphi):
    '''
    Truncated LASSO regression class. Supports truncated LASSO regression
    with known noise, unknown noise, and confidence intervals. Module uses 
    delphi.trainer.Trainer to train truncated linear regression by performing 
    projected stochastic gradient descent on the truncated population log likelihood. 
    Module requires the user to specify an oracle from the delphi.oracle.oracle class, 
    and the survival probability. 
    '''
    def __init__(self,
                args: dict, 
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
        super(TruncatedLassoRegression).__init__()
        # instance variables
        assert store is None or isinstance(store, cox.store.Store), "store is type: {}. expecting cox.store.Store.".format(type(store))
        self.store = store 
        self.trunc_lasso = None
        # algorithm hyperparameters
        self.args = check_and_fill_args(Parameters(args), DEFAULTS)

        assert self.args.l1 > 0, "LASSO regression requires l1 coefficient to be non-zero"
        
    def fit(self, X: Tensor, y: Tensor):
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

        self.train_loader_, self.val_loader_ = make_train_and_val(self.args, X, y) 
        if self.args.noise_var is None:
            self.trunc_lasso = LassoUnknownVariance(self.args, self.train_loader_, self.args.phi) 
        else: 
            self.trunc_lasso = LassoKnownVariance(self.args, self.train_loader_, self.args.phi) 
        
        # run PGD for parameter estimation
        trainer = Trainer(self.trunc_lasso, max_iter=self.args.epochs, trials=self.args.num_trials,
                                        tol=self.args.tol, store=self.store, verbose=self.args.verbose, 
                                        early_stopping=self.args.early_stopping)

        trainer.train_model((self.train_loader_, self.val_loader_))

        with ch.no_grad():
            # assign results from procedure to instance variables
            self.coef = self.trunc_lasso.model.weight.clone()
            if self.args.fit_intercept: self.intercept = self.trunc_lasso.model.bias.clone()
            if self.args.noise_var is None: 
                self.variance = self.trunc_lasso.scale.clone().inverse()
                self.coef *= self.variance
                if self.args.fit_intercept: self.intercept *= self.variance.flatten()

    def predict(self, x: Tensor): 
        """
        Make predictions with regression estimates.
        """
        with ch.no_grad():
            return self.trunc_lasso.model(x)

    @property
    def coef_(self): 
        """
        Lasso regression coefficient weights.
        """
        return self.coef

    @property
    def intercept_(self): 
        """
        Lasso regression intercept.
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


class LassoKnownVariance(KnownVariance):
    '''
    Truncated linear regression with known noise variance model.
    '''
    def __init__(self, args, train_loader): 
        '''
        Args: 
            args (cox.utils.Parameters) : parameter object holding hyperparameters
        '''
        super().__init__(args, train_loader)
        
    def calc_emp_model(self):
        # calculate empirical estimates
        self.emp_model = LassoCV(fit_intercept=self.args.fit_intercept, alphas=[self.args.l1]).fit(self.X, self.y.flatten())
        self.emp_weight = Tensor(self.emp_model.coef_)[None,...]
        if self.args.fit_intercept:
            self.emp_bias = Tensor([self.emp_model.intercept_])
        self.emp_var = ch.var(Tensor(self.emp_model.predict(self.X))[...,None] - self.y, dim=0)[..., None]


class LassoUnknownVariance(UnknownVariance):
    '''
    Parent/abstract class for models to be passed into trainer.  
    '''
    def __init__(self, args, train_loader): 
        '''
        Args: 
            args (cox.utils.Parameters) : parameter object holding hyperparameters
        '''
        super().__init__(args, train_loader)

    def calc_emp_model(self):
        # calculate empirical estimates
        self.emp_model = LassoCV(fit_intercept=self.args.fit_intercept, alphas=[self.args.l1]).fit(self.X, self.y.flatten())
        self.emp_weight = Tensor(self.emp_model.coef_)[None,...]
        if self.args.fit_intercept:
            self.emp_bias = Tensor([self.emp_model.intercept_])
        self.emp_var = ch.var(Tensor(self.emp_model.predict(self.X))[...,None] - self.y, dim=0)[..., None]


