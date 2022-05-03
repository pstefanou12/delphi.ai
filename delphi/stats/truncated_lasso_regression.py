"""
Truncated Lasso Regression.
"""

import torch as ch
from torch import Tensor
from sklearn.linear_model import LassoCV
import cox
import warnings

from .stats import stats
from ..trainer import Trainer
from .truncated_linear_regression import KnownVariance, UnknownVariance
from ..utils.datasets import make_train_and_val
from ..utils.helpers import Parameters
from ..grad import TruncatedLASSOMSE
from ..utils.defaults import check_and_fill_args, TRAINER_DEFAULTS, DELPHI_DEFAULTS, TRUNC_LASSO_DEFAULTS


class TruncatedLassoRegression(stats):
    """
    Truncated LASSO regression class. Supports truncated LASSO regression
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
        super(TruncatedLassoRegression).__init__()
        # instance variables
        assert isinstance(args, Parameters), "args is type: {}. expecting args to be type delphi.utils.helpers.Parameters".format(Parameters)
        assert store is None or isinstance(store, cox.store.Store), "store is type: {}. expecting cox.store.Store.".format(type(store))
        self.store = store 
        self.trunc_lasso = None
        # algorithm hyperparameters
        TRUNC_LASSO_DEFAULTS.update(TRAINER_DEFAULTS)
        TRUNC_LASSO_DEFAULTS.update(DELPHI_DEFAULTS)
        self.args = check_and_fill_args(args, TRUNC_LASSO_DEFAULTS)

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
        # add one feature to x when fitting intercept
        if self.args.fit_intercept:
            X = ch.cat([X, ch.ones(X.size(0), 1)], axis=1)
        self.train_loader_, self.val_loader_ = make_train_and_val(self.args, X, y) 
        if self.args.noise_var is None:
            self.trunc_lasso = LassoUnknownVariance(self.args, self.train_loader_) 
        else: 
            self.trunc_lasso = LassoKnownVariance(self.args, self.train_loader_, X.size(1)) 
        
        # run PGD for parameter estimation
        trainer = Trainer(self.trunc_lasso, self.args, store=self.store) 
        trainer.train_model((self.train_loader_, self.val_loader_))

        # assign results from procedure to instance variables
        if self.args.fit_intercept: 
            self.coef = self.trunc_lasso.model.data[:-1]
            self.intercept = self.trunc_lasso.model.data[-1]
        else: 
            self.coef = self.trunc_lasso.model.data[:]
        return self

    def predict(self, x: Tensor): 
        """
        Make predictions with regression estimates.
        """
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


class LassoKnownVariance(KnownVariance):
    """
    Truncated linear regression with known noise variance model.
    """
    def __init__(self, args, train_loader, d): 
        """
        Args: 
            args (cox.utils.Parameters) : parameter object holding hyperparameters
        """
        super().__init__(args, train_loader, d)
        self.base_radius = len(train_loader.dataset) ** (.5)
        
    def calc_emp_model(self):
        # calculate empirical estimates
        self.emp_model = LassoCV(fit_intercept=False, alphas=[self.args.l1]).fit(self.X, self.y.flatten())
        self.emp_weight = Tensor(self.emp_model.coef_)[None,...].T
        self.noise_var = ch.var(Tensor(self.emp_model.predict(self.X))[...,None] - self.y, dim=0)[..., None]

    def __call__(self, batch): 
        """
        Calculates the negative log likelihood of the current regression estimates of the validation set.
        Args: 
            proc (bool) : boolean indicating whether, the function is being called within 
            a stochastic process, or someone is accessing the parent class"s property
        """
        X, y = batch
        pred = X@self.model
        loss = TruncatedLASSOMSE.apply(pred, y, self.args.phi, self.args.noise_var, self.model, self.args.num_samples, self.args.eps)
        return [loss, None, None]


class LassoUnknownVariance(UnknownVariance):
    """
    Parent/abstract class for models to be passed into trainer.  
    """
    def __init__(self, args, train_loader): 
        """
        Args: 
            args (cox.utils.Parameters) : parameter object holding hyperparameters
        """
        super().__init__(args, train_loader)

    def calc_emp_model(self):
        # calculate empirical estimates
        self.emp_model = LassoCV(fit_intercept=self.args.fit_intercept, alphas=[self.args.l1]).fit(self.X, self.y.flatten())
        self.emp_weight = Tensor(self.emp_model.coef_)[None,...]
        if self.args.fit_intercept:
            self.emp_bias = Tensor([self.emp_model.intercept_])
        self.emp_var = ch.var(Tensor(self.emp_model.predict(self.X))[...,None] - self.y, dim=0)[..., None]


