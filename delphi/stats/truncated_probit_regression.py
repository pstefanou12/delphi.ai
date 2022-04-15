"""
Truncated Logistic Regression.
"""

import torch as ch
from torch import Tensor
from torch import sigmoid as sig
from statsmodels.discrete.discrete_model import Probit
from statsmodels.tools.tools import add_constant
import cox
import math
import warnings

from .truncated_linear_regression import KnownVariance
from .stats import stats
from ..grad import TruncatedProbitMLE
from ..trainer import Trainer
from ..utils.datasets import make_train_and_val
from ..utils.helpers import Parameters, accuracy
from ..utils.defaults import check_and_fill_args, TRAINER_DEFAULTS, DELPHI_DEFAULTS, TRUNC_PROB_REG_DEFAULTS


class TruncatedProbitRegression(stats):
    """
    Truncated Probit Regression supports both binary classification, when the noise distribution in the latent variable model in N(0, 1).
    """
    def __init__(self,
                args: Parameters,
                weight: ch.Tensor=None, 
                store: cox.store.Store=None):
        '''
        Args: 
            phi (delphi.oracle.oracle) : oracle object for truncated regression model 
            alpha (float) : survival probability for truncated regression model
            unknown (bool) : boolean indicating whether the noise variance is known or not - specifying algorithm to use 
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
            batch_size (int) : batch size
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
        super(TruncatedProbitRegression).__init__()
        # instance variables
        assert isinstance(args, Parameters), "args is type: {}. expecting args to be type delphi.utils.helpers.Parameters"
        assert store is None or isinstance(store, cox.store.Store), "store is type: {}. expecting cox.store.Store.".format(type(store))
        self.store = store
        self.trunc_prob_reg = None
        # algorithm hyperparameters
        TRUNC_PROB_REG_DEFAULTS.update(TRAINER_DEFAULTS)
        TRUNC_PROB_REG_DEFAULTS.update(DELPHI_DEFAULTS)
        self.args = check_and_fill_args(args, TRUNC_PROB_REG_DEFAULTS)

        assert weight is None or weight.dim() == 2, 'weight is size: {}. expecting two dims, with size 1 * d'.format(weight.size())
        self.weight = weight
                
    def fit(self, X: Tensor, y: Tensor):
        """
        Train truncated probit regression model by running PSGD on the truncated negative 
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

        k = 1
        self.train_loader_, self.val_loader_ = make_train_and_val(self.args, X, y) 

        self.trunc_prob_reg = TruncatedProbitRegressionModel(self.args, self.weight, self.train_loader_, X.size(1), k)
        trainer = Trainer(self.trunc_prob_reg, self.args, store=self.store) 
        # run PGD for parameter estimation 
        trainer.train_model((self.train_loader_, self.val_loader_))

        # assign results from procedure to instance variables
        if self.args.fit_intercept: 
            self.coef = self.trunc_prob_reg.model.data[:-1]
            self.intercept = self.trunc_prob_reg.model.data[-1]
        else: 
            self.coef = self.trunc_prob_reg.model.data[:]
        return self

    def __call__(self, x: Tensor):
        """
        Calculate probit regression's latent variable, based off of regression estimates.
        """
        self.trunc_prob_reg.model(x)

    def predict(self, x: Tensor): 
        """
        Make class predictions with regression estimates.
        """
        with ch.no_grad():
            return sig(self.trunc_prob_reg.model(x)).argmax(dim=-1)

    def defaults(self): 
        """
        Returns the default hyperparamaters for the algorithm.
        """
        return TRUNC_PROB_REG_DEFAULTS

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


class TruncatedProbitRegressionModel(KnownVariance):
    '''
    Truncated logistic regression model to pass into trainer framework.  
    '''
    def __init__(self, args, weight, train_loader, d, k): 
        '''
        Args: 
            args (delphi.utils.helpers.Parameters) : parameter object holding hyperparameters
        '''
        super().__init__(args, train_loader, d)
        if weight is not None:
            self.weight = weight

    def pretrain_hook(self): 
        self.calc_emp_model()
        # projection set radius
        self.radius = self.args.r * (math.sqrt(math.log(1.0 / self.args.alpha)))

        # assign empirical estimates
        self.model.data.requires_grad = True
        self.model.data = self.emp_weight
        self.params = [self.model]

    def calc_emp_model(self): 
        """
        Calculate empirical probit regression estimates using statsmodels module. 
        Probit MLE.
        """
        # empirical estimates for probit regression
        if self.args.fit_intercept: 
            self.emp_prob_reg = Probit(self.y.numpy(), add_constant(self.X.numpy())).fit()
        else: 
            self.emp_prob_reg = Probit(self.y.numpy(), self.X.numpy()).fit()

        self.emp_weight = Tensor(self.emp_prob_reg.params)[...,None]
        self.weight = self.emp_weight.clone()
        
    def __call__(self, batch):
        '''
        Training step for defined model.
        Args: 
            i (int) : gradient step or epoch number
            batch (Iterable) : iterable of inputs that 
        '''
        inp, targ = batch
        pred = inp@self.model
        loss = TruncatedProbitMLE.apply(pred, targ, self.args.phi, self.args.num_samples, self.args.eps)
        prec1, prec5 = accuracy(pred.reshape(pred.size(0), 1), targ.reshape(targ.size(0), 1).float(), topk=(1,))
        return loss, prec1, prec5