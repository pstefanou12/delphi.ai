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

from .truncated_linear_regression import KnownVariance
from .stats import stats
from ..oracle import oracle
from ..grad import TruncatedProbitMLE
from ..trainer import Trainer
from ..utils.datasets import make_train_and_val
from ..utils.helpers import Bounds, Parameters, check_and_fill_args, accuracy


# CONSTANTS 
DEFAULTS = {
        'phi': (oracle, 'required'), 
        'alpha': (float, 'required'), 
        'epochs': (int, 1),
        'fit_intercept': (bool, True), 
        'trials': (int, 3),
        'val': (float, .2),
        'lr': (float, 1e-1), 
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
        'tol': (float, 1e-3),
        'workers': (int, 0),
        'num_samples': (int, 10),
        'early_stopping': (bool, False), 
        'n_iter_no_change': (int, 5),
        'verbose': (bool, False),
}


class TruncatedProbitRegression(stats):
    """
    Truncated Probit Regression supports both binary classification, when the noise distribution in the latent variable model in N(0, 1).
    """
    def __init__(self,
                args: Parameters, 
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
        self.args = check_and_fill_args(args, DEFAULTS)
                
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

        self.train_loader_, self.val_loader_ = make_train_and_val(self.args, X, y) 

        self.trunc_prob_reg = TruncatedProbitRegressionModel(self.args, self.train_loader_)
        trainer = Trainer(self.trunc_prob_reg, self.args, store=self.store) 
        # run PGD for parameter estimation 
        trainer.train_model((self.train_loader_, self.val_loader_))

        with ch.no_grad():
            self.coef = self.trunc_prob_reg.model.weight.clone()
            self.intercept = self.trunc_prob_reg.model.bias.clone()

    def __call__(self, x: Tensor):
        """
        Calculate probit regression's latent variable, based off of regression estimates.
        """
        with ch.no_grad(): 
            self.trunc_prob_reg.model(x)

    def predict(self, x: Tensor): 
        """
        Make class predictions with regression estimates.
        """
        with ch.no_grad():
            return sig(self.trunc_prob_reg.model(x)).argmax(dim=-1)

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


class TruncatedProbitRegressionModel(KnownVariance):
    '''
    Truncated logistic regression model to pass into trainer framework.  
    '''
    def __init__(self, args, train_loader): 
        '''
        Args: 
            args (delphi.utils.helpers.Parameters) : parameter object holding hyperparameters
        '''
        super().__init__(args, train_loader)

    def pretrain_hook(self): 
        # projection set radius
        self.radius = self.args.r * (math.sqrt(math.log(1.0 / self.args.alpha)))

        # import pdb; pdb.set_trace()  
        # empirical estimates for projection set
        self.w = self.emp_weight 
        if self.args.fit_intercept: 
            self.w = ch.cat([self.emp_weight.flatten(), self.emp_bias])
        
        # assign empirical estimates
        self.model.weight.data = self.emp_weight
        if self.args.fit_intercept:
            self.model.bias.data = self.emp_bias
        self.params = None

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

        if self.args.fit_intercept:
            self.emp_bias = Tensor([self.emp_prob_reg.params[0]])
            self.emp_weight = Tensor(self.emp_prob_reg.params[1:].reshape(1, -1))
        else: 
            self.emp_weight = Tensor(self.emp_prob_reg.params[1:].reshape(1, -1))
        
    def __call__(self, batch):
        '''
        Training step for defined model.
        Args: 
            i (int) : gradient step or epoch number
            batch (Iterable) : iterable of inputs that 
        '''
        inp, targ = batch
        pred = self.model(inp)
        loss = TruncatedProbitMLE.apply(pred, targ, self.args.phi, self.args.num_samples, self.args.eps)
        prec1, prec5 = accuracy(pred.reshape(pred.size(0), 1), targ.reshape(targ.size(0), 1).float(), topk=(1,))
        return loss, prec1, prec5