"""
Truncated Logistic Regression.
"""

import torch as ch
from torch import Tensor
from torch.nn import Softmax, Sigmoid
from torch.distributions import Gumbel
from torch import sigmoid as sig
from sklearn.linear_model import LogisticRegression
import cox
import warnings
import math

from .linear_model import TruncatedLinearModel
from .stats import stats
from ..grad import TruncatedBCE, TruncatedCE
from ..trainer import Trainer
from ..utils.datasets import make_train_and_val
from ..utils.helpers import Parameters, accuracy
from ..utils.defaults import check_and_fill_args, TRAINER_DEFAULTS, DELPHI_DEFAULTS, TRUNC_LOG_REG_DEFAULTS


# CONSTANTS 
G = Gumbel(0, 1)
softmax = Softmax(dim=0)
sig = Sigmoid()


class TruncatedLogisticRegression(stats):
    """
    Truncated Logistic Regression supports both binary cross entropy classification and truncated cross entropy classification.
    """
    def __init__(self,
            args: Parameters,
            store: cox.store.Store=None,
):
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
        super(TruncatedLogisticRegression).__init__()
        # instance variables
        assert isinstance(args, Parameters), "args is type: {}. expecting args to be type delphi.utils.helpers.Parameters"
        assert store is None or isinstance(store, cox.store.Store), "store is type: {}. expecting cox.store.Store.".format(type(store))
        self.store = store
        self.trunc_log_reg = None
        # algorithm hyperparameters
        TRUNC_LOG_REG_DEFAULTS.update(TRAINER_DEFAULTS)
        TRUNC_LOG_REG_DEFAULTS.update(DELPHI_DEFAULTS)
        self.args = check_and_fill_args(args, TRUNC_LOG_REG_DEFAULTS)
                
    def fit(self, X: Tensor, y: Tensor):
        """
        Train truncated logistic regression model by running PSGD on the truncated negative 
        population log likelihood.
        Args: 
            X (torch.Tensor): input feature covariates num_samples by dims
            y (torch.Tensor): dependent variable predictions num_samples by 1
        """
        assert isinstance(X, Tensor), "X is type: {}. expected type torch.Tensor.".format(type(X))
        assert isinstance(y, Tensor), "y is type: {}. expected type torch.Tensor.".format(type(y))
        assert X.size(0) >  X.size(1), "number of dimensions, larger than number of samples. procedure expects matrix with size num samples by num feature dimensions." 
        assert X.size(0) == y.size(0), 'number of samples in X and y is unequal. X has {} samples, and y has {} samples'.format(X.size(0), y.size(0))
        if self.args.multi_class == 'ovr':
            assert y.dim() == 2 and y.size(1) == 1, "y is size: {}. expecting y tensor with size num_samples by 1.".format(y.size()) 
            k = 1
        else: 
            assert y.dim() == 1, "y is size: {}. expecting y tensor with size num_samples.".format(y.size()) 
            k = len(ch.unique(y))
        # add one feature to x when fitting intercept
        if self.args.fit_intercept:
            X = ch.cat([X, ch.ones(X.size(0), 1)], axis=1)

        self.train_loader_, self.val_loader_ = make_train_and_val(self.args, X, y) 

        self.trunc_log_reg = TruncatedLogisticRegressionModel(self.args, self.train_loader_, k, X.size(1))

        trainer = Trainer(self.trunc_log_reg, self.args, store=self.store) 
        # run PGD for parameter estimation 
        trainer.train_model((self.train_loader_, self.val_loader_))

        self.coef = self.trunc_log_reg.model.data[:]
        if self.args.fit_intercept: 
            self.coef = self.trunc_log_reg.model.data[:-1]
            self.intercept = self.trunc_log_reg.model.data[-1]
        return self

    def __call__(self, x: Tensor):
        """
        Calculate logistic regression's latent variable, based off of regression estimates.
        """
        return self.trunc_log_reg.model(x)

    def predict(self, x: Tensor): 
        """
        Make class predictions with regression estimates.
        """
        if self.args.multi_class == 'multinomial':
            return softmax(x@self.trunc_log_reg.model.T).argmax(dim=-1)
        return (sig(x@self.trunc_log_reg.model.T) > .5).float()

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


class TruncatedLogisticRegressionModel(TruncatedLinearModel):
    '''
    Truncated logistic regression model to pass into trainer framework.  
    '''
    def __init__(self, args, train_loader, k, d): 
        '''
        Args: 
            args (delphi.utils.helpers.Parameters) : parameter object holding hyperparameters
        '''
        super().__init__(args, train_loader, k, d)

        self.base_radius = math.sqrt(math.log(1.0 / self.args.alpha))

    def pretrain_hook(self): 
        """
        SkLearn sets up multinomial classification differently. So when doing 
        multinomial classification, we initialize with random estimates.
        """
        self.radius = self.args.r * self.base_radius
        # empirical estimates for projection set
        self.model.data = self.weight.T
        # assign empirical estimates
        self.model.requires_grad = True
        # assign empirical estimates
        self.params = [self.model]
        
    def calc_emp_model(self): 
        """
        Calculate empirical logistic regression estimates using SKlearn module.
        """
        # empirical estimates for logistic regression
        if self.args.multi_class == 'ovr':
            self.log_reg = LogisticRegression(penalty='none', fit_intercept=False, multi_class=self.args.multi_class)
            self.log_reg.fit(self.X, self.y.flatten())
            self.weight = Tensor(self.log_reg.coef_)
    
    def __call__(self, batch):
        '''
        Training step for defined model.
        Args: 
            i (int) : gradient step or epoch number
            batch (Iterable) : iterable of inputs that 
        '''
        inp, targ = batch
        z = inp@self.model.T
        if self.args.multi_class == 'multinomial': 
            loss = TruncatedCE.apply(z, targ, self.args.phi, self.args.num_samples, self.args.eps)
            pred = softmax(z).argmax(-1)
        elif self.args.multi_class == 'ovr': 
            loss = TruncatedBCE.apply(z, targ, self.args.phi, self.args.num_samples, self.args.eps)
            pred = sig(z) > .5
        # calculate precision accuracies 
        if z.size(1) >= 5:
            prec1, prec5 = accuracy(pred.reshape(pred.size(0), 1), targ.reshape(targ.size(0), 1).float(), topk=(1, 5))
        else: 
            prec1, prec5 = accuracy(pred.reshape(pred.size(0), 1), targ.reshape(targ.size(0), 1).float(), topk=(1,))
        return loss, prec1, prec5

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
        pass
    
    def post_training_hook(self): 
        self.args.r *= self.args.rate
        # remove model from computation graph
        self.model.requires_grad = False