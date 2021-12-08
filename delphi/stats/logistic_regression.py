"""
Truncated Logistic Regression.
"""

import torch as ch
from torch import Tensor
from torch.nn import CrossEntropyLoss, Softmax, Sigmoid
from torch.distributions import Gumbel
from torch import sigmoid as sig
from cox.store import Store
from sklearn.linear_model import LogisticRegression
import copy
import cox

from .. import delphi
from .linear_model import TruncatedLinearModel
from .stats import stats
from ..oracle import oracle
from ..grad import TruncatedBCE, TruncatedCE
from ..trainer import Trainer
from ..utils.datasets import make_train_and_val
from ..utils.helpers import Bounds, Parameters, check_and_fill_args, accuracy


# CONSTANTS 
G = Gumbel(0, 1)
softmax = Softmax()
sig = Sigmoid()
DEFAULTS = {
        'phi': (oracle, 'required'),
        'alpha': (float, 'required'), 
        'epochs': (int, 1),
        'fit_intercept': (bool, True), 
        'num_trials': (int, 3),
        'clamp': (bool, True), 
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
        'multi_class': ({'multinomial', 'ovr'}, 'ovr'),
        'early_stopping': (bool, False), 
        'n_iter_no_change': (int, 5),
        'verbose': (bool, False),
}


class TruncatedLogisticRegression(stats):
    """
    Truncated Logistic Regression supports both binary cross entropy classification and truncated cross entropy classification.
    """
    def __init__(self,
            args: dict,
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
        self.store = store
        self.trunc_log_reg = None
        # algorithm hyperparameters
        self.args = check_and_fill_args(Parameters(args), DEFAULTS)
                
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
        if self.args.multi_class == 'ovr':
            assert y.dim() == 2 and y.size(1) == 1, "y is size: {}. expecting y tensor with size num_samples by 1.".format(y.size()) 
        else: 
             assert y.dim() == 1, "y is size: {}. expecting y tensor with size num_samples by 1.".format(y.size()) 


        self.train_loader_, self.val_loader_ = make_train_and_val(self.args, X, y) 

        self.trunc_log_reg = TruncatedLogisticRegressionModel(self.args, self.train_loader_, len(ch.unique(y)))

        trainer = Trainer(self.trunc_log_reg, max_iter=self.args.epochs, trials=self.args.num_trials, tol=self.args.tol, store=self.store, verbose=self.args.verbose) 
        # run PGD for parameter estimation 
        trainer.train_model((self.train_loader_, self.val_loader_))

        with ch.no_grad():
            self.coef = self.trunc_log_reg.model.weight.clone()
            self.intercept = self.trunc_log_reg.model.bias.clone()


    def __call__(self, x: Tensor):
        """
        Calculate logistic regression's latent variable, based off of regression estimates.
        """
        with ch.no_grad(): 
            self.trunc_log_reg.model(x)

    def predict(self, x: Tensor): 
        """
        Make class predictions with regression estimates.
        """
        with ch.no_grad():
            if self.args.multi_class == 'multinomial':
                return softmax(self.trunc_log_reg.model(x)).argmax(dim=-1)
            return (sig(self.trunc_log_reg.model(x)) > .5).float()

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


class TruncatedLogisticRegressionModel(TruncatedLinearModel):
    '''
    Truncated logistic regression model to pass into trainer framework.  
    '''
    def __init__(self, args, train_loader, k): 
        '''
        Args: 
            args (delphi.utils.helpers.Parameters) : parameter object holding hyperparameters
        '''
        super().__init__(args, train_loader, k)

    def pretrain_hook(self): 
        # projection set radius
        self.radius = self.args.r * (ch.sqrt(2.0 * ch.log(1.0 / Tensor([self.args.alpha]))))
        if self.args.clamp:
            self.weight_bounds = Bounds((self.emp_weight - self.radius).flatten(),
                                        (self.emp_weight + self.radius).flatten())
            if self.emp_log_reg.intercept_:
                self.bias_bounds = Bounds(float(self.emp_bias - self.radius),
                                          float(self.emp_bias + self.radius))
        else: 
            pass

        """
        SkLearn sets up multinomial classification differenlty. So when doing 
        multinomial classification, we initialize with random estimates.
        """
        # assign empirical estimates
        if self.args.multi_class == 'ovr':
            self.model.weight.data = self.emp_weight
            self.model.bias.data = self.emp_bias
        update_params = None

    def calc_emp_model(self): 
        """
        Calculate empirical logistic regression estimates using SKlearn module.
        """
        # empirical estimates for logistic regression
        self.emp_log_reg = LogisticRegression(penalty='none', fit_intercept=self.args.fit_intercept, multi_class=self.args.multi_class)
        self.emp_log_reg.fit(self.X, self.y.flatten())
        self.emp_weight = Tensor(self.emp_log_reg.coef_)
        if self.args.fit_intercept:
            self.emp_bias = Tensor(self.emp_log_reg.intercept_)

    def __call__(self, batch):
        '''
        Training step for defined model.
        Args: 
            i (int) : gradient step or epoch number
            batch (Iterable) : iterable of inputs that 
        '''
        inp, targ = batch

        z = self.model(inp)
        if self.args.multi_class == 'multinomial': 
            loss = TruncatedCE.apply(z, targ, self.args.phi, self.args.num_samples, self.args.eps)
            pred = z.argmax(-1)
        elif self.args.multi_class == 'ovr': 
            loss = TruncatedBCE.apply(z, targ, self.args.phi, self.args.num_samples, self.args.eps)
            pred = z >= 0
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
        # project model parameters back to domain 
        if self.args.clamp: 
            if self.args.multi_class: 
               pass 
            else: 
                # project weight coefficients
                self.model.weight.data = ch.stack([ch.clamp(self.model.weight.data[i], float(self.weight_bounds.lower[i]),
                                                             float(self.weight_bounds.upper[i])) for i in
                                                    range(model.weight.size(0))])
                # project bias coefficient
                if self.args.fit_intercept:
                    self.model.bias.data = ch.clamp(self.model.bias, self.bias_bounds.lower, self.bias_bounds.upper).reshape(
                        self.model.bias.size())
        else: 
            pass

    def post_training_hook(self): 
        self.args.r *= self.args.rate


