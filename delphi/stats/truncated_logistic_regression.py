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

from .linear_model import LinearModel
from .stats import stats
from ..grad import TruncatedBCE, TruncatedCE
from ..trainer import Trainer
from ..utils.datasets import make_train_and_val
from ..utils.helpers import Parameters, accuracy, logistic
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
            weight: ch.Tensor=None,
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

        assert weight is None or weight.dim() == 2, 'weight is size: {}. expecting two dims, with size 1 * d'.format(weight.size())
        self.weight = weight

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
        if self.args.multi_class == 'ovr': 
            self.trunc_log_reg = TruncatedLogisticRegressionModel(self.args, self.weight, self.train_loader_, X.size(1), k)
        else:
            self.trunc_log_reg = TruncatedMultinomialLogisticRegressionModel(self.args, self.weight, self.train_loader_, X.size(1), k)

        trainer = Trainer(self.trunc_log_reg, self.args, store=self.store) 
        # run PGD for parameter estimation 
        trainer.train_model((self.train_loader_, self.val_loader_))

        self.coef = self.trunc_log_reg.model.data[:]
        if self.args.fit_intercept: 
            self.intercept = self.coef[-1]
            self.coef = self.coef[:-1]
        return self

    def __call__(self, x: Tensor):
        """
        Calculate logistic regression's latent variable, based off of regression estimates.
        """
        if self.args.fit_intercept: 
            x = ch.cat([x, ch.ones(x.size(0), 1)], axis=1)
        return x@self.trunc_log_reg.model

    def predict(self, x: Tensor): 
        """
        Make class predictions with regression estimates.
        """
        if self.args.fit_intercept:
            stacked = (ch.cat([x, ch.ones(x.size(0), 1)], axis=1)@self.trunc_log_reg.model).repeat(self.args.num_samples, 1, 1)
        else: 
            stacked = (x@self.trunc_log_reg.model).repeat(self.args.num_samples, 1, 1)
        if self.args.multi_class == 'multinomial':
            noised = stacked + G.sample(stacked.size())
            return noised.mean(0).argmax(-1)
        noised = stacked + logistic.sample(stacked.size())
        return noised.mean(0) > 0

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


class TruncatedLogisticRegressionModel(LinearModel):
    '''
    Truncated logistic regression model to pass into trainer framework.  
    '''
    def __init__(self, args, weight, train_loader, d, k): 
        '''
        Args: 
            args (delphi.utils.helpers.Parameters) : parameter object holding hyperparameters
        '''
        super().__init__(args, d, k)
        if weight is not None:
            self.weight = weight
        self.X, self.y = train_loader.dataset[:]
        self.base_radius = math.sqrt(math.log(1.0 / self.args.alpha))

    def pretrain_hook(self): 
        """
        SkLearn sets up multinomial classification differently. So when doing 
        multinomial classification, we initialize with random estimates.
        """
        # calculate empirical estimates for truncated linear model
        self.calc_emp_model()
        self.radius = self.args.r * self.base_radius
        # empirical estimates for projection set
        self.model.data = self.weight
        # assign empirical estimates
        self.model.requires_grad = True
        # assign empirical estimates
        self.params = [self.model]
        
    def calc_emp_model(self): 
        """
        Calculate empirical logistic regression estimates using SKlearn module.
        """
        # empirical estimates for logistic regression
        self.log_reg = LogisticRegression(penalty='none', fit_intercept=False, multi_class=self.args.multi_class)
        self.log_reg.fit(self.X, self.y.flatten())
        self.emp_weight = Tensor(self.log_reg.coef_).T
        self.weight = self.emp_weight.clone()
    
    def __call__(self, batch):
        '''
        Training step for defined model.
        Args: 
            i (int) : gradient step or epoch number
            batch (Iterable) : iterable of inputs that 
        '''
        # import pdb; pdb.set_trace()
        inp, targ = batch
        z = inp@self.model
        loss = TruncatedBCE.apply(z, targ, self.args.phi, self.args.num_samples, self.args.eps)
        # calculate precision accuracies 
        prec1, _ = accuracy(z, targ.reshape(targ.size(0), 1).float(), topk=(1,))
        return loss, prec1, None

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


class TruncatedMultinomialLogisticRegressionModel(LinearModel):
    '''
    Truncated multinomial logistic regression model to pass into trainer framework.  
    '''
    def __init__(self, args, weight, train_loader, d, k): 
        '''
        Args: 
            args (delphi.utils.helpers.Parameters) : parameter object holding hyperparameters
        '''
        super().__init__(args, d, k)
        if weight is not None:
            assert weight.size() == ch.Size([d, k]), "input weight must be size d - num_features by k - num_logits"
            self.weight = weight
        self.X, self.y = train_loader.dataset[:]
        self.base_radius = math.sqrt(math.log(1.0 / self.args.alpha))

    def pretrain_hook(self): 
        """
        SkLearn sets up multinomial classification differently. So when doing 
        multinomial classification, we initialize with random estimates.
        """
        # calculate empirical estimates for truncated linear model
        self.calc_emp_model()
        self.radius = self.args.r * self.base_radius
        # empirical estimates for projection set
        self.model.data = self.weight
        # assign empirical estimates
        self.model.requires_grad = True
        # assign empirical estimates
        self.params = [self.model]
        
    def calc_emp_model(self): 
        """
        Calculate empirical logistic regression estimates using SKlearn module.
        """
        # randomly assign initial estimates
        if self.weight is None:
            # temp = ch.nn.Linear(in_features=self.d, out_features=self.k)
            self.weight = self.weight = ch.randn(self.d, self.k)
    
    def __call__(self, batch):
        '''
        Training step for defined model.
        Args: 
            i (int) : gradient step or epoch number
            batch (Iterable) : iterable of inputs that 
        '''
        inp, targ = batch
        z = inp@self.model
        loss = TruncatedCE.apply(z, targ, self.args.phi, self.args.num_samples, self.args.eps)
        # calculate precision accuracies 
        prec1, prec5 = None, None
        if z.size(1) >= 5:
            prec1, prec5 = accuracy(z, targ, topk=(1, 5))
        else: 
            prec1, = accuracy(z, targ, topk=(1,))
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