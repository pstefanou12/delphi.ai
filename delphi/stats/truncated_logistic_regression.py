"""
Truncated Logistic Regression.
"""
import torch as ch
from torch import Tensor
from torch.nn import Softmax, Sigmoid
from torch.distributions import Gumbel
from torch import sigmoid as sig
from sklearn.linear_model import LogisticRegression
import warnings
import math
from typing import Callable

from .linear_model import LinearModel
from ..trainer import Trainer
from ..grad import TruncatedBCE, TruncatedCE
from ..utils.datasets import make_train_and_val
from ..utils.helpers import Parameters, accuracy 
from ..utils.defaults import check_and_fill_args, TRUNC_LOG_REG_DEFAULTS


# CONSTANTS 
softmax = Softmax(dim=0)
sig = Sigmoid()
OVR = "ovr"
MULTI = "multinomial"
CLASSIFICATION_PROCEDURES = [OVR, MULTI]


class TruncatedLogisticRegression(LinearModel):
    """
    Truncated Logistic Regression supports both binary cross entropy classification and truncated cross entropy classification.
    """
    def __init__(self,
                 args: Parameters,
                 phi: Callable, 
                 alpha: float, 
                 fit_intercept: bool=True,
                 multi_class: str="ovr",
                 emp_weight: ch.Tensor=None,
                 rand_seed: int=0):
        '''
        Args: 
            phi (delphi.oracle.oracle) : oracle object for truncated regression model 
            alpha (float) : survival probability for truncated regression model
            fit_intercept (bool) : boolean indicating whether to fit a intercept or not 
        '''
        args = check_and_fill_args(args, TRUNC_LOG_REG_DEFAULTS)
        super().__init__(args, False, emp_weight=emp_weight)
        self.phi = phi 
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        assert multi_class in CLASSIFICATION_PROCEDURES, f"{multi_class} not in {CLASSIFICATION_PROCEDURES}"
        self.multi_class = multi_class
        self.rand_seed = rand_seed        

        del self.criterion
        del self.criterion_params
        if self.multi_class == OVR:
            self.criterion = TruncatedBCE.apply
        else: 
            self.criterion = TruncatedCE.apply
        self.criterion_params = [self.phi, self.args.num_samples, self.args.eps]

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
        assert y.dim() == 2 and y.size(1), f"y is size: {y.size()}. expecting y tensor with size num_samples by 1." 

        unique_classes = len(ch.unique(y))
        assert unique_classes > 1, "y contains only 1 unique class. 2+ uniques classes are required for classification procedures"
        if self.multi_class == OVR:
            k = 1
        elif self.multi_class == MULTI: 
            k = unique_classes

        # add one feature to x when fitting intercept
        if self.fit_intercept:
            X = ch.cat([X, ch.ones(X.size(0), 1)], axis=1)

        self.train_loader, self.val_loader = make_train_and_val(self.args, X, y) 

        self.trainer = Trainer(self, self.args) 
        self.trainer.train_model(self.train_loader, 
                                 self.val_loader)

        return self
    
    def pretrain_hook(self,
                      train_loader: ch.utils.data.DataLoader): 
        """
        SkLearn sets up multinomial classification differently. So when doing 
        multinomial classification, we initialize with random estimates.
        """
        # calculate empirical estimates for truncated linear model
        self.calc_emp_model(train_loader)
        self.radius = self.args.r * self.base_radius
        
    def calc_emp_model(self, 
                       train_loader: ch.utils.data.DataLoader): 
        """
        Calculate empirical logistic regression estimates using SKlearn module.
        """
        X, y = train_loader.dataset.tensors
        if self.emp_weight is None:
            log_reg = LogisticRegression(penalty=None, fit_intercept=False, multi_class=self.multi_class)
            log_reg.fit(X, y.flatten())
            self.emp_weight = ch.nn.Parameter(ch.from_numpy(log_reg.coef_).float())
        else: 
            self.emp_weight = ch.nn.Parameter(self.emp_weight)
        self.register_parameter("weight", self.emp_weight)
    
    def __call__(self, X, y):
        '''
        Training step for defined model.
        Args: 
            i (int) : gradient step or epoch number
            batch (Iterable) : iterable of inputs that 
        '''
        return X@self.weight.T

    def iteration_hook(self, i, loop_type, loss, batch):
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
        best_params = self.trainer.best_params.reshape(self.emp_weight.size())
        final_params = self.trainer.final_params.reshape(self.emp_weight.size())
        if self.fit_intercept:
            self.best_coef = best_params[:,:-1]
            self.best_intercept = best_params[:,-1]

            self.final_coef = final_params[:,:-1]
            self.final_intercept = final_params[:,-1]

        else:
            self.best_coef = best_params
            self.final_coef = final_params

    def predict_proba(self, X: Tensor): 
        """
        Probability predictions for input features.
        """
        if self.fit_intercept:
            logits = ch.cat([X, ch.ones(X.size(0), 1)], axis=1)@self.weight.T
        else: 
            logits = X@self.weight.T
        if self.multi_class == MULTI:
            return  softmax(logits)
        return sig(logits)

    def predict(self, X: Tensor): 
        """
        Class predictions for input features.
        """
        prob_predictions = self.predict_proba(X)
        if prob_predictions.size(-1) > 1: 
            return prob_predictions.argmax(-1)
        return prob_predictions > .5

    @property
    def best_coef_(self): 
        return self.best_coef

    @property
    def best_intercept_(self): 
        if self.fit_intercept:
            return self.best_intercept
        warnings.warn("intercept not fit, check inputs.") 
    
    @property
    def final_coef_(self): 
        return self.final_coef
    
    @property
    def final_intercept_(self): 
        if self.fit_intercept:
            return self.final_intercept
        warnings.warn("intercept not fit, check inputs.") 

    @property
    def coef_(self): 
        """
        Regression weight.
        """
        return self.best_coef

    @property
    def intercept_(self): 
        """
        Regression intercept.
        """
        if self.fit_intercept:
            return self.best_intercept
        warnings.warn('intercept not fit, check args input.') 


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
        self.base_radius = math.sqrt(math.log(1.0 / self.alpha))

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