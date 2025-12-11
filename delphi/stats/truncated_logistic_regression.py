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
from typing import Callable

from .linear_model import LinearModel
from ..delphi_logger import delphiLogger
from ..trainer import Trainer
from ..grad import TruncatedBCE, TruncatedCE, TruncatedCELabels
from ..utils.datasets import make_train_and_val
from ..utils.helpers import Parameters, accuracy 
from ..utils.defaults import check_and_fill_args, TRUNC_LOG_REG_DEFAULTS


# CONSTANTS 
softmax = Softmax(dim=-1)
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
        logger = delphiLogger()
        args = check_and_fill_args(args, TRUNC_LOG_REG_DEFAULTS)
        super().__init__(args, False, logger, emp_weight=emp_weight)
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
            # self.criterion = TruncatedCELabels.apply
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
            self.K = 1
        elif self.multi_class == MULTI: 
            self.K = unique_classes

        # add one feature to x when fitting intercept
        if self.fit_intercept:
            X = ch.cat([X, ch.ones(X.size(0), 1)], axis=1)
        self.D = X.size(1)

        self.train_loader, self.val_loader = make_train_and_val(self.args, X, y) 

        self.trainer = Trainer(self, self.args, self.logger) 
        self.trainer.train_model(self.train_loader, 
                                 self.val_loader)

        return self
    
    def _calc_emp_model(self): 
        """
        Calculate empirical logistic regression estimates using SKlearn module.
        """
        X, y = self.train_loader.dataset.tensors
        if self._emp_weight is None and self.multi_class == "ovr":
            log_reg = LogisticRegression(penalty=None, fit_intercept=False, multi_class=self.multi_class)
            log_reg.fit(X, y.flatten())
            self._emp_weight = ch.from_numpy(log_reg.coef_).float()
        elif self._emp_weight is None: 
            self._emp_weight = ch.randn(self.K, self.D) 
        self.register_parameter("weight", ch.nn.Parameter(self._emp_weight.clone()))
    
    def pretrain_hook(self): 
        """
        SkLearn sets up multinomial classification differently. So when doing 
        multinomial classification, we initialize with random estimates.
        """
        # calculate empirical estimates for truncated linear model
        self._calc_emp_model()
        self.radius = self.args.r * self.base_radius
    
    def forward(self, X):
        '''
        Training step for defined model.
        Args: 
            i (int) : gradient step or epoch number
            batch (Iterable) : iterable of inputs that 
        '''
        return X@self.weight.T
    

    def post_step_hook(self, i, loop_type, loss, batch):
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
        best_params = self.trainer.best_params.reshape(self.weight.size())
        final_params = self.trainer.final_params.reshape(self.weight.size())
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
            logits = ch.cat([X, ch.ones(X.size(0), 1)], axis=1)@self.best_coef.T
        else: 
            logits = X@self.best_coef.T
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