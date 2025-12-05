"""
Truncated Logistic Regression.
"""

import torch as ch
from torch import Tensor
from statsmodels.discrete.discrete_model import Probit
import math
import warnings
from typing import Callable

from .linear_model import LinearModel
from ..delphi_logger import delphiLogger
from ..grad import TruncatedProbitMLE
from ..trainer import Trainer
from ..utils.datasets import make_train_and_val
from ..utils.helpers import Parameters, accuracy
from ..utils.defaults import check_and_fill_args, TRAINER_DEFAULTS, DELPHI_DEFAULTS, TRUNC_PROB_REG_DEFAULTS


class TruncatedProbitRegression(LinearModel):
    """
    Truncated Probit Regression supports both binary classification, when the noise distribution in the latent variable model in N(0, 1).
    """
    def __init__(self,
                args: Parameters,
                phi: Callable,
                alpha: float, 
                fit_intercept: bool=True,
                emp_weight: ch.Tensor=None, 
                rand_seed: int=0):
        '''
        Args: 
            phi (delphi.oracle.oracle) : oracle object for truncated regression model 
            alpha (float) : survival probability for truncated regression model 
        '''
        logger = delphiLogger()
        args = check_and_fill_args(args, TRUNC_PROB_REG_DEFAULTS)
        super().__init__(args, False, logger, emp_weight=emp_weight)
        self.phi = phi 
        self.alpha = alpha 
        self.fit_intercept = fit_intercept
        self.rand_seed = rand_seed

        del self.criterion
        del self.criterion_params
        self.criterion = TruncatedProbitMLE.apply 
        self.criterion_params = [self.phi, self.args.num_samples, self.args.eps]

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
        if self.fit_intercept:
            X = ch.cat([X, ch.ones(X.size(0), 1)], axis=1)

        k = 1
        self.train_loader, self.val_loader = make_train_and_val(self.args, X, y) 

        self.trainer = Trainer(self, self.args, self.logger) 
        # run PGD for parameter estimation 
        self.trainer.train_model(self.train_loader, 
                                 self.val_loader)

        return self

    def pretrain_hook(self, 
                      train_loader: ch.utils.data.DataLoader): 
        self.calc_emp_model()
        # projection set radius
        self.radius = self.args.r * (math.sqrt(math.log(1.0 / self.alpha)))

    def calc_emp_model(self): 
        """
        Calculate empirical probit regression estimates using statsmodels module. 
        Probit MLE.
        """
        if self.emp_weight is None:
            X, y = self.train_loader.dataset.tensors
        
            # empirical estimates for probit regression
            self.emp_prob_reg = Probit(y.numpy(), X.numpy()).fit()

            self.emp_weight = ch.nn.Parameter(ch.from_numpy(self.emp_prob_reg.params)[...,None].float())
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
        return X@self.weight
    
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
        best_params = self.trainer.best_params
        final_params = self.trainer.final_params
        if self.fit_intercept:
            self.best_coef = best_params[:,:-1]
            self.best_intercept = best_params[:,-1]

            self.final_coef = final_params[:,:-1]
            self.final_intercept = final_params[:,-1]

        else:
            self.best_coef = best_params
            self.final_coef = final_params

    def predict(self, X: Tensor): 
        """
        Make class predictions with regression estimates.
        """
        if self.fit_intercept:
            logits = ch.cat([X, ch.ones(X.size(0), 1)], axis=1)@self.weight
        else: 
            logits = X@self.weight
        return 0.5 * (1 + ch.erf(logits / ch.sqrt(ch.Tensor([2.0])))) > .5
        
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