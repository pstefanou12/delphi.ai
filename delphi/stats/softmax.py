'''
Multinomial logistic regression that uses softmax loss function.
'''

# distribution tests 

import torch as ch
from torch import Tensor
from torch.nn import Softmax, CrossEntropyLoss
import warnings

from delphi.trainer import Trainer
from delphi.utils.helpers import Parameters
from delphi.utils.datasets import make_train_and_val
from .linear_model import LinearModel

# CONSTANT
softmax = Softmax(dim=1)
ce = CrossEntropyLoss()


class SoftmaxRegression(LinearModel):
    '''
    Truncated logistic regression model to pass into trainer framework.
    '''
    def __init__(self, 
                 args: Parameters, 
                 fit_intercept: bool=True):
        '''
        Args:
            args (cox.utils.Parameters) : parameter object holding hyperparameters
        '''
        super().__init__(args, dependent=False)
        self.fit_intercept = fit_intercept
        del self.criterion

        self.criterion = ce

        self.d, self.k = None, None

    def fit(self, X, y): 
        '''
        Trains model on given data.
        Args:
            X (Tensor) : input data
            y (Tensor) : target data
        '''
        assert isinstance(X, Tensor), "X is type: {}. expected type torch.Tensor.".format(type(X))
        assert isinstance(y, Tensor), "y is type: {}. expected type torch.Tensor.".format(type(y))
        assert X.size(0) >  X.size(1), "number of dimensions, larger than number of samples. procedure expects matrix with size num samples by num feature dimensions." 
        assert X.size(0) == y.size(0), 'number of samples in X and y is unequal. X has {} samples, and y has {} samples'.format(X.size(0), y.size(0))
        assert y.dim() == 2 and y.size(1), f"y is size: {y.size()}. expecting y tensor with size num_samples by 1." 

        y = y.flatten() 
        unique_classes = len(ch.unique(y))
        assert unique_classes > 1, "y contains only 1 unique class. 2+ uniques classes are required for classification procedures"

        # add one feature to x when fitting intercept
        if self.fit_intercept:
            X = ch.cat([X, ch.ones(X.size(0), 1)], axis=1)

        self.d, self.k = X.size(1), unique_classes
        self.train_loader, self.val_loader = make_train_and_val(self.args, X, y) 

        self.trainer = Trainer(self, self.args) 
        self.trainer.train_model(self.train_loader, 
                                 self.val_loader)

        return self

    def pretrain_hook(self,
                      train_loader: ch.utils.data.DataLoader):
        weight = ch.nn.Parameter(ch.randn(self.k, self.d))
        self.register_parameter("weight", weight)
        
    def predict(self, x): 
        with ch.no_grad():
            return softmax(x@self.best_coef.T).argmax(dim=-1)

    def __call__(self, X, y):
        '''
        Training step for defined model.
        Args:
            batch (Iterable) : iterable of inputs that
        '''
        return X@self.weight.T
    
    def post_training_hook(self):
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