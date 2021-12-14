"""
Linear model class for delphi.
"""

import torch as ch
from torch import Tensor
from torch.nn import Linear
from sklearn.linear_model import LinearRegression

from .. import delphi


class TruncatedLinearModel(delphi.delphi):
    '''
    Truncated linear model parent class.
    '''
    def __init__(self, args, train_loader, k=1): 
        '''
        Args: 
            args (cox.utils.Parameters) : parameter object holding hyperparameters
            k (int): number of output logits
        '''
        super().__init__(args)
        self.X, self.y = train_loader.dataset[:]
        self.k = k
        # calculate empirical estimates for truncated linear model
        self.calc_emp_model()
        self.model = Linear(in_features=self.X.size(1), out_features=self.k, bias=self.args.fit_intercept)
        # noise distribution scale
        self.lambda_ = ch.nn.Parameter(ch.ones(1, 1))
        self.base_radius = 1.0

    def calc_emp_model(self): 
        '''
        Calculates empirical estimates for a truncated linear model. Assigns 
        estimates to a Linear layer. By default calculates OLS for truncated linear regression.
        '''
        self.emp_model = LinearRegression(fit_intercept=self.args.fit_intercept).fit(self.X, self.y)
        self.emp_weight = Tensor(self.emp_model.coef_)
        if self.args.fit_intercept:
            self.emp_bias = Tensor(self.emp_model.intercept_)
        self.emp_var = ch.var(Tensor(self.emp_model.predict(self.X)) - self.y, dim=0)[..., None]

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
        if self.args.fit_intercept: 
            temp_w = ch.cat([self.model.weight.flatten(), self.model.bias])
            w_diff = temp_w - self.w
            w_diff = w_diff[None, ...].renorm(p=2, dim=0, maxnorm=self.radius)
            self.model.weight.data, self.model.bias.data = self.emp_weight + w_diff[:,:-1], self.emp_bias + w_diff[:,-1]
        else: 
            w_diff = self.model.weight - self.w
            w_diff = w_diff.renorm(p=2, dim=0, maxnorm=self.radius)
            self.model.weight.data = self.emp_weight + w_diff 
