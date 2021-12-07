"""
Linear model class for delphi.
"""

import torch as ch
import torch.linalg as LA
from torch import Tensor
import torch.nn as nn
from torch.nn import Linear
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import LinearRegression

from .. import delphi


class TruncatedLinearModel(delphi.delphi):
    '''
    Truncated linear regression with known noise variance model.
    '''
    def __init__(self, args, train_loader, phi, k=1): 
        '''
        Args: 
            args (cox.utils.Parameters) : parameter object holding hyperparameters
            k (int): number of output logits
        '''
        super().__init__(args)
        self.X, self.y = train_loader.dataset[:]
        self.phi = phi
        self.k = k
        # calculate empirical estimates for truncated linear model
        self.calc_emp_model()
        self.model = Linear(in_features=self.X.size(1), out_features=self.k, bias=self.args.fit_intercept)
        # noise distribution scale
        self.lambda_ = ch.nn.Parameter(ch.ones(1, 1))

    def calc_emp_model(self): 
        '''
        Calculates empirical estiamtes for a truncated linear model. Assigns 
        estimates to a Linear layer. By default calculates OLS for truncated linear regression.
        '''
        self.emp_model = LinearRegression(fit_intercept=self.args.fit_intercept).fit(self.X, self.y)
        self.emp_weight = Tensor(self.emp_model.coef_)
        if self.args.fit_intercept:
            self.emp_bias = Tensor(self.emp_model.intercept_)
        self.emp_var = ch.var(Tensor(self.emp_model.predict(self.X)) - self.y, dim=0)[..., None]

