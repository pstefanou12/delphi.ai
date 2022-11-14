"""
Linear model class for delphi.
"""

from delphi.utils.helpers import Parameters
import torch as ch
from torch import Tensor
from torch.nn import Parameter
from sklearn.linear_model import LinearRegression
import cox
from typing import List

from ..delphi import delphi
from ..utils.helpers import Bounds


class LinearModel(delphi):
    '''
    Truncated linear model parent class.
    '''
    def __init__(self, 
                args: Parameters,
                defaults: dict={},
                store: cox.store.Store=None): 
        '''
        Args: 
            args (cox.utils.Parameters) : parameter object holding hyperparameters
            k (int): number of output logits
        '''
        super().__init__(args, defaults=defaults, store=store)
        self.d, self.k = None, None
        self.base_radius = 1.0

    def pretrain_hook(self):
        self.calc_emp_model()
        # use OLS as empirical estimate to define projection set
        self.radius = self.args.r * self.base_radius
        # empirical estimates for projection set
        # generate noise variance radius bounds if unknown 
        self.var_bounds = Bounds(float(self.emp_noise_var.flatten() / self.args.r), float(self.emp_noise_var.flatten() / Tensor([self.args.alpha]).pow(2))) 
        
        if self.args.noise_var is None:
            lambda_ = self.emp_noise_var.inverse()
            self._parameters = [{"params": [Parameter(self.emp_weight * lambda_)]},
                                {"params": Parameter(lambda_), "lr": self.args.var_lr}]

            self.criterion_params = [ 
                self._parameters[1]["params"], self.args.phi,
                self.args.num_samples, self.args.eps,
            ]
        else:
            self.register_parameter("weight", Parameter(self.emp_weight))

    def calc_emp_model(self): 
        '''
        Calculates empirical estimates for a truncated linear model. Assigns 
        estimates to a Linear layer. By default calculates OLS for truncated linear regression.
        '''
        X, y = self.train_loader_.dataset.tensors
        self.ols = LinearRegression(fit_intercept=False).fit(X, y)

        self.register_buffer('emp_noise_var', ch.var(Tensor(self.ols.predict(X)) - y, dim=0)[..., None])
        self.register_buffer('emp_weight', Tensor(self.ols.coef_).T)

    def post_training_hook(self): 
        if self.args.r is not None: self.args.r *= self.args.rate
        # remove model from computation graph
        if self.args.noise_var is None:
            self.weight = self._parameters[0]['params'][0].data 
            self.lambda_ = self._parameters[1]['params'][0].data
            self.lambda_.requires_grad = False
        self.weight.requires_grad = False