"""
Linear model class for delphi.
"""

from delphi.utils.helpers import Parameters
import torch as ch
from torch import Tensor
from torch.nn import Parameter
from sklearn.linear_model import LinearRegression
import cox

from ..delphi import delphi
from ..utils.helpers import Bounds


class LinearModel(delphi):
    '''
    Truncated linear model parent class.
    '''
    def __init__(self, 
                args: Parameters,
                defaults: dict={},
                store: cox.store.Store=None,
                d: int=1,
                k: int=1): 
        '''
        Args: 
            args (cox.utils.Parameters) : parameter object holding hyperparameters
            k (int): number of output logits
        '''
        super().__init__(args, defaults=defaults, store=store)
        self.d = d
        self.k = k
        self.base_radius = 1.0

    def pretrain_hook(self):
        self.calc_emp_model()
        # use OLS as empirical estimate to define projection set
        self.radius = self.args.r * self.base_radius
        # empirical estimates for projection set
        # assign empirical estimates
        self.weight.requires_grad = True
        # generate noise variance radius bounds if unknown 
        self.var_bounds = Bounds(float(self.noise_var.flatten() / self.args.r), float(self.noise_var.flatten() / Tensor([self.args.alpha]).pow(2))) 
        
        if self.args.noise_var is None:
            self._parameters = [{"params": [Parameter(self.weight)]},
                                {"params": Parameter(self.lambda_.data), "lr": self.args.var_lr}]

            self.criterion_params = [ 
                self._parameters[1]["params"], self.args.phi,
                self.args.num_samples, self.args.eps,
            ]

    def calc_emp_model(self): 
        '''
        Calculates empirical estimates for a truncated linear model. Assigns 
        estimates to a Linear layer. By default calculates OLS for truncated linear regression.
        '''
        import pdb; pdb.set_trace()
        X, y = self.train_loader_.dataset.tensors
        self.ols = LinearRegression(fit_intercept=False).fit(X, y)
        self.emp_weight = Tensor(self.ols.coef_).T
        self.noise_var = ch.var(Tensor(self.ols.predict(X)) - y, dim=0)[..., None]

        self.register_parameter('lambda_', Parameter(self.noise_var.inverse()))
        if self.args.noise_var is None:
            self.register_parameter('weight', Parameter(self.emp_weight * self.lambda_))
        else: 
             self.register_parameter('weight', Parameter(self.emp_weight))

    def post_training_hook(self): 
        if self.args.r is not None: self.args.r *= self.args.rate
        # remove model from computation graph
        if self.args.noise_var is None:
            self._parameters[0]['params'][0].requires_grad = False
            self.weight = self._parameters[0]['params'][0].data 
            self.parameters[1]['params'][0].requires_grad = False
            self.lambda_ = self._parameters[1]['params'][0].data
        else: 
            self.weight.requires_grad = False
            self.lambda_.requires_grad = False