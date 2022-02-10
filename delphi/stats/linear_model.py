"""
Linear model class for delphi.
"""

from delphi.utils.helpers import Parameters
import torch as ch
from torch import Tensor
from torch.nn import Linear, Parameter
from sklearn.linear_model import LinearRegression

from .. import delphi
from ..utils.helpers import Bounds


class TruncatedLinearModel(delphi.delphi):
    '''
    Truncated linear model parent class.
    '''
    def __init__(self, 
                args: Parameters, 
                train_loader: ch.utils.data.DataLoader, 
                d=1,
                k=1): 
        '''
        Args: 
            args (cox.utils.Parameters) : parameter object holding hyperparameters
            k (int): number of output logits
        '''
        super().__init__(args)
        self.X, self.y = train_loader.dataset[:]
        self.d = d
        self.k = k
        # calculate empirical estimates for truncated linear model
        self.calc_emp_model()
        self.weight = ch.randn(k, d)
        self.model = Parameter()
        # noise distribution scale
        self.lambda_ = ch.nn.Parameter(ch.ones(1, 1))
        self.base_radius = 1.0

    def pretrain_hook(self):
        # use OLS as empirical estimate to define projection set
        self.radius = self.args.r * self.base_radius
        # empirical estimates for projection set
        self.model.data = self.weight
        # assign empirical estimates
        self.model.requires_grad = True
        # generate noise variance radius bounds if unknown 
        self.var_bounds = Bounds(float(self.noise_var.flatten() / self.args.r), float(self.noise_var.flatten() / Tensor([self.args.alpha]).pow(2))) 
        # assign empirical estimates
        self.lambda_.requires_grad = True if self.args.noise_var is None else False
        self.lambda_.data = self.noise_var.inverse()
        self.params = [{"params": [self.model]},
            {"params": self.lambda_, "lr": self.args.var_lr}]

    def calc_emp_model(self): 
        '''
        Calculates empirical estimates for a truncated linear model. Assigns 
        estimates to a Linear layer. By default calculates OLS for truncated linear regression.
        '''
        self.ols = LinearRegression(fit_intercept=False).fit(self.X, self.y)
        self.weight = Tensor(self.ols.coef_)
        self.noise_var = ch.var(Tensor(self.ols.predict(self.X)) - self.y, dim=0)[..., None]

    """
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
    """

    def post_training_hook(self): 
        self.args.r *= self.args.rate
        # remove model from computation graph
        self.model.requires_grad = False

        self.lambda_.requires_grad = False