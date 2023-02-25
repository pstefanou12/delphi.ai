"""
Linear model class for delphi.
"""

from delphi.utils.helpers import Parameters
import torch as ch
from torch import Tensor
from torch.nn import Parameter
from sklearn.linear_model import LinearRegression
import cox
from scipy.linalg import lstsq

from ..delphi import delphi
from ..utils.helpers import Bounds


class LinearModel(delphi):
    '''
    Truncated linear model parent class.
    '''
    def __init__(self, 
                args: Parameters,
                dependent: bool,
                emp_weight=None,
                defaults: dict={},
                store: cox.store.Store=None): 
        '''
        Args: 
            args (cox.utils.Parameters) : parameter object holding hyperparameters
            k (int): number of output logits
        '''
        super().__init__(args, defaults=defaults, store=store)
        self._emp_weight = emp_weight
        self.register_buffer('emp_weight', self._emp_weight)
        self.d, self.k = None, None
        self.base_radius = 1.0
        self.dependent = dependent

        self.s = self.args.c_s * (ch.sqrt(ch.log(Tensor([1/self.args.alpha]))) + 1)

    def calc_emp_model(self, train_loader): 
        '''
        Calculates empirical estimates for a truncated linear model. Assigns 
        estimates to a Linear layer. By default calculates OLS for truncated linear regression.
        '''
        X, y = train_loader.dataset.tensors
        if self._emp_weight is None: 
            coef_, _, self.rank_, self.singular_ = lstsq(X, y)
            self.ols_coef_ = Tensor(coef_)
            # self.ols = LinearRegression(fit_intercept=False).fit(X, y)
            # self.register_buffer('emp_noise_var', ch.var(Tensor(self.ols.predict(X)) - y, dim=0)[..., None])
            self.register_buffer('emp_noise_var', ch.var(Tensor(X@coef_) - y, dim=0)[..., None])
            # import pdb; pdb.set_trace()
            # self.register_buffer('emp_weight', Tensor(self.ols.coef_))
            self.register_buffer('emp_weight', Tensor(coef_))
        else: 
            self.register_buffer('emp_weight', self._emp_weight)

        if self.dependent:
            calc_sigma_0 = lambda X: ch.bmm(X.view(X.size(0), X.size(1), 1), \
                        X.view(X.size(0), 1, X.size(1))).sum(0)
            XXT_sum = calc_sigma_0(X)
            Sigma_0 = ((1 / (self.s * len(X))) * XXT_sum)
            self.register_buffer('Sigma_0', Sigma_0)
            assert ch.det(self.Sigma_0) != 0, 'Sigma_0 is singular and non-invertible'
            self.register_buffer('Sigma', self.Sigma_0.clone())

    def pretrain_hook(self, train_loader):
        self.calc_emp_model(train_loader)
        # use OLS as empirical estimate to define projection set
        self.radius = self.args.r * self.base_radius
        # empirical estimates for projection set
        # generate noise variance radius bounds if unknown 
        if self.args.noise_var is None:
            self.var_bounds = Bounds(float(self.emp_noise_var.flatten() / self.args.r), float(self.emp_noise_var.flatten() / Tensor([self.args.alpha]).pow(2))) 
        
        if self.args.noise_var is None:
            lambda_ = self.emp_noise_var.clone().inverse()
            self._parameters = [{"params": [Parameter(self.emp_weight.clone() * lambda_)]},
                                {"params": Parameter(lambda_), "lr": self.args.var_lr}]

            self.criterion_params = [ 
                self._parameters[1]["params"], self.args.phi,
                self.args.num_samples, self.args.eps,
            ]
        else:
            self.register_parameter("weight", Parameter(self.emp_weight.clone()))

    def iteration_hook(self, i, is_train, loss, batch):
        if not self.args.constant: self.schedule.step()

    def post_training_hook(self): 
        if self.args.r is not None: self.args.r *= self.args.rate
        # remove model from computation graph
        if self.args.noise_var is None:
            self.weight = self._parameters[0]['params'][0].data
            self.lambda_ = self._parameters[1]['params'][0].data
            self.lambda_.requires_grad = False

        self.weight.requires_grad = False
        self.emp_weight /= self.beta

    @property
    def ols_coef_(self): 
        """
        OLS empirical estimates for coefficients.
        """
        return self._ols_coef_.T

    @ols_coef_.setter
    def ols_coef_(self, value): 
        self._ols_coef_ = value