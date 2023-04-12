"""
Truncated Linear Regression.
"""

from re import A
import torch as ch
from torch import Tensor
import torch.linalg as LA
from cox.store import Store
import warnings
import collections
from torch.nn import Parameter
from scipy.linalg import lstsq
from typing import List, Callable

from .linear_model import LinearModel
from ..grad import TruncatedMSE, TruncatedUnknownVarianceMSE, SwitchGrad
from ..utils.datasets import make_train_and_val
from ..utils.helpers import Parameters
from .linear_model import LinearModel
from ..trainer import train_model
from ..utils.helpers import Bounds
from ..utils.defaults import TRUNC_REG_DEFAULTS, TRUNC_LDS_DEFAULTS


class TruncatedLinearRegression(LinearModel):
    """
    Truncated linear regression class. Supports truncated linear regression
    with known noise, unknown noise, and confidence intervals. Module uses 
    delphi.trainer.Trainer to train truncated linear regression by performing 
    projected stochastic gradient descent on the truncated population log likelihood. 
    Module requires the user to specify an oracle from the delphi.oracle.oracle class, 
    and the survival probability. 
    """
    def __init__(self,
                phi: Callable,
                args: Parameters,
                noise_var:ch.Tensor=None,
                dependent: bool=False,
                emp_weight: ch.Tensor=None,
                rand_seed=0,
                store: Store=None):
        """
        Args: 
            phi (delphi.oracle.oracle) : oracle object for truncated regression model 
            alpha (float) : survival probability for truncated regression model
            fit_intercept (bool) : boolean indicating whether to fit a intercept or not 
            val (int) : number of samples to use for validation set 
            tol (float) : gradient tolerance threshold 
            workers (int) : number of workers to spawn 
            r (float) : size for projection set radius 
            rate (float): rate at which to increase the size of the projection set, when procedure does not converge - input as a decimal percentage
            num_samples (int) : number of samples to sample in gradient 
            batch_size (int) : batch size
            lr (float) : initial learning rate for regression weight parameters 
            var_lr (float) : initial learning rate to use for variance parameter in the settign where the variance is unknown 
            step_lr (int) : number of gradient steps to take before decaying learning rate for step learning rate 
            custom_lr_multiplier (str) : "cosine" (cosine annealing), "adam" (adam optimizer) - different learning rate schedulers available
            lr_interpolation (str) : "linear" linear interpolation
            step_lr_gamma (float) : amount to decay learning rate when running step learning rate
            momentum (float) : momentum for SGD optimizer 
            eps (float) :  epsilon value for gradient to prevent zero in denominator
            dependent (bool) : boolean indicating whether dataset is dependent and you should run SwitchGrad instead
            store (cox.store.Store) : cox store object for logging 
        """
        if dependent: 
            super().__init__(args, dependent, emp_weight=emp_weight, defaults=TRUNC_LDS_DEFAULTS, store=store)
            self.args.__setattr__('lr', (2/self.args.alpha) ** self.args.c_eta)
        else:    
            super().__init__(args, dependent, emp_weight=emp_weight, defaults=TRUNC_REG_DEFAULTS, store=store)
        self.phi = phi
        self.noise_var = noise_var
        self.rand_seed = rand_seed
        if self.dependent: assert self.noise_var is not None, "if linear dynamical system, noise variance must be known"

        del self.criterion
        del self.criterion_params 
        if self.dependent: 
            self.criterion = SwitchGrad.apply
        elif self.noise_var is None: 
            self.criterion = TruncatedUnknownVarianceMSE.apply
        else: 
            self.criterion = TruncatedMSE.apply
            self.criterion_params = [ 
                self.phi, self.noise_var,
                self.args.num_samples, self.args.eps]

        # property instance variables 
        self.coef, self.intercept = None, None

    def fit(self, 
            X: Tensor, 
            y: Tensor):
        """
        Train truncated linear regression model by running PSGD on the truncated negative 
        population log likelihood.
        Args: 
            X (torch.Tensor): input feature covariates num_samples by dims
            y (torch.Tensor): dependent variable predictions num_samples by 1
        """
        assert isinstance(X, Tensor), "X is type: {}. expected type torch.Tensor.".format(type(X))
        assert isinstance(y, Tensor), "y is type: {}. expected type torch.Tensor.".format(type(y))
        assert X.size(0) >  X.size(1), "number of dimensions, larger than number of samples. procedure expects matrix with size num samples by num feature dimensions." 
        assert y.dim() == 2 and y.size(1) <= X.size(1), "y is size: {}. expecting y tensor to have y.size(1) < X.size(1).".format(y.size()) 
        if self.noise_var is not None:
            assert self.noise_var.size(0) == y.size(1), "noise var size is: {}. y size is: {}. expecting noise_var.size(0) == y.size(1)".format(self.noise_var.size(0), y.size(1))

        # add number of samples to args 
        self.args.__setattr__('T', X.size(0))
        if self.dependent:
            self.criterion_params = [ 
                self.phi, self.args.c_gamma, self.args.alpha, self.args.T, 
                self.noise_var, self.args.num_samples, self.args.eps,
            ]

        # add one feature to x when fitting intercept
        if self.args.fit_intercept:
            X = ch.cat([X, ch.ones(X.size(0), 1)], axis=1)

        # normalize features so that the maximum l_2 norm is 1
        self.beta = ch.ones(1, 1)
        if X.norm(dim=1, p=2).max() > 1 and not self.dependent:  
            l_inf = LA.norm(X, dim=-1, ord=float('inf')).max()
            self.beta = l_inf * (X.size(1) ** .5)

        self.train_loader, self.val_loader = make_train_and_val(self.args, X / self.beta, y)
        best_params, self.history, best_loss = train_model(self.args, self, 
                                                        self.train_loader, self.val_loader, 
                                                        rand_seed=self.rand_seed,
                                                        store=self.store)

        # reparameterize the regression's parameters
        if self.noise_var is None: 
            lambda_ = best_params[1]['params'][0]
            v = best_params[0]['params'][0]
            self.variance = lambda_.inverse()
            self.weight = v * self.variance
        else: 
            self.weight = best_params

        # re-scale coefficients
        self.weight /= self.beta

        # assign results from procedure to instance variables
        if self.args.fit_intercept: 
            self.coef = self.weight[:-1]
            self.intercept = self.weight[-1]
        else: 
            self.coef = self.weight[:]
        return self

    def pretrain_hook(self, 
                    train_loader):
        self.calc_emp_model(train_loader)
        # use OLS as empirical estimate to define projection set
        self.radius = self.args.r * self.base_radius
        # empirical estimates for projection set
        # generate noise variance radius bounds if unknown 
        if self.noise_var is None:
            self.var_bounds = Bounds(float(self.emp_noise_var.flatten() / self.args.r), float(self.emp_noise_var.flatten() / Tensor([self.args.alpha]).pow(2))) 
        
        if self.noise_var is None:
            lambda_ = self.emp_noise_var.clone().inverse()
            self._parameters = [{"params": [Parameter(self.emp_weight.clone() * lambda_)]},
                                {"params": Parameter(lambda_), "lr": self.args.var_lr}]

            self.criterion_params = [ 
                self._parameters[1]["params"], self.phi,
                self.args.num_samples, self.args.eps,
            ]
        else:
            self.register_parameter("weight", Parameter(self.emp_weight.clone()))

    def calc_emp_model(self, 
                        train_loader: ch.utils.data.DataLoader) -> None: 
        '''
        Calculates empirical estimates for a truncated linear model. Assigns 
        estimates to a Linear layer. By default calculates OLS for truncated linear regression.
        '''
        X, y = train_loader.dataset.tensors
        if self._emp_weight is None: 
            coef_, _, self.rank_, self.singular_ = lstsq(X, y)
            self.ols_coef_ = Tensor(coef_)
            self.register_buffer('emp_noise_var', ch.var(Tensor(X@coef_) - y, dim=0)[..., None])
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

    def post_training_hook(self): 
        if self.args.r is not None: self.args.r *= self.args.rate
        # remove model from computation graph
        if self.noise_var is None:
            self.weight = self._parameters[0]['params'][0].data
            self.lambda_ = self._parameters[1]['params'][0].data
            self.lambda_.requires_grad = False

        self.weight.requires_grad = False
        self.emp_weight /= self.beta
    
    def predict(self, 
                X: Tensor): 
        """
        Make predictions with regression estimates.
        """
        assert self.coef is not None, "must fit model before using predict method"
        if self.args.fit_intercept: 
            return X@self.coef + self.intercept
        return X@self.coef

    def nll(self,
                X: Tensor, 
                y: Tensor) -> Tensor:
        with ch.no_grad(): 
            return self.criterion(self.predict(X), y, *self.criterion_params)

    def emp_nll(self, 
                X: Tensor, 
                y: Tensor) -> Tensor:
        if self.args.fit_intercept: 
            X = ch.cat([X, ch.ones(X.size(0), 1)], axis=1)
        with ch.no_grad():
            return self.criterion(X@self.emp_weight, y, *self.criterion_params)
    
    @property
    def coef_(self): 
        """
        Regression coefficient weights.
        """
        return self.coef.clone()

    @property
    def intercept_(self): 
        """
        Regression intercept.
        """
        if self.intercept is not None:
            return self.intercept.clone()
        warnings.warn("intercept not fit, check args input.") 
    
    @property
    def variance_(self): 
        """
        Noise variance prediction for linear regression with
        unknown noise variance algorithm.
        """
        if self.noise_var is None: 
            return self.variance
        else: 
            warnings.warn("no variance prediction because regression with known variance was run")
    
    @property
    def ols_coef_(self): 
        """
        OLS empirical estimates for coefficients.
        """
        return self._ols_coef_

    @ols_coef_.setter
    def ols_coef_(self, value): 
        self._ols_coef_ = value

    @property
    def ols_intercept_(self):
        """
        OLS empirical estimates for intercept.
        """
        return self.trunc_reg.emp_bias.clone()

    @property
    def ols_variance_(self): 
        """
        OLS empirical estimates for noise variance.
        """
        return self.trunc_reg.emp_var.clone()

    def __call__(self,
                X: ch.Tensor,
                y: ch.Tensor) -> ch.Tensor:
        if self.noise_var is None:
            weight = self._parameters[0]['params'][0]
            lambda_ = self._parameters[1]['params'][0]
            return X@weight * lambda_.inverse()

        if self.dependent:
            self.Sigma += ch.bmm(X.view(X.size(0), X.size(1), 1),  
                                X.view(X.size(0), 1, X.size(1))).mean(0)
        return X@self.weight

    def pre_step_hook(self, 
                        inp: ch.Tensor) -> None:
        # TODO: find a cleaner way to do this
        if self.noise_var is not None and not self.dependent:
            self.weight.grad += (self.args.l1 * ch.sign(inp)).mean(0)[...,None]
        if self.dependent:
            self.weight.grad = self.Sigma.inverse()@self.weight.grad
                
    def iteration_hook(self, 
                        i: int, 
                        loop_type: str, 
                        loss: ch.Tensor, 
                        batch: ch.Tensor) -> None:
        if self.noise_var is None:
            # project model parameters back to domain 
            var = self._parameters[1]['params'][0].inverse()
            self._parameters[1]['params'][0].data = ch.clamp(var, self.var_bounds.lower, self.var_bounds.upper).inverse()

    def parameters(self) -> List: 
        if self._parameters is None: 
            raise "model parameters are not set"
        elif isinstance(self._parameters, collections.OrderedDict):
            return self._parameters.values()
        return self._parameters

