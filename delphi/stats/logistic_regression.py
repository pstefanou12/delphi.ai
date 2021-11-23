"""
Truncated Logistic Regression.
"""


import torch as ch
from torch import Tensor
from torch.nn import Linear
from torch.utils.data import TensorDataset, DataLoader
from torch import sigmoid as sig
from cox.utils import Parameters
from cox.store import Store
import config
from sklearn.linear_model import LogisticRegression
import copy

from .. import delphi
from .stats import stats
from ..oracle import oracle
from ..grad import TruncatedBCE, TruncatedCE
from ..trainer import Trainer
from ..utils.helpers import Bounds, ProcedureComplete


class TruncatedLogisticRegression(stats):
    """
    """
    def __init__(
            self,
            phi: oracle,
            alpha: float,
            epochs: int=10,
            clamp: bool=True,
            n: int=10, 
            val: int=50,
            tol: float=1e-2,
            r: float=2.0,
            num_samples: int=100,
            bs: int=10,
            lr: float=1e-1,
            var_lr: float=1e-1, 
            step_lr: int=100, 
            custom_lr_multiplier: str=None,
            lr_interpolation: str=None,
            step_lr_gamma: float=.9,
            eps: float=1e-5, 
            multi_class='ovr',
            fit_intercept=True,
            **kwargs):
        """
        """
        super(TruncatedLogisticRegression).__init__()
        # instance variables
        self.phi = phi 
        self.trunc_log_reg = None

        self.custom_lr_multiplier = custom_lr_multiplier
        self.step_lr = step_lr 
        self.step_lr_gamma = step_lr_gamma
        self.lr_interpolation = lr_interpolation


        config.args = Parameters({ 
            'alpha': Tensor([alpha]),
            'bs': bs, 
            'workers': 1,
            'epochs':epochs,
            'momentum': 0.0, 
            'weight_decay': 0.0,   
            'num_samples': num_samples,
            'lr': lr,  
            'eps': eps,
            'tol': tol,
            'val': val,
            'clamp': clamp,
            'fit_intercept': fit_intercept,
            'multi_class': multi_class,
            'r': r,
            'verbose': False,
        })
                
    def fit(self, X: Tensor, y: Tensor):
        """
        """
        self.trunc_log_reg = TruncatedLogisticRegressionModel(config.args, X, y, self.phi, self.custom_lr_multiplier, self.lr_interpolation, self.step_lr, self.step_lr_gamma)

        trainer = Trainer(self.trunc_log_reg) 
        
        # run PGD for parameter estimation 
        trainer.train_model()

        with ch.no_grad():
            self.coef = self.trunc_log_reg.model.weight.clone()
            self.intercept = self.trunc_log_reg.model.bias.clone()

    def __call__(self, x: Tensor): 
        """
        Make predictions with regression estimates.
        """
        with ch.no_grad():
            return self.trunc_log_reg.model(x)

    @property
    def coef_(self): 
        """
        Regression weight.
        """
        return self.coef

    @property
    def intercept_(self): 
        """
        Regression intercept.
        """
        if config.args.fit_intercept:
            return self.intercept


class TruncatedLogisticRegressionModel(delphi.delphi):
    '''
    Truncated logistic regression model to pass into trainer framework.  
    '''
    def __init__(self, args,  X, y, phi, custom_lr_multiplier, lr_interpolation, step_lr, step_lr_gamma): 
        '''
        Args: 
            args (cox.utils.Parameters) : parameter object holding hyperparameters
        '''
        super().__init__(args, custom_lr_multiplier, lr_interpolation, step_lr, step_lr_gamma)
        # separate into training and validation set
        rand_indices = ch.randperm(X.size(0))
        train_indices, val_indices = rand_indices[self.args.val:], rand_indices[:self.args.val]
        self.X_train, self.y_train = X[train_indices], y[train_indices]
        self.X_val, self.y_val = X[val_indices], y[val_indices]

        self.train_ds = TensorDataset(self.X_train, self.y_train)
        self._train_loader = DataLoader(self.train_ds, batch_size=self.args.bs, num_workers=1)

        # use OLS as empirical estimate to define projection set
        self.phi = phi

        # empirical estimates for logistic regression
        self.emp_log_reg = LogisticRegression(penalty='none', fit_intercept=self.args.fit_intercept, multi_class=self.args.multi_class)
        self.emp_log_reg.fit(self.X_train, self.y_train.flatten())
        self.emp_weight = Tensor(self.emp_log_reg.coef_)
        self.emp_bias = Tensor(self.emp_log_reg.intercept_)

        # projection set radius
        self.radius = self.args.r * (ch.sqrt(2.0 * ch.log(1.0 / self.args.alpha)))
        if self.args.clamp:
            self.weight_bounds = Bounds((self.emp_weight - self.radius).flatten(),
                                        (self.emp_weight + self.radius).flatten())
            if self.emp_log_reg.intercept_:
                self.bias_bounds = Bounds(float(self.emp_bias - self.radius),
                                          float(self.emp_bias + self.radius))
        else: 
            pass

        if self.args.multi_class == 'multinomial': 
            self.model = Linear(in_features=self.X_train.size(1), out_features=len(y_train.unique()), bias=True)
        else: 
            self.model = Linear(in_features=self.X_train.size(1), out_features=1, bias=True)
        
        """
        SkLearn sets up multinomial classification differenlty. So when doing 
        multinomial classification, we initialize with random estimates.
        """
        # assign empirical estimates
        if self.args.multi_class == 'ovr':
            self.model.weight.data = self.emp_weight
            self.model.bias.data = self.emp_bias
        update_params = None

    def check_nll(self): 
        """
        Calculates the check_grad of the current regression estimates of the validation set. It 
        then updates the best estimates accordingly based off of the check_grad's norm.
        """
        pred = self.model(self.X_val)
        if self.args.multi_class == 'multinomial': 
            loss = TruncatedCE.apply(pred, self.y_val, self.phi) 
        else: 
            loss = TruncatedBCE.apply(pred, self.y_val, self.phi)
        return loss

    def train_step(self, i, batch):
        '''
        Training step for defined model.
        Args: 
            i (int) : gradient step or epoch number
            batch (Iterable) : iterable of inputs that 
        '''
        inp, targ = batch

        pred = self.model(inp)
        if self.args.multi_class == 'multinomial': 
            loss = TruncatedCE.apply(pred, targ, self.phi)
        elif self.args.multi_class == 'ovr': 
            loss = TruncatedBCE.apply(pred, targ, self.phi)
        else: 
            raise Exception('multi class input invalid')

        return loss, None, None

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
        # project model parameters back to domain 
        if self.args.clamp: 
            if self.args.multi_class: 
               pass 
            else: 
                # project weight coefficients
                self.model.weight.data = ch.stack([ch.clamp(self.model.weight.data[i], float(self.weight_bounds.lower[i]),
                                                             float(self.weight_bounds.upper[i])) for i in
                                                    range(model.weight.size(0))])
                # project bias coefficient
                if self.args.fit_intercept:
                    self.model.bias.data = ch.clamp(self.model.bias, self.bias_bounds.lower, self.bias_bounds.upper).reshape(
                        self.model.bias.size())
        else: 
            pass

    def epoch_hook(self, i, loop_type, loss, prec1, prec5, batch):
        # check for convergence every at each epoch
        loss = self.check_nll()
        print("Iteration {} | Log Likelihood: {}".format(i, round(float(abs(loss)), 3)))

    @property 
    def train_loader(self): 
        return self._train_loader

