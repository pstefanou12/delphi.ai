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
            steps: int=1000,
            unknown: bool=True,
            clamp: bool=True,
            n: int=10, 
            val: int=50,
            tol: float=1e-2,
            workers: int=0,
            r: float=2.0,
            num_samples: int=100,
            bs: int=10,
            lr: float=1e-1,
            var_lr: float=1e-1, 
            step_lr: int=100, 
            custom_lr_multiplier: str=None,
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
        self.alpha = Tensor([alpha]) 
        self.unknown = unknown 
        self.model = None
        self.clamp = clamp
        self.iter_hook = None
        self.n = n 
        self.val = val
        self.tol = tol
        self.workers = workers
        self.r = r 
        self.num_samples = num_samples
        self.bs = bs
        self.lr = lr 
        self.ds = None
        self.multi_class = multi_class
        self.fit_intercept = fit_intercept

        config.args = Parameters({ 
            'steps': steps,
            'momentum': 0.0, 
            'weight_decay': 0.0,   
            'num_samples': num_samples,
            'lr': lr,  
            'var_lr': var_lr,
            'eps': eps,
        })

        # set attribute for learning rate scheduler
        if custom_lr_multiplier: 
            config.args.__setattr__('custom_lr_multiplier', custom_lr_multiplier)
        else: 
            config.args.__setattr__('step_lr', step_lr)
            config.args.__setattr__('step_lr_gamma', step_lr_gamma)
        
    def fit(self, X: Tensor, y: Tensor):
        """
        """
        # create dataset and dataloader
        # separate into training and validation set
        rand_indices = ch.randperm(X.size(0))
        train_indices, val_indices = rand_indices[self.val:], rand_indices[:self.val]
        self.X_train, self.y_train = X[train_indices], y[train_indices]
        self.X_val, self.y_val = X[val_indices], y[val_indices]

        self.train_ds = TensorDataset(self.X_train, self.y_train)
        train_loader = DataLoader(self.train_ds, batch_size=self.bs, num_workers=self.workers)

        self.val_ds = TensorDataset(self.X_val, self.y_val) 
        val_loader = DataLoader(self.val_ds, batch_size=len(self.val_ds), num_workers=self.workers)

        self.trunc_log_reg = TruncatedLogisticRegressionModel(config.args, self.X_train, self.y_train, self.X_val, self.y_val, self.phi, self.tol, self.r, self.alpha, self.clamp, self.multi_class, self.fit_intercept, self.n)

        trainer = Trainer(self.trunc_log_reg) 
        
        # run PGD for parameter estimation 
        trainer.train_model((train_loader, None))


class TruncatedLogisticRegressionModel(delphi.delphi):
    '''
    Truncated logistic regression model to pass into trainer framework.  
    '''
    def __init__(self, args,  X_train, y_train, X_val, y_val, phi, tol, r, alpha, clamp, multi_class, fit_intercept,  n=100, store=None, table=None, schema=None): 
        '''
        Args: 
            args (cox.utils.Parameters) : parameter object holding hyperparameters
        '''
        super().__init__(args, store=store, table=table, schema=schema)
        # use OLS as empirical estimate to define projection set
        self.phi = phi
        self.r = r
        self.alpha = alpha
        self.multi_class = multi_class
        self.fit_intercept=fit_intercept 

        # initialize projection set
        self.clamp = clamp
        # empirical estimates for logistic regression
        self.emp_log_reg = LogisticRegression(penalty='none', fit_intercept=self.fit_intercept, multi_class=self.multi_class)
        self.emp_log_reg.fit(X_train, y_train.flatten())
        self.emp_weight = Tensor(self.emp_log_reg.coef_)
        print("emp weight: {}".format(self.emp_weight))
        self.emp_bias = Tensor(self.emp_log_reg.intercept_)

        # projection set radius
        self.radius = self.r * (ch.sqrt(2.0 * ch.log(1.0 / self.alpha)))
        if self.clamp:
            self.weight_bounds = Bounds((self.emp_weight - self.r).flatten(),
                                        (self.emp_weight + self.r).flatten())
            if self.emp_log_reg.intercept_:
                self.bias_bounds = Bounds(float(self.emp_bias - self.radius),
                                          float(self.emp_bias + self.radius))
        else: 
            pass

        # validation set
        # use steps counter to keep track of steps taken
        self.n, self.steps = n, 0
        self.X_val, self.y_val = X_val, y_val
        self.X_train, self.y_train = X_train, y_train
        self.tol = tol
        # track best estimates based off of gradient norm
        self.best_grad_norm = None
        self.best_state_dict = None
        self.best_opt = None
       
        if multi_class == 'multinomial': 
            self.model = Linear(in_features=self.X_train.size(1), out_features=len(y_train.unique()), bias=True)
        else: 
            self.model = Linear(in_features=self.X_train.size(1), out_features=1, bias=True)
        
        """
        SkLearn sets up multinomial classification differenlty. So when doing 
        multinomial classification, we initialize with random estimates.
        """
        # assign empirical estimates
        if self.multi_class == 'ovr':
            self.model.weight.data = self.emp_weight
            self.model.bias.data = self.emp_bias
        update_params = None

    def check_grad(self): 
        """
        Calculates the check_grad of the current regression estimates of the validation set. It 
        then updates the best estimates accordingly based off of the check_grad's norm.
        """
        pred = self.model(self.X_val)
        if self.multi_class == 'multinomial': 
            loss = TruncatedCE.apply(pred, self.y_val, self.phi) 
            grad, = ch.autograd.grad(loss, [pred])
            grad = grad.sum(0)
        else: 
            loss = TruncatedBCE.apply(pred, self.y_val, self.phi)
            grad, = ch.autograd.grad(loss, [pred])
            grad = grad.sum(0)

        grad_norm = grad.norm(dim=-1)
        # check that gradient magnitude is less than tolerance
        if self.steps != 0 and grad_norm < self.tol: 
            print("Final Gradient Estimate: {}".format(grad_norm))
            raise ProcedureComplete()
        
        print("{} Steps | Gradient Estimate: {}".format(self.steps, grad_norm))
        # if smaller gradient norm, update best
        if self.best_grad_norm is None or grad_norm < self.best_grad_norm: 
            self.best_grad_norm = grad_norm
            # keep track of state dict
            self.best_state_dict, self.best_opt = copy.deepcopy(self.model.state_dict()), copy.deepcopy(self.optimizer.state_dict())
        elif 1e-1 <= grad_norm - self.best_grad_norm: 
            # load in the best model state and optimizer dictionaries
            self.model.load_state_dict(self.best_state_dict)
            self.optimizer.load_state_dict(self.best_opt)

    def train_step(self, i, batch):
        '''
        Training step for defined model.
        Args: 
            i (int) : gradient step or epoch number
            batch (Iterable) : iterable of inputs that 
        '''
        self.optimizer.zero_grad()
        inp, targ = batch

        pred = self.model(inp)
        if self.multi_class == 'multinomial': 
            loss = TruncatedCE.apply(pred, targ, self.phi)
        elif self.multi_class == 'ovr': 
            loss = TruncatedBCE.apply(pred, targ, self.phi)
        else: 
            raise Exception('multi class input invalid')

        loss.backward()
        self.optimizer.step()

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
        # increase number of steps taken
        self.steps += 1
        # project model parameters back to domain 
        if self.clamp: 
            if self.multi_class: 
               pass 
            else: 
                # project weight coefficients
                self.model.weight.data = ch.stack([ch.clamp(self.model.weight.data[i], float(self.weight_bounds.lower[i]),
                                                             float(self.weight_bounds.upper[i])) for i in
                                                    range(model.weight.size(0))])
                # project bias coefficient
                if self.model.bias:
                    self.model.bias.data = ch.clamp(self.model.bias, self.bias_bounds.lower, self.bias_bounds.upper).reshape(
                        self.model.bias.size())
        else: 
            pass

        # check for convergence every n steps
        if self.steps % self.n == 0: 
            grad = self.check_grad()

