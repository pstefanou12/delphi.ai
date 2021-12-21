"""
Censored multivariate normal distribution with oracle access (ie. known truncation set).
"""

import torch as ch
from torch import Tensor
from torch.distributions import Bernoulli
from torch.utils.data import TensorDataset
import cox
import math

from .. import delphi
from .distributions import distributions
from ..utils.datasets import make_train_and_val_distr
from ..grad import TruncatedBooleanProductNLL
from ..trainer import Trainer
from ..utils.helpers import Parameters
from ..utils.defaults import check_and_fill_args, TRAINER_DEFAULTS, DELPHI_DEFAULTS, TRUNC_BOOL_PROD_DEFAULTS


class TruncatedBernoulli(distributions):
    """
    Truncated boolean product distribution class.
    """
    def __init__(self,
            args: Parameters,
            store: cox.store.Store=None):
        """
        """
        super(TruncatedBernoulli).__init__()
        # instance variables
        assert isinstance(args, Parameters), "args is type: {}. expecting args to be type delphi.utils.helpers.Parameters"
        assert store is None or isinstance(store, cox.store.Store), "store is type: {}. expecting cox.store.Store.".format(type(store))
        self.store = store 
        self.trunc_bool_prod = None
        # algorithm hyperparameters
        TRUNC_BOOL_PROD_DEFAULTS.update(TRAINER_DEFAULTS)
        TRUNC_BOOL_PROD_DEFAULTS.update(DELPHI_DEFAULTS)
        self.args = check_and_fill_args(args, TRUNC_BOOL_PROD_DEFAULTS)

    def fit(self, S: Tensor):
        """
        """
        assert isinstance(S, Tensor), "S is type: {}. expected type torch.Tensor.".format(type(S))
        assert S.size(0) > S.size(1), "input expected to be shape num samples by dimenions, current input is size {}.".format(S.size()) 
        
        self.train_loader_, self.val_loader_ = make_train_and_val_distr(self.args, S, TensorDataset)
        self.trunc_bool = TruncatedBooleanProductDistribution(self.args, self.train_loader_.dataset)
        # run PGD to predict actual estimates
        trainer = Trainer(self.trunc_bool, self.args, store=self.store)
        trainer.train_model((self.train_loader_, self.val_loader_))
        return self
    
    @property 
    def probs_(self): 
        """
        Returns the probability vector for the d dimensional Bernoulli distribution. 
        """
        return self.trunc_bool.model.probs.clone()
    
    @property
    def logits_(self): 
        """
        Returns the logits vector for the d dimensional Bernoulli distribution.
        """
        return self.trunc_bool.model.logits.clone()


class TruncatedBooleanProductDistribution(delphi.delphi):
    """
    Model for truncated boolean product distributions to be passed into trainer.
    """
    def __init__(self, args, train_ds): 
        """
        Args: 
            args (cox.utils.Parameters) : parameter object holding hyperparameters
        """
        super().__init__(args)
        self.train_ds = train_ds
        self.model = None
        self.emp_p, self.emp_z = None, None
        # initialize empirical estimates
        self.calc_emp_model()

    def pretrain_hook(self):
        self.radius = self.args.r * math.log((1 / self.args.alpha) ** .5)
        # initialize empirical model 
        self.model = Bernoulli(logits=self.emp_z)
        self.model.logits.requires_grad = True
        self.params = [self.model.logits]
    
    def calc_emp_model(self): 
        # percentage of points in S that have label 1
        self.emp_p = self.train_ds.tensors[0].mean(0)
        self.emp_z = ch.log(self.emp_p / (1 - self.emp_p))

    def __call__(self, batch):
        """
        Training step for defined model.
        Args: 
            i (int) : gradient step or epoch number
            batch (Iterable) : iterable of inputs that 
        """
        loss = TruncatedBooleanProductNLL.apply(self.model.logits, *batch, self.args.phi, self.args.num_samples)
        return loss, None, None

    def iteration_hook(self, i, loop_type, loss, prec1, prec5, batch):
        """
        Iteration hook for defined model. Method is called after each 
        training update.
        Args:
            loop_type (str) : "train" or "val"; indicating type of loop
            loss (ch.Tensor) : loss for that iteration
            prec1 (float) : accuracy for top prediction
            prec5 (float) : accuracy for top-5 predictions
        """
        logit_diff = (self.model.logits - self.emp_z)[...,None]
        logit_diff = logit_diff.renorm(p=2, dim=0, maxnorm=self.radius).flatten()
        self.model.logits.data = self.emp_z + logit_diff

    def post_training_hook(self): 
        self.args.r *= self.args.rate
        # remove distribution from the computation graph
        self.model.logits.requires_grad = False
