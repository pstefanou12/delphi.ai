'''
Multinomial logistic regression that uses softmax loss function.
'''

# distribution tests 

from re import A, I
from tkinter import W
import unittest
import numpy as np
import torch as ch
from torch import Tensor
from torch.distributions import MultivariateNormal, Uniform, Gumbel
from torch.distributions.kl import kl_divergence
from torch.distributions.multivariate_normal import _batch_mahalanobis
from torch.distributions.transforms import SigmoidTransform
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.nn import CosineSimilarity, Softmax, CrossEntropyLoss
from torch.nn import MSELoss
import torch.linalg as LA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import confusion_matrix
import random
from cox.store import Store

from delphi import delphi
from delphi.trainer import Trainer
from delphi import oracle
from delphi.utils.helpers import Parameters, cov, accuracy
from delphi.utils.datasets import make_train_and_val
from .linear_model import LinearModel

# CONSTANT
mse_loss =  MSELoss()
base_distribution = Uniform(0, 1)
transforms_ = [SigmoidTransform().inv]
logistic = TransformedDistribution(base_distribution, transforms_)
cos_sim = CosineSimilarity()
softmax = Softmax(dim=1)
ce = CrossEntropyLoss()
G = Gumbel(0, 1)



class SoftmaxModel(LinearModel):
    '''
    Truncated logistic regression model to pass into trainer framework.
    '''
    def __init__(self, args, d, k):
        '''
        Args:
            args (cox.utils.Parameters) : parameter object holding hyperparameters
        '''
        super().__init__(args, d=d, k=k)

    def calc_emp_model(self): 
        pass    

    def pretrain_hook(self):
        self.model.data = self.weight
        self.model.requires_grad = True
        self.params = [self.model]
        
    def predict(self, x): 
        with ch.no_grad():
            return softmax(x@self.model).argmax(dim=-1)

    def __call__(self, batch):
        '''
        Training step for defined model.
        Args:
            batch (Iterable) : iterable of inputs that
        '''
        inp, targ = batch
        z = inp@self.model
        loss = ce(z, targ)
        
        # calculate precision accuracies
        prec1, prec5 = None, None
        if z.size(1) >= 5:
            prec1, prec5 = accuracy(z, targ, topk=(1, 5))
        else:
            prec1, = accuracy(z, targ, topk=(1,))
        return loss, prec1, prec5
    
    def calc_logits(self, inp): 
        return inp@self.model