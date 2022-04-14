'''
Multinomial logistic rergression that uses gumbel max loss function.
'''

from tkinter import W
import torch as ch
from torch.distributions import Gumbel
from torch.nn import MSELoss

from .linear_model import LinearModel
from delphi.utils.helpers import accuracy
from delphi.grad import GumbelCE

# CONSTANT
mse_loss =  MSELoss()
G = Gumbel(0, 1)


class GumbelCEModel(LinearModel):
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
        stacked = (x@self.model).repeat(self.args.num_samples, 1, 1)
        noised = stacked + G.sample(stacked.size())
        return noised.mean(0).argmax(-1)

    def __call__(self, batch):
        '''
        Training step for defined model.
        Args:
            batch (Iterable) : iterable of inputs that
        '''
        inp, targ = batch
        z = inp@self.model
        loss = GumbelCE.apply(z, targ)
        
        # calculate precision accuracies
        prec1, prec5 = None, None
        if z.size(1) >= 5:
            prec1, prec5 = accuracy(z, targ, topk=(1, 5))
        else:
            prec1, = accuracy(z, targ, topk=(1,))
        return loss, prec1, prec5
    
    def calc_logits(self, inp): 
        return inp@self.model