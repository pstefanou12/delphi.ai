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
        if self.dependent: 
            self.s = self.args.c_s * (ch.sqrt(ch.log(Tensor([1/self.args.alpha]))) + 1)

    def calc_emp_model(self, 
                        train_loader: ch.Tensor): 
        '''
        Calculates empirical estimates for a truncated linear model. Assigns 
        estimates to a Linear layer. By default calculates OLS for truncated liner regression.
        '''
        pass
        
    def iteration_hook(self, i, is_train, loss, batch):
        if not self.args.constant: self.schedule.step()