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
from ..delphi_logger import delphiLogger
from ..utils.helpers import Bounds


class LinearModel(delphi):
    '''
    Truncated linear model parent class.
    '''
    def __init__(self, 
                args: Parameters,
                dependent: bool,
                logger: delphiLogger,
                emp_weight=None): 
        '''
        Args: 
            args (cox.utils.Parameters) : parameter object holding hyperparameters
            k (int): number of output logits
        '''
        super().__init__(args, logger)
        self.emp_weight = emp_weight
        
        self.d, self.k = None, None
        self.base_radius = 2.0
        self.dependent = dependent
        if self.dependent: 
            self.s = self.args.c_s * (ch.sqrt(ch.log(Tensor([1/self.args.alpha]))) + 1)