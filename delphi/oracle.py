import warnings
import torch as ch
from torch import Tensor
from torch.distributions.multivariate_normal import MultivariateNormal
from abc import ABC

from .utils.helpers import Bounds


class oracle(ABC):
    """
    Oracle for data sets.
    """
    def __call__(self, x):
        """
        Membership oracle.
        Args: 
            x: samples to check membership
        """
        pass


class Interval(oracle):
    """
    Interval truncation
    """
    # tensor inputs for vector interval truncation
    def __init__(self, lower, upper):
        self.bounds = Bounds(lower, upper)

    def __call__(self, x):
        # check sample device
        if x.is_cuda:
            return ((self.bounds.lower.cuda() < x).prod(dim=1) * (x < self.bounds.upper.cuda()).prod(dim=1))
        return ((self.bounds.lower < x).prod(dim=1) * (x < self.bounds.upper).prod(dim=1))


class IntervalUnion(oracle):
    """
    Receives an iterable of tuples that contain a lower, upper bound tensor.
    """

    def __init__(self, intervals):
        self.oracles = [Interval(int_[0], int_[1]) for int_ in intervals]

    def __call__(self, x):
        result = Tensor([])
        for oracle_ in self.oracles:
            result = ch.logical_or(result, oracle_(x)) if result.nelement() > 0 else oracle_(x)
        return result


class Left(Interval):
    """
    Left truncation
    """
    def __init__(self, left):
        """
        Args: 
            left: left bound - size (d,)
        """
        super().__init__(left, ch.full(left.size(), float('inf')))


class Right(Interval):
    """
    Right truncation
    """

    def __init__(self, right):
        """
        Args: 
            right: right bound - size (d,)
        """
        super().__init__(-ch.full(right.size(), float('inf')), right)

class Lambda(oracle):   
    """
    Lambda function oracle. Takes in a lambda function/callable that can be applied to one [n,] sized sample as pytorch tensor
    """
    def __init__(self, lambda_): 
        self.lambda_ = lambda_
    
    def __call__(self, x):
        return ch.cat([self.lambda_(x[i]).unsqueeze(0) for i in range(x.size(0))])
