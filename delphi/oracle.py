import warnings
import torch as ch
from torch import Tensor
from torch.distributions.multivariate_normal import MultivariateNormal, _batch_mahalanobis
from abc import ABC
from decimal import Decimal
from orthnet import Hermite
import math

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


class Interval:
    """
    Interval truncation
    """
    # tensor inputs for vector interval truncation
    def __init__(self, lower, upper):
        self.bounds = Bounds(lower, upper)

    def __call__(self, x):
        # check sample device
        if x.is_cuda:
            return ((self.bounds.lower.cuda() < x).prod(-1) * (x < self.bounds.upper.cuda()).prod(-1))
        return ((self.bounds.lower < x).prod(-1) * (x < self.bounds.upper).prod(-1))


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


class Sphere(oracle):
    """
    Spherical truncation
    """
    def __init__(self, covariance_matrix, centroid, radius):
        self._unbroadcasted_scale_tril = covariance_matrix.cholesky()
        self.centroid = centroid
        self.radius = radius

    def __call__(self, x):
        diff = x - self.centroid
        dist = ch.sqrt(_batch_mahalanobis(self._unbroadcasted_scale_tril, diff))
        return (dist < self.radius).float().flatten()


# LAMBDA FUNCTIONS TRIED
#  2D DIMENSIONAL GAUSSIAN LAMBDA FUNCTIONS
set_two_d = lambda x: (x[1].pow(2) + x[0].pow(2) > .5)
horseshoe = lambda x: x[0] > 0 and x[0] ** 2 + x[1] ** 2 > 1 and x[0] ** 2 + x[1] ** 2 < 2
horseshoe_dot = lambda x: x[0] > 0 and 1 < x.pow(2).sum() < 2 or ((x[0] - .5).pow(2) + x[1].pow(2)) < (1 / 6)
triangle = lambda x: x[1] >= 0 and x[1] <= +x[0] + 1 and x[1] <= 1 - x[0] and not (
            (x[0] / 2) ** 2 + (x[1] - 0.52) ** 2 <= 0.02)
# 3D DIMENSIONAL GAUSSIAN LAMBDA FUNCTIONS
three_d_union_check = lambda x: x[0] > 0 and x[2] > 0 or x.pow(2).sum() < 1.0


class Unknown_Gaussian(oracle):
    """
    Oracle that learns truncation set
    """

    def __init__(self, emp_loc, emp_covariance_matrix, S, d):
        # empirical estimates used for membership oracle
        self._emp_dist = MultivariateNormal(emp_loc, emp_covariance_matrix)

        self._d = d
        # calculate the normalizing hermite polynomial constant for degree d
        self._norm_const = Tensor([])
        for i in range(self._d + 1):
            try:
                self._norm_const = ch.cat([self._norm_const, ch.sqrt(Tensor([math.factorial(i)]))])
            except OverflowError:
                self._norm_const = ch.cat([self._norm_const, Tensor([Decimal(math.factorial(i)) ** Decimal(.5)])])
                warnings.warn("Overflow error: converting floats to Decimal")
        self._norm_const = self._norm_const.unsqueeze(1)

        # truncation coefficient
        self._C_v = self.H_v(S).mean(0)
        # must learn distribution for membership oracle
        self._dist = None

    def __call__(self, x):
        if self.dist is None:
            raise Exception("must learn underlying distribution for membership oracle")
        return x[((ch.exp(self.emp_dist.log_prob(x)) / ch.exp(self.dist.log_prob(x))) * self.psi_k(
            x) > .5).flatten().nonzero(as_tuple=False).flatten()]

    # x - (n, d) matrix
    def H_v(self, x):
        return ch.div(Hermite(x.unsqueeze(1).double(), self._d).tensor, self._norm_const).prod(2)

    def psi_k(self, x):
        """
        Characteristic function, determines whether a sample falls within truncation set or not.
        """
        return ch.clamp((self._C_v * self.H_v(x)).sum(1), 0.0)

    @property
    def emp_dist(self):
        return self._emp_dist

    @property
    def dist(self):
        return self._dist

    @dist.setter
    def dist(self, params):
        self._dist = MultivariateNormal(params[0], params[1])

    @property
    def C_v(self):
        return self._C_v

    @property
    def norm_const(self):
        return self._norm_const

    @property
    def d(self):
        return self._d

class DNN_Lower(oracle): 
    """
    Lower bound truncation on the DNN logits.
    """
    def __init__(self, lower): 
        self.lower = lower
        
    def __call__(self, x): 
        return (x > self.lower).float()
    
class DNN_Logit_Ball(oracle): 
    """
    Truncation ball placed on DNN logits.
    INTUITION: logits that are neither very large nor very small insinuate
    that the classification is not 
    """
    def __init__(self, lower, upper): 
        self.lower = lower 
        self.upper = upper
        
    def __call__(self, x): 
        return ((x < self.lower) | (x > self.upper)).float()
        

class Identity(oracle): 
    """
    Identity membership oracle for DNNs. All logits are accepted within the truncation set.
    """
    def __call__(self, x): 
        return ch.ones(x.size())
