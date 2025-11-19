import torch as ch
import torch.linalg as LA
from torch import Tensor
from torch.distributions.multivariate_normal import MultivariateNormal, _batch_mahalanobis
from abc import ABC
from decimal import Decimal
import math
from scipy.linalg import sqrtm

from .utils.helpers import Bounds, cov


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
        return ((self.bounds.lower < x).prod(-1) * (x < self.bounds.upper).prod(-1))[...,None]


class KIntervalUnion(oracle):
    """
    Receives an iterable of tuples that contain a lower, upper bound tensor.
    """

    def __init__(self, intervals):
        self.oracles = [Interval(int_[0], int_[1]) for int_ in intervals]

    def __call__(self, x):
        result = Tensor([])
        for oracle_ in self.oracles:
            result = ch.logical_or(result, oracle_(x)) if result.nelement() > 0 else oracle_(x)
        return result[..., None]

    def __str__(self): 
        return 'k-interval union'


class DiffLogitOracle(oracle): 

    def __init__(self, left): 
        self.left = left

    def __call__(self, x): 
        topk = ch.topk(x, 2, dim=-1)[0]
        return topk.diff() > self.left

    def __str__(self): 
        return 'diff logit oracle'


class Left_Regression(oracle):
    """
    Left Regression Truncation.
    """
    def __init__(self, left):
        """
        Args: 
            left: left truncation
        """
        super(Left_Regression, self).__init__()
        self.left = left

    def __call__(self, x): 
        return x > self.left

    def __str__(self): 
        return 'left regression'


class Left_K_Logit(oracle):
    """
    Truncated the kth logit by, only accepting inputs the fall with z[k] > left.
    """
    def __init__(self, left, k):
        """
        Args: 
            left: left truncation
            k: logit to truncate
        """
        super(Left_K_Logit, self).__init__()
        self.left = left
        self.k = k

    def __call__(self, x): 
        return (x[...,self.k] > self.left)[...,None]

    def __str__(self): 
        return 'left truncate kth logit'


class Right_K_Logit(oracle):
    """
    Truncated the kth logit by, only accepting inputs the fall with z[k] < right.
    """
    def __init__(self, right, k):
        """
        Args: 
            right: right truncation
            k: logit to truncate
        """
        super(Right_K_Logit, self).__init__()
        self.right = right
        self.k = k

    def __call__(self, x): 
        return (x[...,self.k] > self.right)[...,None]

    def __str__(self): 
        return 'right truncate kth logit'


class Left_Distribution(oracle):
    """
    Left Distribution Truncation.
    """
    def __init__(self, left):
        """
        Args: 
            left: left truncation for distributions - multiplies whether each dimension of sample is within bounds
        """
        super(Left_Distribution, self).__init__()
        self.left = left

    def __call__(self, x): 
        return (x > self.left).prod(dim=-1, keepdim=True)

    def __str__(self): 
        return 'left'


class Right_Regression(oracle):
    """
    Right Regression Truncation.
    """
    def __init__(self, right):
        """
        Args: 
            right: right truncation
        """
        super(Right_Regression, self).__init__()
        self.right = right

    def __call__(self, x): 
        return x < self.right

    def __str__(self): 
        return 'right'


class Right_Distribution(oracle):
    """
    Right Distribution Truncation.
    """
    def __init__(self, right):
        """
        Args: 
            right (torch.Tensor): right truncation
        """
        super(Right_Distribution, self).__init__()
        assert isinstance(right, Tensor), "right is type: {}. expecting right to be type torch.Tensor.".format(type(right)) 
        assert right.dim() == 1, "right size: {}. expecting right size (d,).".format(right.size())
        self.right = right

    def __call__(self, x): 
        return (x < self.right).prod(dim=-1)

    def __str__(self): 
        return 'right'


class Lambda(oracle):   
    """
    Lambda function oracle. Takes in a lambda function/callable that can be applied to one [n,] sized sample as pytorch tensor
    """
    def __init__(self, lambda_): 
        self.lambda_ = lambda_
    
    def __call__(self, x):
        return ch.cat([self.lambda_(x[i]).unsqueeze(0) for i in range(x.size(0))])

    def __str__(self): 
        return 'lambda'


class Sphere(oracle):
    """
    Spherical truncation
    """
    def __init__(self, covariance_matrix, centroid, radius):
        self._unbroadcasted_scale_tril = LA.cholesky(covariance_matrix)
        self.centroid = centroid
        self.radius = radius

    def __call__(self, x):
        diff = x - self.centroid
        dist = ch.sqrt(_batch_mahalanobis(self._unbroadcasted_scale_tril, diff))
        return (dist < self.radius).float().flatten()

    def __str__(self): 
        return 'sphere'


# LAMBDA FUNCTIONS TRIED
#  2D DIMENSIONAL GAUSSIAN LAMBDA FUNCTIONS
set_two_d = lambda x: (x[1].pow(2) + x[0].pow(2) > .5)
horseshoe = lambda x: x[0] > 0 and x[0] ** 2 + x[1] ** 2 > 1 and x[0] ** 2 + x[1] ** 2 < 2
horseshoe_dot = lambda x: x[0] > 0 and 1 < x.pow(2).sum() < 2 or ((x[0] - .5).pow(2) + x[1].pow(2)) < (1 / 6)
triangle = lambda x: x[1] >= 0 and x[1] <= +x[0] + 1 and x[1] <= 1 - x[0] and not (
            (x[0] / 2) ** 2 + (x[1] - 0.52) ** 2 <= 0.02)
# 3D DIMENSIONAL GAUSSIAN LAMBDA FUNCTIONS
three_d_union_check = lambda x: x[0] > 0 and x[2] > 0 or x.pow(2).sum() < 1.0


class UnknownGaussian(oracle):
    def __init__(self, emp_loc, emp_covariance_matrix, S, k):
        self.emp_loc = emp_loc  # shape (d,) or (,) for univariate
        self.emp_covariance_matrix = emp_covariance_matrix
        self._emp_dist = MultivariateNormal(emp_loc, emp_covariance_matrix)
        self._d = int(emp_loc.shape[0])
        self._k = k
        self._factorials = ch.tensor([math.factorial(i) for i in range(k + 1)], dtype=ch.float32)

        self.multi_indices = self._generate_multi_indices(self._k)

        self._norm_const = self._compute_norm_constants()

        self._C_v = self._compute_hermite_coefficients(S)

        self._dist = None

    def _generate_multi_indices(self, k):
        # For d=1 this returns [[0],[1],...,[k]]
        from itertools import product
        indices = []
        for degree in range(k + 1):
            for combo in product(range(degree + 1), repeat=self._d):
                if sum(combo) == degree:
                    indices.append(list(combo))
        return ch.tensor(indices, dtype=ch.long)

    def _hermite_polynomial_1d(self, z, degree):
        """Probabilist's Hermite He_n(z) computed via stable recurrence."""
        if degree == 0:
            return ch.ones_like(z)
        elif degree == 1:
            return z
        else:
            He_prev_prev = ch.ones_like(z)
            He_prev = z
            for n in range(2, degree + 1):
                He_curr = z * He_prev - (n - 1) * He_prev_prev
                He_prev_prev, He_prev = He_prev, He_curr
            return He_prev

    def H_v(self, x):
        """
        Evaluate multivariate Hermite polynomials *at standardized inputs*.
        For univariate: z = (x - emp_loc) / sqrt(emp_var).
        Returns tensor shape (n, num_indices).
        """
        n = x.shape[0]
        num_indices = len(self.multi_indices)
        result = ch.ones(n, num_indices, dtype=x.dtype, device=x.device)

        # Standardize: for each dim use emp_loc and emp_covariance_matrix diag
        # For univariate:
        if self._d == 1:
            var = self.emp_covariance_matrix.reshape(-1)[0] if self.emp_covariance_matrix.numel() == 1 else self.emp_covariance_matrix[0,0]
            std = ch.sqrt(var)
            z = (x[:, 0] - self.emp_loc[0]) / (std + 1e-12)   # (n,)
            for idx, V in enumerate(self.multi_indices):
                deg = int(V[0].item())
                He = self._hermite_polynomial_1d(z, deg)      # (n,)
                # Normalize by sqrt(deg!) so basis is orthonormal under N(0,1)
                if deg > 0:
                    He = He / ch.sqrt(self._factorials[deg])
                result[:, idx] = He
            return result

        # fallback for multivariate (not used here)
        for idx, V in enumerate(self.multi_indices):
            prod = ch.ones(n, dtype=x.dtype, device=x.device)
            for dim in range(self._d):
                degree = int(V[dim].item())
                if degree > 0:
                    # standardize dimension
                    var = self.emp_covariance_matrix[dim, dim]
                    std = ch.sqrt(var)
                    z = (x[:, dim] - self.emp_loc[dim]) / (std + 1e-12)
                    He = self._hermite_polynomial_1d(z, degree) / ch.sqrt(self._factorials[degree])
                    prod = prod * He
            result[:, idx] = prod
        return result

    def _compute_hermite_coefficients(self, S):
        """
        Compute c_v = E_D[H_V(z)] where H_V evaluated at standardized samples from S.
        S should be samples *from the truncated set* (i.e., only samples x in S).
        """
        H_vals = self.H_v(S)   # shape (n, num_indices)
        return H_vals.mean(dim=0)
    
    def fit_ridge_coefficients(self, H_all, y, lambda_reg=1e-3, dtype=ch.float32, device='cpu'):
        # Solve (H^T H + lambda I) c = H^T y
        HtH = H_all.t() @ H_all   # (m, m)
        m = HtH.size(0)
        A = HtH + lambda_reg * ch.eye(m, dtype=dtype, device=device)
        rhs = H_all.t() @ y
        c = ch.linalg.solve(A, rhs)   # (m,)
        return c

    def psi_k(self, x, clamp_range=(0,1), use_sigmoid=True, alpha=10.0):
        """
        Evaluate psi approximation on x (tensor shape (n,d)).
        Returns (n, 1) tensor with values ~[0,1].
        """
        H_vals = self.H_v(x)                      # (n, num_indices)
        expansion = (self._C_v * H_vals).sum(dim=1)   # (n,)
        if use_sigmoid:
            psi_vals = ch.sigmoid(alpha * expansion)
        else:
            # ReLU then clamp to [0,1]
            psi_vals = ch.clamp(expansion, min=clamp_range[0], max=clamp_range[1])
    
        return psi_vals.unsqueeze(-1)   # (n,1)

    def _compute_norm_constants(self):
        """Compute sqrt(V!) for each multi-index V."""
        norms = []
        for V in self.multi_indices:
            # V! = product of v_i! for each component
            V_factorial = ch.prod(self._factorials[V.long()])
            norms.append(ch.sqrt(V_factorial.float()))
        return ch.tensor(norms)

    def __call__(self, x):
        if self._dist is None:
            raise Exception("must learn underlying distribution for membership oracle")
        
        # Compute the rescaled characteristic function
        ratio = ch.exp(self._emp_dist.log_prob(x) - self._dist.log_prob(x))
        return (ratio.unsqueeze(-1) * self.psi_k(x) > 0.5).float()

    @property
    def emp_dist(self):
        return self._emp_dist

    @property
    def dist(self):
        return self._dist

    @dist.setter
    def dist(self, dist_):
        self._dist = dist_

    @property
    def C_v(self):
        return self._C_v

    @property
    def norm_const(self):
        return self._norm_const

    @property
    def d(self):
        return self._d

    def __str__(self): 
        return 'unknown gaussian'
    

class Identity(oracle): 
    """
    Identity membership oracle for DNNs. All logits are accepted within the truncation set.
    """
    def __call__(self, x): 
        return ch.ones(x.size()).prod(-1, keepdim=True)

    def __str__(self): 
        return 'identity'


class LogitBall(oracle): 
    """
    Truncation based off of norm of logits. Logt norm needs to be smaller than input bound.
    In other words, retain the input that the classifier is less certain on. Smaller 
    unnormalized log probabilities implies uncertainty in classification.
    """
    def __init__(self, bound): 
        self.bound = bound
        
    def __call__(self, x): 
        return (x.norm(dim=-1, keepdim=True) <= self.bound)

    def __str__(self): 
        return 'logit ball'


class LogitSum(oracle): 

    def __init__(self, ceiling): 
        self.ceiling = ceiling

    def __call__(self, x): 
        return x.sum(1, keepdim=True) < self.ceiling


class TruncateLogit(oracle): 
    def __init__(self, logit): 
        self.logit = logit

    def __call__(self, x):
        return x.argmax(-1, keepdim=True) != self.logit

   
class RandomTruncation(oracle): 
    def __init__(self, threshold): 
        self.threshold = threshold 

    def __call__(self, x): 
        if x.dim() == 3: 
            return ch.rand(x.size(0), x.size(1), 1) > self.threshold
        return ch.rand(x.size(0)) > self.threshold


class LogitBallComplement(oracle): 
    
    """
    Truncation based off of complement norm of logits. Logit norm needs to be greater than input bound.
    In other words, retain the inputs that the classifier is more certain on. Larger 
    unnormalized log probabilities implies more certraining in classification.
    """
    def __init__(self, bound, temperature=ch.ones(1)): 
        self.bound = bound
        self.temperature = temperature
        
    def __call__(self, x): 
        x_ = x
        return (x_.norm(dim=-1, keepdim=True) >= self.bound)

    def __str__(self): 
        return 'logit ball complement'


class Sum_Ceiling(oracle):
    """
    Sums sample from a truncated boolean product distribution and returns True if x \in \S, 
    if sum <= ceiling, else False.
    """
    # tensor inputs for vector interval truncation
    def __init__(self, ceil):
        self.ceil = ceil

    def __call__(self, x):
        return (x.sum(1) <= self.ceil)


class Sum_Floor(oracle):
    """
    Sums sample from a truncated boolean product distribution and returns True if x \in \S, 
    if sum >= ceiling, else False.
    """
    # tensor inputs for vector interval truncation
    def __init__(self, floor):
        self.floor = floor

    def __call__(self, x):
        return (x.sum(1) >= self.floor)


class GumbelLogisticLeftTruncation(oracle):
    """
    The difference beween two samples from a Gumbel distribution is equivalent to 
    a sample from a logistic distribution. Thus, this oracle takes noised logits 
    from a multinomial logistic regression, calculates the difference and uses the
    left truncation mechanism for truncated logistic regression.
    """
    def __init__(self, left): 
        """
        Args: 
           left (float): left truncation threshold
        """
        self.left = left 
        
    def __call__(self, x): 
        """
        Args: 
            x (torch.Tensor): n by 2 array of gumbel noised logits
        """
        return ((x[:,:,1] - x[:,:,0]) > self.left)[...,None]

    def __str__(self): 
      return 'Gumbel Logistic Left Truncation Set'