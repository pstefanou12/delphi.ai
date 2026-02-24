"""Membership oracle definitions for truncated distribution estimation."""

from abc import ABC

import torch as ch
import torch.linalg as LA
from torch import Tensor
from torch.distributions.multivariate_normal import (
    MultivariateNormal,
    _batch_mahalanobis,
)

from delphi.utils.helpers import Bounds, cov
from delphi.polynomials import NormalizedProbabilistHermitePolynomial


class oracle(ABC):  # pylint: disable=invalid-name,too-few-public-methods
    """
    Oracle for data sets.
    """

    def __call__(self, x):
        """
        Membership oracle.
        Args:
            x: samples to check membership
        """


class Interval(oracle):  # pylint: disable=too-few-public-methods
    """
    Interval truncation
    """

    # tensor inputs for vector interval truncation
    def __init__(self, lower, upper):
        """
        Args:
            lower: lower bound tensor
            upper: upper bound tensor
        """
        self.bounds = Bounds(lower, upper)

    def __call__(self, x):
        """Return membership indicator for interval truncation."""
        return ((self.bounds.lower < x).prod(-1) * (x < self.bounds.upper).prod(-1))[
            ..., None
        ]


class KIntervalUnion(oracle):
    """
    Receives an iterable of tuples that contain a lower, upper bound tensor.
    """

    def __init__(self, intervals):
        """
        Args:
            intervals: iterable of (lower, upper) bound tuples
        """
        self.oracles = [Interval(int_[0], int_[1]) for int_ in intervals]

    def __call__(self, x):
        """Return membership indicator for union of interval oracles."""
        result = Tensor([])
        for oracle_ in self.oracles:
            result = (
                ch.logical_or(result, oracle_(x))
                if result.nelement() > 0
                else oracle_(x)
            )
        return result[..., None]

    def __str__(self):
        """Return string representation."""
        return "k-interval union"


class DiffLogitOracle(oracle):  # pylint: disable=too-few-public-methods
    """
    Oracle that thresholds the difference between the top two logits.
    """

    def __init__(self, left):
        """
        Args:
            left: minimum required difference between top two logits
        """
        self.left = left

    def __call__(self, x):
        """Return True if the top-2 logit difference exceeds the left threshold."""
        topk = ch.topk(x, 2, dim=-1)[0]
        return topk.diff() > self.left

    def __str__(self):
        """Return string representation."""
        return "diff logit oracle"


class Left_Regression(oracle):  # pylint: disable=invalid-name
    """
    Left Regression Truncation.
    """

    def __init__(self, left):
        """
        Args:
            left: left truncation
        """
        super().__init__()
        self.left = left

    def __call__(self, x):
        """Return membership indicator for left regression truncation."""
        return x > self.left

    def __str__(self):
        """Return string representation."""
        return "left regression"


class Left_K_Logit(oracle):  # pylint: disable=invalid-name
    """
    Truncated the kth logit by, only accepting inputs the fall with z[k] > left.
    """

    def __init__(self, left, k):
        """
        Args:
            left: left truncation
            k: logit to truncate
        """
        super().__init__()
        self.left = left
        self.k = k

    def __call__(self, x):
        """Return membership indicator for left kth-logit truncation."""
        return (x[..., self.k] > self.left)[..., None]

    def __str__(self):
        """Return string representation."""
        return "left truncate kth logit"


class Right_K_Logit(oracle):  # pylint: disable=invalid-name
    """
    Truncated the kth logit by, only accepting inputs the fall with z[k] < right.
    """

    def __init__(self, right, k):
        """
        Args:
            right: right truncation
            k: logit to truncate
        """
        super().__init__()
        self.right = right
        self.k = k

    def __call__(self, x):
        """Return membership indicator for right kth-logit truncation."""
        return (x[..., self.k] > self.right)[..., None]

    def __str__(self):
        """Return string representation."""
        return "right truncate kth logit"


class Left_Distribution(oracle):  # pylint: disable=invalid-name
    """
    Left Distribution Truncation.
    """

    def __init__(self, left):
        """
        Args:
            left: left truncation for distributions - multiplies whether each dimension
                of sample is within bounds
        """
        super().__init__()
        self.left = left

    def __call__(self, x):
        """Return membership indicator for left distribution truncation."""
        return (x > self.left).prod(dim=-1, keepdim=True)

    def __str__(self):
        """Return string representation."""
        return "left"


class Right_Regression(oracle):  # pylint: disable=invalid-name
    """
    Right Regression Truncation.
    """

    def __init__(self, right):
        """
        Args:
            right: right truncation
        """
        super().__init__()
        self.right = right

    def __call__(self, x):
        """Return membership indicator for right regression truncation."""
        return x < self.right

    def __str__(self):
        """Return string representation."""
        return "right"


class Right_Distribution(oracle):  # pylint: disable=invalid-name
    """
    Right Distribution Truncation.
    """

    def __init__(self, right):
        """
        Args:
            right (torch.Tensor): right truncation
        """
        super().__init__()
        assert isinstance(right, Tensor), (
            f"right is type: {type(right)}. expecting right to be type torch.Tensor."
        )
        assert right.dim() == 1, (
            f"right size: {right.size()}. expecting right size (d,)."
        )
        self.right = right

    def __call__(self, x):
        """Return membership indicator for right distribution truncation."""
        return (x < self.right).prod(dim=-1)

    def __str__(self):
        """Return string representation."""
        return "right"


class Lambda(oracle):
    """
    Lambda function oracle. Takes in a lambda function/callable that can be applied
    to one [n,] sized sample as pytorch tensor
    """

    def __init__(self, lambda_):
        """
        Args:
            lambda_: callable applied to individual samples
        """
        self.lambda_ = lambda_

    def __call__(self, x):
        """Return membership indicator by applying the stored callable to each sample."""
        return ch.cat([self.lambda_(x[i]).unsqueeze(0) for i in range(x.size(0))])

    def __str__(self):
        """Return string representation."""
        return "lambda"


class Sphere(oracle):
    """
    Spherical truncation
    """

    def __init__(self, covariance_matrix, centroid, radius):
        """
        Args:
            covariance_matrix: covariance matrix defining the Mahalanobis metric
            centroid: center of the sphere
            radius: radius threshold
        """
        self._unbroadcasted_scale_tril = LA.cholesky(covariance_matrix)  # pylint: disable=not-callable
        self.centroid = centroid
        self.radius = radius

    def __call__(self, x):
        """Return membership indicator for spherical truncation."""
        diff = x - self.centroid
        dist = ch.sqrt(_batch_mahalanobis(self._unbroadcasted_scale_tril, diff))
        return (dist < self.radius).float().flatten()

    def __str__(self):
        """Return string representation."""
        return "sphere"


#  2D DIMENSIONAL GAUSSIAN LAMBDA FUNCTIONS
def set_two_d(x):
    """Return True if the squared norm of x exceeds 0.5."""
    return x[1].pow(2) + x[0].pow(2) > 0.5


def horseshoe(x):
    """Return True if x lies within the horseshoe region."""
    return x[0] > 0 and x[0] ** 2 + x[1] ** 2 > 1 and x[0] ** 2 + x[1] ** 2 < 2


def horseshoe_dot(x):
    """Return True if x lies within the horseshoe-with-dot region."""
    return (
        x[0] > 0
        and 1 < x.pow(2).sum() < 2
        or ((x[0] - 0.5).pow(2) + x[1].pow(2)) < (1 / 6)
    )


def triangle(x):
    """Return True if x lies within the triangle region (excluding the inner circle)."""
    return (
        x[1] >= 0
        and x[1] <= +x[0] + 1
        and x[1] <= 1 - x[0]
        and not ((x[0] / 2) ** 2 + (x[1] - 0.52) ** 2 <= 0.02)
    )


# 3D DIMENSIONAL GAUSSIAN LAMBDA FUNCTIONS
def three_d_union_check(x):
    """Return True if x satisfies the 3D union membership condition."""
    return x[0] > 0 and x[2] > 0 or x.pow(2).sum() < 1.0


class UnknownGaussian(oracle):  # pylint: disable=too-many-instance-attributes
    """
    Oracle for an unknown Gaussian distribution estimated via Hermite polynomial expansion.
    """

    def __init__(self, k, s):  # pylint: disable=invalid-name
        """
        Args:
            k: maximum Hermite polynomial degree
            s (torch.Tensor): whitened sample set of shape (n, d), assumed N(0, I)
        """
        self.emp_loc = s.mean(0)
        self.emp_cov = cov(s)

        # verify that the s is whitened to N(0, I)
        if (
            ch.norm(self.emp_loc - ch.zeros(s.size(1))) >= 1e-3
            or ch.norm(self.emp_cov - ch.eye(s.size(1))) >= 1e-3
        ):
            raise ValueError(
                f"input dataset must be whitened (eg. N(O, I)). \n"
                f" dataset mean: {self.emp_loc}, covariance matrix: {self.emp_cov}"
            )

        self._emp_dist = MultivariateNormal(self.emp_loc, self.emp_cov)
        self._d = int(s.shape[1])
        self._k = k

        self.herm_poly = NormalizedProbabilistHermitePolynomial(self._k, self._d)
        self._c_v = self._compute_hermite_coefficients(s)  # pylint: disable=invalid-name
        self._dist = None

    def _compute_hermite_coefficients(self, s):  # pylint: disable=invalid-name
        """
        Compute c_v = E_D[H_V(z)] where H_V evaluated at standardized samples from s.
        s should be samples *from the truncated set* (i.e., only samples x in s).
        """
        h_vals = self.herm_poly.h_v(s)  # shape (n, num_indices)
        return h_vals.mean(dim=0)

    def psi_k(self, x):
        """
        Evaluate psi approximation on x (tensor shape (n,d)).
        Returns (n, 1) tensor with values ~[0,1].
        """
        h_vals = self.herm_poly.h_v(x)  # (n, num_indices)
        expansion = (self._c_v * h_vals).sum(dim=1)  # (n,)
        return ch.clamp(expansion, min=0)[..., None]

    def __call__(self, x):
        """Return membership indicator using learned distribution ratio and psi approximation."""
        if self._dist is None:
            raise ValueError("must learn underlying distribution for membership oracle")

        ratio = ch.exp(self._emp_dist.log_prob(x) - self._dist.log_prob(x))[:, None]
        return ((ratio * self.psi_k(x)) > 0.5).float()

    @property
    def emp_dist(self):
        """Return the empirical distribution."""
        return self._emp_dist

    @property
    def dist(self):
        """Return the learned distribution."""
        return self._dist

    @dist.setter
    def dist(self, dist_):
        """Set the learned distribution."""
        self._dist = dist_

    @property
    def c_v(self):  # pylint: disable=invalid-name
        """Return the Hermite coefficients."""
        return self._c_v

    @property
    def d(self):
        """Return the dimensionality."""
        return self._d

    def __str__(self):
        """Return string representation."""
        return "unknown gaussian"


class Identity(oracle):  # pylint: disable=too-few-public-methods
    """
    Identity membership oracle for DNNs. All logits are accepted within the truncation set.
    """

    def __call__(self, x):
        """Return all-ones membership indicator."""
        return ch.ones(x.size()).prod(-1, keepdim=True)

    def __str__(self):
        """Return string representation."""
        return "identity"


class LogitBall(oracle):
    """
    Truncation based off of norm of logits. Logit norm needs to be smaller than input bound.
    In other words, retain the input that the classifier is less certain on. Smaller
    unnormalized log probabilities implies uncertainty in classification.
    """

    def __init__(self, bound):
        """
        Args:
            bound: maximum logit norm for membership
        """
        self.bound = bound

    def __call__(self, x):
        """Return membership indicator for logit ball truncation."""
        return x.norm(dim=-1, keepdim=True) <= self.bound

    def __str__(self):
        """Return string representation."""
        return "logit ball"


class LogitSum(oracle):  # pylint: disable=too-few-public-methods
    """
    Oracle that accepts samples whose logit sum is below a ceiling value.
    """

    def __init__(self, ceiling):
        """
        Args:
            ceiling: upper bound on logit sum for membership
        """
        self.ceiling = ceiling

    def __call__(self, x):
        """Return membership indicator based on logit sum ceiling."""
        return x.sum(1, keepdim=True) < self.ceiling


class TruncateLogit(oracle):  # pylint: disable=too-few-public-methods
    """
    Oracle that excludes samples whose argmax logit matches a specified index.
    """

    def __init__(self, logit):
        """
        Args:
            logit: logit index to exclude
        """
        self.logit = logit

    def __call__(self, x):
        """Return membership indicator excluding the specified argmax logit."""
        return x.argmax(-1, keepdim=True) != self.logit


class RandomTruncation(oracle):  # pylint: disable=too-few-public-methods
    """
    Oracle that randomly accepts samples with probability above a threshold.
    """

    def __init__(self, threshold):
        """
        Args:
            threshold: probability threshold; samples with random draw above this are accepted
        """
        self.threshold = threshold

    def __call__(self, x):
        """Return random membership indicator based on threshold."""
        if x.dim() == 3:
            return ch.rand(x.size(0), x.size(1), 1) > self.threshold
        return ch.rand(x.size(0)) > self.threshold


class LogitBallComplement(oracle):
    """
    Truncation based off of complement norm of logits. Logit norm needs to be greater than input
    bound. In other words, retain the inputs that the classifier is more certain on. Larger
    unnormalized log probabilities implies more certainty in classification.
    """

    def __init__(self, bound, temperature=ch.ones(1)):
        """
        Args:
            bound: minimum logit norm for membership
            temperature: temperature scaling tensor (default ones(1))
        """
        self.bound = bound
        self.temperature = temperature

    def __call__(self, x):
        """Return membership indicator for logit ball complement truncation."""
        x_ = x
        return x_.norm(dim=-1, keepdim=True) >= self.bound

    def __str__(self):
        """Return string representation."""
        return "logit ball complement"


class Sum_Ceiling(oracle):  # pylint: disable=invalid-name,too-few-public-methods
    r"""
    Sums sample from a truncated boolean product distribution and returns True if x \in \S,
    if sum <= ceiling, else False.
    """

    # tensor inputs for vector interval truncation
    def __init__(self, ceil):
        """
        Args:
            ceil: ceiling value for the sum
        """
        self.ceil = ceil

    def __call__(self, x):
        """Return membership indicator based on sum ceiling."""
        return x.sum(1) <= self.ceil


class Sum_Floor(oracle):  # pylint: disable=invalid-name,too-few-public-methods
    r"""
    Sums sample from a truncated boolean product distribution and returns True if x \in \S,
    if sum >= ceiling, else False.
    """

    # tensor inputs for vector interval truncation
    def __init__(self, floor):
        """
        Args:
            floor: floor value for the sum
        """
        self.floor = floor

    def __call__(self, x):
        """Return membership indicator based on sum floor."""
        return x.sum(1) >= self.floor


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
        return ((x[:, :, 1] - x[:, :, 0]) > self.left)[..., None]

    def __str__(self):
        """Return string representation."""
        return "Gumbel Logistic Left Truncation Set"
