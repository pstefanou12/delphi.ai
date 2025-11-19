"""
Truncated normal distribution without oracle access (ie. unknown truncation set)
"""

from typing import Callable, Optional
import torch as ch

from .unknown_truncated_multivariate_normal import UnknownTruncationMultivariateNormal 
from ..utils.helpers import Parameters


class UnknownTruncationNormal(UnknownTruncationMultivariateNormal):
    """
    Truncated normal distribution class.
    """
    def __init__(self,
                 args: Parameters,
                 k: int,
                 alpha: float,
                 dims: int, 
                 covariance_matrix: Optional[ch.Tensor] = None):
        super().__init__(args, k, alpha, dims, covariance_matrix=covariance_matrix)

    @property 
    def variance_(self): 
        """
        Returns the standard deviation for the normal distribution.
        """
        return self.unknown_truncated.model.covariance_matrix.clone()
