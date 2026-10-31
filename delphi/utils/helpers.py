# Author: pstefanou12@
"""
Helper code (functions, classes, etc.)
"""

from collections.abc import Callable
import math
import pprint
from typing import Optional

import cox
import torch as ch
import torch.linalg as LA


class Parameters:
    """Parameters class: a thin wrapper around a dictionary.

    Example::

        ps = Parameters({"a": 1, "b": 3})
        ps.a  # returns 1
    """

    def __init__(self, params):
        super().__setattr__("params", params)

        # ensure no overlapping (in case) params
        collisions = set()
        for k in self.params.keys():
            collisions.add(k.lower())

        assert len(collisions) == len(self.params.keys())

    def as_dict(self):
        """Return the underlying params dictionary."""
        return self.params

    def __getattr__(self, x):
        if x in vars(self):
            return vars(self)[x]

        if x not in self.params:
            return None

        return self.params[x]

    def __setattr__(self, x, v):
        # Fix for some back-compatibility with some pickling bugs
        if x == "params":
            super().__setattr__(x, v)
            return

        if x in vars(self):
            vars(self)[x] = v

        self.params[x] = v

    def __delattr__(self, key):
        del self.params[key]

    def __iter__(self):
        return iter(self.params)

    def __len__(self):
        return len(self.params)

    def __str__(self):
        pp = pprint.PrettyPrinter()
        return pp.pformat(self.params)

    def __repr__(self):
        return str(self)

    def __getstate__(self):
        return self.params

    def __contains__(self, x):
        return x in self.params

    def __setstate__(self, x):
        super().__setattr__("params", x)  # pylint: disable=attribute-defined-outside-init


def vech_size(dims: int) -> int:
    """Return the length of ``vech`` for a ``dims x dims`` matrix.

    Args:
        dims: Side length of the matrix.

    Returns:
        ``dims * (dims + 1) / 2``.
    """
    return dims * (dims + 1) // 2


def vech_side(size: int) -> int:
    """Return the side of the matrix whose ``vech`` has ``size`` entries.

    Args:
        size: Length of the lower triangle, as ``vech_size`` returns.

    Returns:
        The side length, inverting ``vech_size``.
    """
    return int(round((math.sqrt(8 * size + 1) - 1) / 2))


def vech(matrix: ch.Tensor) -> ch.Tensor:
    """Return the lower triangle of a symmetric matrix as a vector.

    Args:
        matrix: Symmetric matrix of shape (..., dims, dims).

    Returns:
        Vectors of shape (..., ``vech_size(dims)``).
    """
    rows, cols = ch.tril_indices(matrix.size(-1), matrix.size(-1))
    return matrix[..., rows, cols]


def unvech(vector: ch.Tensor, dims: int) -> ch.Tensor:
    """Rebuild a symmetric matrix from its lower triangle.

    Args:
        vector: Lower triangle, as produced by ``vech``, shaped (..., size).
        dims: Side length of the matrix.

    Returns:
        Symmetric matrices of shape (..., dims, dims).
    """
    rows, cols = ch.tril_indices(dims, dims)
    matrix = vector.new_zeros(vector.shape[:-1] + (dims, dims))
    matrix[..., rows, cols] = vector
    matrix[..., cols, rows] = vector
    return matrix


def whitened_hessian(
    hessian: Callable[..., ch.Tensor], factor: ch.Tensor
) -> Callable[..., ch.Tensor]:
    """Return ``hessian`` in the coordinates ``factor`` whitens.

    For a preconditioner ``C = L L^T`` and ``theta = L^{-T} psi``, the chain
    rule gives the Hessian in psi as ``L^{-1} H L^{-T}``; it is the identity
    where ``H = C``, so its spectrum measures how far C is from H.

    Args:
        hessian: Maps a parameter, or None for the current one, to H.
        factor: The lower Cholesky factor L of C.

    Returns:
        The whitened Hessian, with hessian's signature.
    """

    def whitened(theta: Optional[ch.Tensor] = None) -> ch.Tensor:
        # Two triangular solves: L^{-1} H, then L^{-1} (L^{-1} H)^T, since
        # H is symmetric.
        left = ch.linalg.solve_triangular(factor, hessian(theta), upper=False)  # pylint: disable=not-callable
        return ch.linalg.solve_triangular(factor, left.T, upper=False)  # pylint: disable=not-callable

    return whitened


def setup_store_with_metadata(args, store):
    """Set up a store for training by recording all args as metadata."""
    args_dict = args.as_dict()
    schema = cox.store.schema_from_dict(args_dict)
    store.add_table("metadata", schema)
    store["metadata"].append_row(args_dict)


def has_attr(obj, k):
    """Check both that obj.k exists and is not equal to None."""
    try:
        return k in obj
    except KeyError:
        return False
    except AttributeError:
        return False


def accuracy(output, target, topk=(1,), exact=False):
    """Compute top-k accuracy for the specified values of k.

    Args:
        output (Tensor): model output of shape (N, classes) or (N, attributes)
            for sigmoid/multitask binary classification.
        target (Tensor): correct labels of shape (N,) for multiclass or
            (N, attributes) for multitask binary classification.
        topk (tuple): for each item ``k`` in this tuple, return the top-k
            accuracy.
        exact (bool): if False return aggregate statistics; if True return
            per-example correctness.

    Returns:
        A list of top-k accuracies (or per-example correctness if exact=True).
    """
    with ch.no_grad():
        # Binary Classification
        if len(target.shape) > 1:
            assert output.shape == target.shape, (
                "Detected binary classification but output shape != target shape"
            )
            return [
                ch.round(ch.sigmoid(output)).eq(ch.round(target)).float().mean(),
                -1.0,
            ]
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        res_exact = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float()
            ck_sum = correct_k.sum(0, keepdim=True)
            res.append(ck_sum.mul_(100.0 / batch_size))
            res_exact.append(correct_k)
        if not exact:
            return res
        return res_exact


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        """Initialize and reset the meter."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        """Reset all tracked values to zero."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Update meter with a new value and optional count."""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calc_spectral_norm(A):  # pylint: disable=invalid-name
    """Calculate the spectral norm (largest singular value) of matrix A."""
    _, s, _ = LA.svd(A)  # pylint: disable=not-callable
    return s.max()


def calc_thickness(X):  # pylint: disable=invalid-name
    """Return the thickness (minimum real eigenvalue) of a PSD matrix X."""
    return LA.eig(X).eigenvalues.real.min()  # pylint: disable=not-callable
