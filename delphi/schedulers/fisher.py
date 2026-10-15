# Author: pstefanou12@
"""Learning-rate schedule derived from a model's Hessian."""

import torch as ch

from delphi.schedulers.inverse_time import InverseTimeLR


class FisherLR(InverseTimeLR):
    """Set both the initial rate and its decay from the Hessian spectrum.

    Starts at ``1 / smoothness``, the largest step that keeps stochastic
    gradient descent stable, and decays as ``1 / (mu * t)``. Writing that as
    ``1 / (smoothness + mu * t)`` shows it is ``InverseTimeLR`` with
    ``lr = 1 / smoothness`` and ``tau = smoothness / mu``, so the decay itself
    is inherited.

    For a truncated exponential family the Hessian is ``Cov[T(z)]``, making
    both constants properties of the model rather than values to tune.

    Attributes:
        mu: Strong-convexity constant, the smallest Hessian eigenvalue.
        smoothness: Smoothness constant, the largest Hessian eigenvalue.
    """

    def __init__(self, optimizer, mu: float, smoothness: float, last_epoch: int = -1):
        """Initialize FisherLR.

        Args:
            optimizer: Optimizer whose parameter groups are scheduled.
            mu: Smallest Hessian eigenvalue.
            smoothness: Largest Hessian eigenvalue.
            last_epoch: Index of the last epoch, per the torch convention.

        Raises:
            ValueError: If either constant is not positive, or mu exceeds
                smoothness.
        """
        if mu <= 0.0 or smoothness <= 0.0:
            raise ValueError(
                f"mu and smoothness must be positive; got {mu} and {smoothness}."
            )
        if mu > smoothness:
            raise ValueError(f"mu cannot exceed smoothness; got {mu} and {smoothness}.")
        self.mu = mu
        self.smoothness = smoothness
        # The theory fixes the starting rate, so whatever lr the optimizer was
        # built with is replaced rather than scaled. This has to happen before
        # LRScheduler.__init__, which captures base_lrs and applies the first
        # rate immediately.
        for group in optimizer.param_groups:
            group["lr"] = 1.0 / smoothness
            group.pop("initial_lr", None)
        super().__init__(optimizer, smoothness / mu, last_epoch)

    @staticmethod
    def spectrum(hessian: ch.Tensor, independent: bool = True) -> tuple[float, float]:
        """Return the constants this schedule needs from a Hessian estimate.

        Args:
            hessian: Hessian, shaped (k, k).
            independent: Whether the coordinates are independent, in which
                case the off-diagonal entries are sampling noise and only the
                diagonal carries curvature.

        Returns:
            Tuple of (mu, smoothness).

        Raises:
            ValueError: If the estimate has no positive curvature.
        """
        # Eigenvalues of a sampled near-diagonal matrix can fall below the
        # true minimum, and mu sets the decay constant, so a product family
        # reads its diagonal instead.
        values = (
            hessian.diagonal() if independent else ch.linalg.eigvalsh(hessian)  # pylint: disable=not-callable
        )
        mu, smoothness = float(values.min()), float(values.max())
        if mu <= 0.0:
            raise ValueError(
                f"Hessian estimate has non-positive curvature {mu}; "
                "the objective is flat in at least one direction."
            )
        return mu, smoothness
