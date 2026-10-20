# Author: pstefanou12@
"""Learning-rate schedule derived from a model's Hessian."""

from collections.abc import Callable
from typing import Optional

import torch as ch

from delphi.optim.lr_scheduler.inverse_time import InverseTimeLR


class FisherLR(InverseTimeLR):
    """Set both the initial rate and its decay from the Hessian spectrum.

    Starts at ``1 / smoothness``, the largest step that keeps stochastic
    gradient descent stable, and decays as ``1 / (mu * t)``. Writing that as
    ``1 / (smoothness + mu * t)`` shows it is ``InverseTimeLR`` with
    ``lr = 1 / smoothness`` and ``tau = smoothness / mu``, so the decay itself
    is inherited.

    For a truncated exponential family the Hessian is ``Cov[T(z)]``, making
    both constants properties of the model rather than values to tune.

    Measured once, at the initialization, they can be off by an order of
    magnitude by the time the iterate reaches the optimum. Given a
    ``curvature`` callable the schedule re-measures them every
    ``refresh_every`` steps at the current iterate, keeping ``t``, so the
    decay continues rather than restarting.

    Attributes:
        mu: Strong-convexity constant, the smallest Hessian eigenvalue.
        smoothness: Smoothness constant, the largest Hessian eigenvalue.
        curvature: Returns fresh (mu, smoothness) at the current iterate, or
            raises ValueError when it cannot.
        refresh_every: Steps between refreshes; 0 never refreshes.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        optimizer,
        mu: float,
        smoothness: float,
        last_epoch: int = -1,
        curvature: Optional[Callable[[], tuple[float, float]]] = None,
        refresh_every: int = 0,
    ):
        """Initialize FisherLR.

        Args:
            optimizer: Optimizer whose parameter groups are scheduled.
            mu: Smallest Hessian eigenvalue.
            smoothness: Largest Hessian eigenvalue.
            last_epoch: Index of the last epoch, per the torch convention.
            curvature: Returns fresh (mu, smoothness); None never refreshes.
            refresh_every: Steps between refreshes; 0 never refreshes.

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
        if refresh_every < 0:
            raise ValueError(
                f"refresh_every must be non-negative; got {refresh_every}."
            )
        self.mu = mu
        self.smoothness = smoothness
        # Before LRScheduler.__init__, which calls step() once.
        self.curvature = curvature
        self.refresh_every = refresh_every
        # Before LRScheduler.__init__, which captures base_lrs and applies
        # the first rate immediately.
        for group in optimizer.param_groups:
            group["lr"] = 1.0 / smoothness
            group.pop("initial_lr", None)
        super().__init__(optimizer, smoothness / mu, last_epoch)

    def step(self, epoch=None):  # pylint: disable=arguments-renamed
        """Refresh the constants when one is due, then advance the schedule."""
        upcoming = self.last_epoch + 1
        if (
            self.curvature is not None
            and self.refresh_every
            and upcoming > 0
            and upcoming % self.refresh_every == 0
        ):
            self._refresh()
        super().step(epoch)

    def _refresh(self) -> None:
        """Re-measure the constants, keeping the old ones if that fails."""
        try:
            mu, smoothness = self.curvature()  # type: ignore[misc]
        except ValueError:
            return
        if not 0.0 < mu <= smoothness:
            return
        self.mu, self.smoothness = mu, smoothness
        self.tau = smoothness / mu
        self.base_lrs = [1.0 / smoothness for _ in self.base_lrs]

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
        # A sampled near-diagonal matrix can show eigenvalues below the true
        # minimum, and mu sets the decay constant.
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
