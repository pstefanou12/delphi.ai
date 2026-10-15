# Author: pstefanou12@
"""Inverse-time learning-rate schedule."""

from typing import Union

import torch as ch
from torch.optim import lr_scheduler


class InverseTimeLR(lr_scheduler.LRScheduler):
    """Decay the learning rate as ``lr / (1 + t / tau)``.

    Geometric schedules such as StepLR sum to a finite total step, so they can
    freeze before reaching the optimum. A 1/t decay satisfies the
    Robbins-Monro conditions -- the steps sum to infinity while their squares
    do not -- which is what clears the noise floor a constant step leaves
    behind.

    Attributes:
        tau: Steps over which the rate falls by half.
    """

    def __init__(self, optimizer, tau: float, last_epoch: int = -1):
        """Initialize InverseTimeLR.

        Args:
            optimizer: Optimizer whose parameter groups are scheduled.
            tau: Decay constant; larger values decay more slowly.
            last_epoch: Index of the last epoch, per the torch convention.

        Raises:
            ValueError: If tau is not positive.
        """
        if tau <= 0.0:
            raise ValueError(f"tau must be positive; got {tau}.")
        self.tau = tau
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[Union[float, ch.Tensor]]:
        """Return the decayed rate for each parameter group."""
        return [base / (1.0 + self.last_epoch / self.tau) for base in self.base_lrs]
