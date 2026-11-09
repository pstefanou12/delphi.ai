# Author: pstefanou12@
"""Truncated multivariate normal estimation, with or without the truncation set.

``TruncatedMultivariateNormal`` picks the estimator from what the caller
knows:

- ``phi`` given: the set is a membership oracle, and
  ``KnownTruncationMultivariateNormal`` fits by projected SGD on the
  truncated NLL.
- ``phi=None``: the set is unknown. With ``method="hermite"``, the default,
  ``UnknownTruncationMultivariateNormal`` fits from the sample's Hermite
  expansion, which also recovers the set. With ``method="l1"``,
  ``KnownTruncationMultivariateNormal`` first learns the set by L1 polynomial
  regression, then fits as if it were known.
- ``covariance_matrix`` given: either one with the covariance fixed, learning
  only the mean.
"""

from collections.abc import Callable
from typing import Literal, Optional, Union, cast

import torch as ch

from delphi.truncated.distributions import (
    known_truncated_multivariate_normal as known_mvn,
)
from delphi.truncated.distributions import (
    known_truncated_multivariate_normal_known_covariance as known_mvn_known_cov,
)
from delphi.truncated.distributions import (
    unknown_truncated_multivariate_normal as unknown_mvn,
)
from delphi.truncated.distributions import (
    unknown_truncated_multivariate_normal_known_covariance as unknown_mvn_known_cov,
)

Estimator = Union[
    known_mvn.KnownTruncationMultivariateNormal,
    known_mvn_known_cov.KnownTruncationMultivariateNormalKnownCovariance,
    unknown_mvn.UnknownTruncationMultivariateNormal,
    unknown_mvn_known_cov.UnknownTruncationMultivariateNormalKnownCovariance,
]

Method = Literal["hermite", "l1"]


class TruncatedMultivariateNormal:
    """Estimates a multivariate normal from samples truncated to a set.

    Every mode reports the same estimates: ``loc_`` and
    ``covariance_matrix_`` at the averaged iterate, and the ``avg_``,
    ``ema_``, ``final_`` and ``best_`` variants of each. With a known
    covariance, every ``covariance_matrix_`` is the one given.

    Attributes:
        phi: The truncation set's membership oracle, or None when unknown.
        covariance_matrix: The known covariance, or None when learned.
        method: How an unknown set is handled: ``"hermite"`` or ``"l1"``.
        estimator_: The estimator the arguments select, which fits the
            sample and holds every diagnostic.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        *,
        phi: Optional[Callable] = None,
        alpha: Optional[float] = None,
        covariance_matrix: Optional[ch.Tensor] = None,
        method: Method = "hermite",
        hyperparameters: Optional[dict] = None,
    ):
        """Initialize TruncatedMultivariateNormal.

        Args:
            phi: Membership oracle of the truncation set; None when the set
                is unknown.
            alpha: Survival mass of the set, if known.
            covariance_matrix: The known covariance, shaped (dims, dims);
                None learns it.
            method: The estimator for an unknown set: ``"hermite"`` fits the
                sample's Hermite expansion; ``"l1"`` learns the set by L1
                polynomial regression, then fits as if it were known.
                Ignored when phi is given.
            hyperparameters: Overrides of the selected estimator's config.

        Raises:
            ValueError: If covariance_matrix is not square, or method is not
                one of the two.
        """
        if method not in ("hermite", "l1"):
            raise ValueError(f"method must be 'hermite' or 'l1'; got {method!r}.")
        if covariance_matrix is not None and (
            covariance_matrix.dim() != 2
            or covariance_matrix.size(0) != covariance_matrix.size(1)
        ):
            raise ValueError(
                "covariance_matrix must be square; got "
                f"{tuple(covariance_matrix.shape)}."
            )
        self.phi = phi
        self.covariance_matrix = covariance_matrix
        self.method = method
        self.estimator_: Estimator = self._select(alpha, hyperparameters)

    def _select(
        self, alpha: Optional[float], hyperparameters: Optional[dict]
    ) -> Estimator:
        """Return the estimator for what the caller knows."""
        phi, covariance = self.phi, self.covariance_matrix
        if (phi is not None or self.method == "l1") and covariance is not None:
            return known_mvn_known_cov.KnownTruncationMultivariateNormalKnownCovariance(
                phi, covariance, alpha, hyperparameters=hyperparameters
            )
        if phi is not None or self.method == "l1":
            return known_mvn.KnownTruncationMultivariateNormal(
                phi, alpha, hyperparameters=hyperparameters
            )
        if covariance is not None:
            return unknown_mvn_known_cov.UnknownTruncationMultivariateNormalKnownCovariance(
                covariance, alpha, hyperparameters=hyperparameters
            )
        return unknown_mvn.UnknownTruncationMultivariateNormal(
            alpha, hyperparameters=hyperparameters
        )

    def fit(self, S: ch.Tensor) -> "TruncatedMultivariateNormal":  # pylint: disable=invalid-name
        """Fit the selected estimator to the truncated sample.

        Args:
            S: Samples from the truncated distribution, shaped (n, dims).

        Returns:
            The fitted model.

        Raises:
            ValueError: If S's width is not the known covariance's side.
        """
        covariance = self.covariance_matrix
        if (
            covariance is not None
            and isinstance(S, ch.Tensor)
            and S.dim() == 2
            and S.size(1) != covariance.size(0)
        ):
            raise ValueError(
                f"covariance_matrix is {covariance.size(0)} by "
                f"{covariance.size(0)}, but the samples have {S.size(1)} columns."
            )
        self.estimator_.fit(S)
        return self

    def truncation_set(self) -> Callable:
        """Return the truncation set: phi when given, else the recovered set.

        Raises:
            RuntimeError: If the set is unknown and the model is not fit.
        """
        if self.phi is not None:
            return self.phi
        if self.method == "l1":
            known = cast(known_mvn.KnownTruncationMultivariateNormal, self.estimator_)
            if known.truncation_set_ is None:
                raise RuntimeError("the truncation set is learned by fit.")
            return known.truncation_set_
        # Without phi or "l1", _select built an unknown-truncation estimator.
        unknown = cast(
            unknown_mvn.UnknownTruncationMultivariateNormalBase, self.estimator_
        )
        return unknown.truncation_set()

    def _covariance(self, name: str) -> ch.Tensor:
        """Return an estimate's covariance, or the known one."""
        if self.covariance_matrix is not None:
            return self.covariance_matrix
        return getattr(self.estimator_, f"{name}_covariance_matrix_")

    @property
    def avg_loc_(self) -> ch.Tensor:
        """Mean at the averaged iterate."""
        return self.estimator_.avg_loc_

    @property
    def ema_loc_(self) -> ch.Tensor:
        """Mean at the moving average of the iterates."""
        return self.estimator_.ema_loc_

    @property
    def final_loc_(self) -> ch.Tensor:
        """Mean at the last iterate."""
        return self.estimator_.final_loc_

    @property
    def best_loc_(self) -> ch.Tensor:
        """Mean at the iterate with the lowest validation loss."""
        return self.estimator_.best_loc_

    @property
    def loc_(self) -> ch.Tensor:
        """Mean at the averaged iterate."""
        return self.avg_loc_

    @property
    def avg_covariance_matrix_(self) -> ch.Tensor:
        """Covariance at the averaged iterate."""
        return self._covariance("avg")

    @property
    def ema_covariance_matrix_(self) -> ch.Tensor:
        """Covariance at the moving average of the iterates."""
        return self._covariance("ema")

    @property
    def final_covariance_matrix_(self) -> ch.Tensor:
        """Covariance at the last iterate."""
        return self._covariance("final")

    @property
    def best_covariance_matrix_(self) -> ch.Tensor:
        """Covariance at the iterate with the lowest validation loss."""
        return self._covariance("best")

    @property
    def covariance_matrix_(self) -> ch.Tensor:
        """Covariance at the averaged iterate."""
        return self.avg_covariance_matrix_

    def __str__(self):
        """Return a human-readable name for this distribution."""
        return str(self.estimator_)
