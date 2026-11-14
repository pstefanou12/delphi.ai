# Author: pstefanou12@
"""Truncated classification: a latent-variable classifier seen through phi.

A latent ``z = h_W(x) + e`` labels the sample, and the pair (x, y) is kept
with probability ``phi(z)``. The descent minimizes the truncated label NLL
in R_alpha, the weights at which every sample keeps mass alpha on each
label; the classifiers differ in the latent's noise and labeling.
"""

import abc
from collections.abc import Callable
import math
from typing import Generic, Optional, TypeVar

import torch as ch
import torch.optim

from delphi.configs.stats.truncated import classification
import delphi.projection_sets
from delphi.stats import linear_model
from delphi.stats.truncated.truncated_linear_model import TruncatedLinearModel
from delphi.utils import constants

_ClassificationConfigT = TypeVar(
    "_ClassificationConfigT", bound=classification.TruncatedClassificationConfig
)

# Above this radius e^-radius underflows float64, so a set that still
# leaves out the start never will.
_MAX_START_RADIUS = 700.0
_NAIVE_ITERATIONS = 200


class TruncatedClassification(
    TruncatedLinearModel[_ClassificationConfigT],
    Generic[_ClassificationConfigT],
    abc.ABC,
):
    """Truncated classification with a known noise law.

    A subclass sets ``loss``, the criterion, and ``default_sampler``, and
    supplies the untruncated NLL its start minimizes, each sample's label
    masses, its form of R_alpha and the label its latent means predict.
    The training log and ``summary`` report the accuracy of those labels on
    the observed sample.

    Attributes:
        sampler: Estimates the per-label masses and scores of the latent.
    """

    loss: type[ch.autograd.Function]
    # Built with num_samples when the caller passes no sampler.
    default_sampler: Callable[[int], object]

    def __init__(  # pylint: disable=too-many-arguments
        self,
        phi: Callable,
        alpha: Optional[float] = None,
        *,
        hyperparameters: Optional[dict] = None,
        sampler: Optional[object] = None,
        fit_intercept: bool = True,
        emp_weight: Optional[ch.Tensor] = None,
        rand_seed: int = 0,
    ):
        """Initialize the model.

        Args:
            phi: Truncation oracle over the latent, returning a mask or a
                survival probability.
            alpha: Lower bound on every training sample's mass on each
                label, ``P(z in S, label y)`` at the truth; None if unknown.
            sampler: Estimates the loss's per-label moments;
                ``default_sampler`` at ``num_samples`` by default.
            fit_intercept: Whether to fit an intercept.
            emp_weight: Start in place of the naive fit, which ignores
                phi; see ``TruncatedLinearModel``.
            rand_seed: Seed applied at the start of every run.
            hyperparameters: Hyperparameter dict.

        Raises:
            ValueError: If alpha exceeds 1/2, which no two label masses
                summing to at most 1 can both reach.
        """
        if alpha is not None and alpha > 0.5:
            raise ValueError(f"alpha bounds every label mass, so <= 1/2; got {alpha}.")
        super().__init__(
            phi,
            alpha,
            fit_intercept,
            emp_weight,
            rand_seed,
            hyperparameters=hyperparameters,
        )
        self.sampler = sampler or type(self).default_sampler(
            self.hyperparameters.num_samples
        )
        self.criterion = self.loss.apply
        self._batch_accuracy: Optional[float] = None

    def criterion_parameters(self, X: ch.Tensor, y: ch.Tensor) -> list:  # pylint: disable=invalid-name
        return [self.phi, self.sampler, self.hyperparameters.eps]

    def forward(self, X: ch.Tensor) -> ch.Tensor:  # pylint: disable=invalid-name
        """Return the loss's latent means for each row of the design matrix."""
        return self._latent_mean(X @ self.weight)

    def _latent_mean(self, linear: ch.Tensor) -> ch.Tensor:
        """Return the latent means from ``X W``; the identity unless overridden."""
        return linear

    @abc.abstractmethod
    def _predicted_label(self, latent: ch.Tensor) -> ch.Tensor:
        """Return the most likely label at latent means latent, shaped (n, 1)."""

    def _accuracy(self, latent: ch.Tensor, y: ch.Tensor) -> float:
        """Return the share of labels y the latent means predict."""
        return float((self._predicted_label(latent) == y.long()).double().mean())

    @abc.abstractmethod
    def _untruncated_nll(self, latent: ch.Tensor, y: ch.Tensor) -> ch.Tensor:
        """Return the mean NLL without truncation at latent means latent."""

    @abc.abstractmethod
    def _label_mass(self, linear: ch.Tensor) -> ch.Tensor:
        """Return each sample's per-label mass R_alpha bounds, shaped (n, C).

        Args:
            linear: ``X W`` at the weight, shaped (n, k).
        """

    @abc.abstractmethod
    def _r_alpha(
        self,
        X: ch.Tensor,  # pylint: disable=invalid-name
        anchor: ch.Tensor,
        radius: float,
    ) -> delphi.projection_sets.ProjectionSet:
        """Return R_alpha at radius ``log(1 / alpha')``, holding anchor."""

    def _calc_emp_model(self, X: ch.Tensor, y: ch.Tensor) -> None:  # pylint: disable=invalid-name
        """Set emp_weight to the naive fit unless given, and register it."""
        if self.emp_weight is None:
            self.emp_weight = self._naive_fit(X, y) / self.covariate_scale
        self.register_weight(self.emp_start)

    def _naive_fit(self, X: ch.Tensor, y: ch.Tensor) -> ch.Tensor:  # pylint: disable=invalid-name
        """Return the untruncated MLE, which ignores phi, shaped (d + 1, k)."""
        design = X.double()
        k = linear_model.require_fitted(self.k, "k")
        weight = ch.zeros(X.size(1), k, dtype=ch.float64, requires_grad=True)
        optimizer = torch.optim.LBFGS(
            [weight],
            max_iter=_NAIVE_ITERATIONS,
            tolerance_grad=1e-9,
            line_search_fn="strong_wolfe",
        )

        def closure() -> ch.Tensor:
            optimizer.zero_grad()
            loss = self._untruncated_nll(self._latent_mean(design @ weight), y)
            loss.backward()
            return loss

        optimizer.step(closure)
        return weight.detach().float()

    def compute_loss(self, batch, criterion=None) -> ch.Tensor:
        """Average the NLL, which is per sample, over the batch."""
        X, y = batch  # pylint: disable=invalid-name
        with ch.no_grad():
            self._batch_accuracy = self._accuracy(self(X), y)
        return super().compute_loss(batch, criterion).mean()

    def batch_metrics(self) -> dict:
        """Return the accuracy on the last batch at the current iterate."""
        if self._batch_accuracy is None:
            return {}
        return {"accuracy": self._batch_accuracy}

    def default_radius(self) -> float:
        """Return ``2 log(1 / alpha)``, or the start's when alpha is unknown.

        R_alpha holds the truth, and so lies on the boundary at alpha =
        alpha*; R_{alpha^2} holds the sample's MLE around it as well.
        """
        if self.alpha is not None:
            return 2.0 * math.log(1.0 / self.alpha)
        X, _ = linear_model.require_fitted(self._data, "data")  # pylint: disable=invalid-name
        mass = self._label_mass(X @ self.emp_start).min()
        return min(-math.log(max(float(mass), 1e-300)), _MAX_START_RADIUS)

    def _constrained_projection_set(
        self,
        X: ch.Tensor,  # pylint: disable=invalid-name
        y: ch.Tensor,
        radius: float,
    ) -> delphi.projection_sets.ProjectionSet:
        """Return R_alpha anchored at emp_weight, its radius grown to hold it.

        Raises:
            ValueError: If no radius holds emp_weight, as when phi leaves a
                label no mass at its latent means.
        """
        del y
        anchor = self.emp_start
        projection = self._r_alpha(X, anchor, radius)
        while not projection.contains(anchor):
            if radius >= _MAX_START_RADIUS:
                raise ValueError(
                    "no R_alpha holds emp_weight: phi leaves a label no mass "
                    "at its latent means."
                )
            radius = min(radius * self.hyperparameters.rate, _MAX_START_RADIUS)
            projection.radius = radius
            self.logger.warning(f"emp_weight lies outside R_alpha; r -> {radius:.4f}.")
        return projection

    def _summary_estimates(self) -> list[tuple[str, str]]:
        X, y = linear_model.require_fitted(self._data, "data")  # pylint: disable=invalid-name
        k = linear_model.require_fitted(self.k, "k")
        weight = linear_model.require_fitted(
            self._estimate(constants.Estimator.AVG), "coef_"
        )
        weight = weight.reshape(-1, k)
        accuracy = self._accuracy(self._latent_mean(X @ weight), y)
        rows = [("accuracy of coef_", f"{accuracy:.4f} on the observed sample")]
        return rows + super()._summary_estimates()

    def _hessian(self, X: ch.Tensor, weight: ch.Tensor) -> ch.Tensor:  # pylint: disable=invalid-name
        """Return the NLL's Fisher information."""
        return self.loss.hessian(  # type: ignore[attr-defined]
            self._latent_mean(X @ weight), X, self.phi, self.sampler
        )
