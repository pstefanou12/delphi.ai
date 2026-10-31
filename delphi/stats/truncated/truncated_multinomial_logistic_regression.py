# Author: pstefanou12@
"""Truncated multinomial logistic regression: K classes seen through phi.

The latent framework of Ilyas, Zampetakis and Daskalakis, "A Theoretical
and Practical Framework for Regression and Classification from Truncated
Samples" (AISTATS 2020), extended from binary labels to K classes. The
latents ``z_k = w_k^T x + b_k + g_k``, for i.i.d. standard Gumbel g, label
the sample ``y = argmax_k z_k``, so that without truncation
``P(y = k | x)`` is the softmax of the logits; the pair (x, y) is kept with
probability ``phi(z)``.

Class 0 is the reference: its weights and intercept are fixed at zero, so
its latent is pure noise. Without truncation the logits are identified only
up to a shared shift, which this fixes; phi sees the latents in these
coordinates.
"""

from typing import Optional

import torch as ch
import torch.nn.functional

from delphi import configs
import delphi.projection_sets
from delphi.stats import linear_model
from delphi.stats.truncated import losses
from delphi.stats.truncated.truncated_classification import (
    TruncatedClassification,
)
from delphi.truncated import sampling


class TruncatedMultinomialLogisticRegression(
    TruncatedClassification[configs.TruncatedMultinomialLogisticRegressionConfig]
):
    """Truncated multinomial logistic regression over labels 0, ..., K - 1.

    ``coef_`` and ``intercept_`` hold classes 1 to K - 1, shaped (K - 1, d)
    and (K - 1,); class 0's are zero. ``phi`` sees latents shaped
    (n, m, K), and alpha is at most 1 / K.

    The projection set bounds every sample's untruncated label
    probabilities, which bound its truncated label masses above: a convex
    superset of R_alpha that needs no sampling, where R_alpha itself would
    need its masses searched over K - 1 latent dimensions.
    """

    defaults = configs.TruncatedMultinomialLogisticRegressionConfig
    loss = losses.TruncatedCE
    sampler: sampling.GumbelSampler
    default_sampler = sampling.StratifiedGumbelSampler

    def __str__(self):
        """Return a human-readable name for this model."""
        return "truncated multinomial logistic regression"

    @staticmethod
    def _check_inputs(X: ch.Tensor, y: ch.Tensor) -> None:  # pylint: disable=invalid-name
        """Raise unless y holds every label 0, ..., K - 1, in one column."""
        TruncatedClassification._check_inputs(X, y)
        integral = not y.is_floating_point() or bool((y == y.round()).all())
        if y.size(1) != 1 or not integral or y.min() < 0:
            raise ValueError("expected labels in {0, ..., K - 1} shaped (n, 1).")
        if ch.unique(y).numel() != int(y.max()) + 1 or y.max() == 0:
            raise ValueError("y must hold every label 0, ..., K - 1, for K >= 2.")

    def _calc_emp_model(self, X: ch.Tensor, y: ch.Tensor) -> None:  # pylint: disable=invalid-name
        """Fit classes 1 to K - 1, then start from the naive fit.

        Raises:
            ValueError: If alpha exceeds 1 / K, which K label masses summing
                to at most 1 cannot all reach.
        """
        self.k = int(y.max())
        if self.alpha is not None and self.alpha * (self.k + 1) > 1.0:
            raise ValueError(
                f"alpha bounds all {self.k + 1} label masses, so <= "
                f"1/{self.k + 1}; got {self.alpha}."
            )
        super()._calc_emp_model(X, y)

    def _latent_mean(self, linear: ch.Tensor) -> ch.Tensor:
        """Return the K logits, class 0's fixed at zero."""
        return torch.nn.functional.pad(linear, (1, 0))

    def _untruncated_nll(self, latent: ch.Tensor, y: ch.Tensor) -> ch.Tensor:
        return torch.nn.functional.cross_entropy(latent, y.flatten().long())

    def _label_mass(self, linear: ch.Tensor) -> ch.Tensor:
        """Return each sample's untruncated label probabilities."""
        return ch.softmax(self._latent_mean(linear).double(), dim=1)

    def _projection_set(
        self,
        X: ch.Tensor,  # pylint: disable=invalid-name
        anchor: ch.Tensor,
        radius: float,
    ) -> delphi.projection_sets.ProjectionSet:
        return LabelProbabilitySet(
            X, anchor, radius, self.hyperparameters.bisection_steps
        )

    def hessian(self, theta: Optional[ch.Tensor] = None) -> ch.Tensor:
        """Return the NLL's Fisher information in classes 1 to K - 1."""
        k = linear_model.require_fitted(self.k, "k")
        full = super().hessian(theta)
        d = full.size(0) // (k + 1)
        free = full.reshape(d, k + 1, d, k + 1)[:, 1:, :, 1:]
        return free.reshape(d * k, d * k)

    def predict_proba(self, X: ch.Tensor) -> ch.Tensor:  # pylint: disable=invalid-name
        """Return ``P(y = k | x)`` without truncation, shaped (n, K).

        Args:
            X: Covariates, shaped (n, d).
        """
        return ch.softmax(self._latent_mean(super().predict(X)), dim=1)

    def predict(self, X: ch.Tensor) -> ch.Tensor:  # pylint: disable=invalid-name
        """Return the most likely label without truncation, shaped (n, 1).

        Args:
            X: Covariates, shaped (n, d).
        """
        return self._latent_mean(super().predict(X)).argmax(1, keepdim=True)

    def _summary_data(self) -> list[tuple[str, str]]:
        return [
            ("classes", str(linear_model.require_fitted(self.k, "k") + 1))
            if name == "outputs"
            else (name, value)
            for name, value in super()._summary_data()
        ]


class LabelProbabilitySet(delphi.projection_sets.ProjectionSet):
    """Weights at which every sample keeps probability e^-radius per label.

    The probabilities are the untruncated softmax of the logits
    ``(0, x_i^T W)``, whose logarithm is concave in W, so the set is
    convex.
    """

    def __init__(
        self,
        X: ch.Tensor,  # pylint: disable=invalid-name
        anchor: ch.Tensor,
        radius: float,
        bisection_steps: int,
    ):
        """Build the set over the training covariates.

        Args:
            X: Design matrix, shaped (n, d).
            anchor: Weight of shape (d, K - 1) to retract towards.
            radius: The set's ``log(1 / alpha')``.
            bisection_steps: Halvings of the segment a projection searches.
        """
        self._design = X.double()
        super().__init__(anchor, radius, bisection_steps)

    def contains(self, weight: ch.Tensor) -> bool:
        linear = self._design @ weight.detach().double().reshape(self.anchor.shape)
        logits = torch.nn.functional.pad(linear, (1, 0))
        radius = 0.0 if self.radius is None else self.radius
        return bool(ch.log_softmax(logits, dim=1).min() >= -radius)
