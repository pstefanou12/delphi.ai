# Author: pstefanou12@
"""Abstract truncated linear model.

The model is ``z = h_W(x) + e`` with ``e ~ D_N``, observed only where
``phi`` admits it. A subclass names the noise ``D_N``, its untruncated MLE
and the convex set the descent is kept in; the truncated NLL, its Hessian,
projected SGD through the trainer, the estimates and the report live here.
"""

import abc
from collections.abc import Callable
import math
from typing import Generic, Optional, TypeVar

import torch as ch
import torch.distributions
import torch.utils.data

from delphi import configs
from delphi.stats import linear_model, summary
from delphi.stats.truncated import losses, projection_sets
import delphi.trainer
from delphi.utils import datasets

_TruncatedConfigT = TypeVar(
    "_TruncatedConfigT", bound=configs.TruncatedLinearModelConfig
)


class TruncatedLinearModel(  # pylint: disable=too-many-instance-attributes
    linear_model.LinearModel[_TruncatedConfigT],
    Generic[_TruncatedConfigT],
    abc.ABC,
):
    """Projected SGD on the truncated NLL of a linear model.

    Subclasses supply ``noise``, ``untruncated_fit`` and, to constrain the
    descent, ``build_projection_set``. A label link (sign, argmax), learned
    noise parameters or dependent data are further subclasses that replace
    the criterion, the parameters or the optimizer respectively.

    Attributes:
        phi: Truncation oracle over latents shaped (n, m, k).
        alpha: Survival probability lower bound of the truncation set.
        rand_seed: Seed applied at the start of every run.
        untruncated_weight: The untruncated MLE, set during fit.
        start_weight: The descent's start, inside the projection set.
        projection_set: The set the iterates are kept in, set during fit.
        n_samples_: Number of observed samples, set during fit.
    """

    # The covariates are correlated, so the Hessian's off-diagonal entries
    # carry curvature and FisherLR reads its eigenvalues.
    independent_coordinates: bool = False

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        hyperparameters: dict,
        phi: Callable,
        alpha: float,
        fit_intercept: bool = True,
        rand_seed: int = 0,
    ):
        """Initialize the truncated linear model.

        Args:
            hyperparameters: Hyperparameter dict.
            phi: Truncation oracle over latents shaped (n, m, k).
            alpha: Survival probability in (0, 1].
            fit_intercept: Whether to fit an intercept.
            rand_seed: Seed applied at the start of every run.

        Raises:
            ValueError: If alpha lies outside (0, 1].
        """
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must lie in (0, 1]; got {alpha}.")
        super().__init__(hyperparameters, fit_intercept=fit_intercept)
        self.phi = phi
        self.alpha = alpha
        self.rand_seed = rand_seed
        self.criterion = losses.TruncatedLinearModelNLL.apply

        self.untruncated_weight: Optional[ch.Tensor] = None
        self.start_weight: Optional[ch.Tensor] = None
        self.projection_set: Optional[projection_sets.ProjectionSet] = None
        self.n_samples_: Optional[int] = None
        self.trainer: Optional[delphi.trainer.Trainer] = None
        self._train_tensors: Optional[tuple[ch.Tensor, ch.Tensor]] = None
        # Running totals, and their values when the current run began, so
        # run_metrics reports each run's own.
        self._projections, self._projection_distance = 0, 0.0
        self._run_start = (0, 0.0)

    @property
    @abc.abstractmethod
    def noise(self) -> torch.distributions.Distribution:
        """The noise D_N, with batch shape (k,) and a differentiable log_prob."""

    def build_projection_set(
        self, design: ch.Tensor, targ: ch.Tensor
    ) -> projection_sets.ProjectionSet:
        """Return the convex set the descent is kept in.

        Args:
            design: Design matrix, shaped (n, d + 1) with an intercept.
            targ: Targets, shaped (n, k).

        Returns:
            The whole space, unless a subclass constrains it.
        """
        del design, targ
        return projection_sets.Unconstrained(self.required_untruncated_weight)

    def fit(self, X: ch.Tensor, y: ch.Tensor):  # pylint: disable=invalid-name
        """Fit the model to the observed (truncated) samples.

        Args:
            X: Covariates, shaped (n, d).
            y: Responses, shaped (n, k).

        Returns:
            The fitted model.

        Raises:
            TypeError: If X or y is not a tensor.
            ValueError: If their shapes do not describe n > d samples.
        """
        self._check_inputs(X, y)
        self.d, self.k = X.size(1), y.size(1)
        self.n_samples_ = X.size(0)
        self.covariate_scale = 1.0
        design = self.design_matrix(X)
        # Daskalakis et al. (2020), (4.1): ||x||_2 <= 1 once divided by
        # B sqrt(d), with B the largest entry, the intercept's 1 included.
        self.covariate_scale = float(design.abs().max()) * math.sqrt(design.size(1))
        design = design / self.covariate_scale
        self._start(design, y)
        self.criterion_params = [
            self.phi,
            self.noise,
            self.hyperparameters.num_samples,
            self.hyperparameters.eps,
        ]
        self._projections, self._projection_distance = 0, 0.0
        train_loader, val_loader = self._loaders(design, y)
        self.trainer = delphi.trainer.Trainer(self, self.hyperparameters)
        self.procedure = self.trainer.train_model(
            train_loader, val_loader, rand_seed=self.rand_seed
        )
        return self

    @staticmethod
    def _check_inputs(X: ch.Tensor, y: ch.Tensor) -> None:  # pylint: disable=invalid-name
        """Raise unless X and y are tensors of n > d paired samples."""
        if not isinstance(X, ch.Tensor) or not isinstance(y, ch.Tensor):
            raise TypeError(f"X and y must be tensors; got {type(X)} and {type(y)}.")
        if X.dim() != 2 or y.dim() != 2 or X.size(0) != y.size(0):
            raise ValueError(
                "expected X shaped (n, d) and y shaped (n, k); got "
                f"{tuple(X.size())} and {tuple(y.size())}."
            )
        if X.size(0) <= X.size(1):
            raise ValueError(
                f"need more samples than dimensions; X is {tuple(X.size())}."
            )

    def _start(self, design: ch.Tensor, targ: ch.Tensor) -> None:
        """Build the projection set and register the projected start."""
        untruncated = self.untruncated_fit(design, targ)
        self.untruncated_weight = untruncated
        projection = (
            self.build_projection_set(design, targ)
            if self.hyperparameters.project
            else projection_sets.Unconstrained(untruncated)
        )
        self.projection_set = projection
        start = (
            ch.zeros_like(untruncated)
            if self.hyperparameters.start == "zero"
            else untruncated
        )
        self.start_weight = projection.project(start).clone()
        self.register_parameter("weight", torch.nn.Parameter(self.start_weight.clone()))
        self._train_tensors = (design, targ)

    def _loaders(self, design: ch.Tensor, targ: ch.Tensor):
        """Return the training loader, and a validation loader if val > 0.

        Without a validation split, every sample feeds the descent and
        ``evaluate`` scores the iterates on the training set instead.
        """
        hp = self.hyperparameters
        if hp.val > 0.0:
            return datasets.make_train_and_val(hp, design, targ)
        batch_size = design.size(0) if hp.batch_size == -1 else hp.batch_size
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(design, targ),
            batch_size=batch_size,
            shuffle=hp.shuffle,
        )
        return train_loader, None

    def evaluate(self) -> dict:
        """Return the truncated NLL of the current iterate on the training set."""
        design, targ = linear_model.require_fitted(self._train_tensors, "data")
        return {
            "loss": self.required_criterion(  # pylint: disable=not-callable
                self(design), targ, *self.criterion_params
            )
        }

    @property
    def required_untruncated_weight(self) -> ch.Tensor:
        """The untruncated MLE, which fit computes first."""
        return linear_model.require_fitted(
            self.untruncated_weight, "untruncated_weight"
        )

    @property
    def theta(self) -> ch.Tensor:
        """The trained weight, which FisherLR re-measures curvature at."""
        return self.weight

    def hessian(self, theta: Optional[ch.Tensor] = None) -> ch.Tensor:
        """Return the Hessian of the truncated NLL, for deriving step sizes.

        Args:
            theta: Weight to evaluate at; defaults to the start.

        Returns:
            The Hessian in the flattened weight.
        """
        weight = (
            linear_model.require_fitted(self.start_weight, "start_weight")
            if theta is None
            else theta
        )
        design, _ = linear_model.require_fitted(self._train_tensors, "data")
        return losses.TruncatedLinearModelNLL.hessian(
            design @ weight,
            design,
            self.phi,
            self.noise,
            self.hyperparameters.num_samples,
            self.hyperparameters.eps,
        )

    def mass_on_set(self, params: ch.Tensor) -> float:
        """Return the mean survival probability the observed x have under params.

        Args:
            params: A flattened weight.

        Returns:
            ``mean_i P(h_W(x_i) + e in S)``, which alpha lower-bounds at the
            truth.
        """
        design, _ = linear_model.require_fitted(self._train_tensors, "data")
        pred = design @ params.reshape(-1, linear_model.require_fitted(self.k, "k"))
        with ch.no_grad():
            kept = self.phi(
                pred[:, None, :]
                + self.noise.sample((pred.size(0), self.hyperparameters.num_samples))
            )
        return float(kept.float().mean())

    def pre_phase_hook(self) -> None:
        """Mark the projection counts at the start of the run."""
        self._run_start = (self._projections, self._projection_distance)

    def step_post_hook(self, optimizer, args, kwargs) -> None:
        """Project the iterate back onto the projection set."""
        with ch.no_grad():
            projection = linear_model.require_fitted(
                self.projection_set, "projection_set"
            )
            projected = projection.project(self.weight)
            if projected is not self.weight:
                self._projections += 1
                self._projection_distance += float((projected - self.weight).norm())
                self.weight.copy_(projected)

    def run_metrics(self) -> dict:
        """Return how often the run projected, and how far in total."""
        return {
            "projections": self._projections - self._run_start[0],
            "projection_distance": self._projection_distance - self._run_start[1],
        }

    def summary(self) -> str:
        """Return a plain-text report on how the fit went.

        Covers the data, the optimization, the estimates against the
        untruncated fit, the projection set and the curvature at the
        averaged estimate. There are no standard errors or p-values.

        Returns:
            The report, ready to print.

        Raises:
            RuntimeError: If called before fit.
        """
        return summary.render(self)
