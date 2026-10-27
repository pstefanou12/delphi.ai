# Author: pstefanou12@
"""Abstract truncated linear model.

The model is ``z = h_W(x) + e``, observed only where the oracle ``phi``
admits it. A subclass names its parameters, its criterion and the convex set
the descent is kept in; projected SGD through the trainer, the radius phases
and the report live here.
"""

import abc
from collections.abc import Callable
import math
from typing import Generic, Optional, TypeVar

import torch as ch
import torch.nn
import torch.utils.data

from delphi import configs
import delphi.projection_sets
from delphi.stats import linear_model
import delphi.trainer
from delphi.truncated.distributions import summary
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

    Subclasses supply ``criterion_parameters`` and, to constrain the descent
    or refine it, ``build_projection_set`` and ``hessian``; ``refine_steps``
    adds ``StatisticalModel.refine`` after the descent.

    The descent starts from ``emp_weight``: the caller's, else OLS. With
    alpha known the projection radius is fixed and one phase runs. Without
    it the radius starts at the ``radius`` hyperparameter, or where alpha = 1
    would put it, and grows by ``rate`` each phase, until a phase never
    projects or it reaches ``max_radius``.

    Attributes:
        phi: Truncation oracle over latents shaped (n, m, k), returning a
            mask or a survival probability.
        alpha: Survival probability lower bound, or None if unknown.
        rand_seed: Seed applied at the start of every run.
        projection_set: The set the iterates are kept in, built on the
            first phase.
        radius: The projection radius of the current phase.
    """

    # The covariates are correlated, so the Hessian's off-diagonal entries
    # carry curvature and FisherLR reads its eigenvalues.
    independent_coordinates: bool = False

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        hyperparameters: dict,
        phi: Callable,
        alpha: Optional[float] = None,
        fit_intercept: bool = True,
        emp_weight: Optional[ch.Tensor] = None,
        rand_seed: int = 0,
    ):
        """Initialize the truncated linear model.

        Args:
            hyperparameters: Hyperparameter dict.
            phi: Truncation oracle over latents shaped (n, m, k).
            alpha: Survival probability in (0, 1], or None if unknown.
            fit_intercept: Whether to fit an intercept.
            emp_weight: Start for the descent in place of OLS, shaped
                (d + 1, k) with the intercept as its last row, in the data's
                units.
            rand_seed: Seed applied at the start of every run.

        Raises:
            ValueError: If alpha lies outside (0, 1].
        """
        if alpha is not None and not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must lie in (0, 1]; got {alpha}.")
        super().__init__(
            hyperparameters, emp_weight=emp_weight, fit_intercept=fit_intercept
        )
        self.phi = phi
        self.alpha = alpha
        self.rand_seed = rand_seed
        self.projection_set: Optional[delphi.projection_sets.ProjectionSet] = None
        self.radius: Optional[float] = None
        self.trainer: Optional[delphi.trainer.Trainer] = None
        self._data: Optional[tuple[ch.Tensor, ch.Tensor]] = None
        self._projections = 0

    @abc.abstractmethod
    def criterion_parameters(self, X: ch.Tensor, y: ch.Tensor) -> list:  # pylint: disable=invalid-name
        """Return the criterion's arguments after the prediction and target.

        Called once the parameters exist, with the training data.
        """

    def _calc_emp_model(self, X: ch.Tensor, y: ch.Tensor) -> None:  # pylint: disable=invalid-name
        """Set emp_weight to OLS unless given, and register the weight from it.

        Args:
            X: Design matrix in the descent's coordinates, shaped (n, d + 1)
                with an intercept.
            y: Targets, shaped (n, k).
        """
        if self.emp_weight is None:
            ols = ch.linalg.lstsq(X, y).solution  # pylint: disable=not-callable
            self.emp_weight = ols / self.covariate_scale
        self.register_weight(self.emp_start)

    @property
    def emp_start(self) -> ch.Tensor:
        """emp_weight in the descent's coordinates, shaped (d + 1, k)."""
        k = linear_model.require_fitted(self.k, "k")
        start = self.required_emp_weight.reshape(-1, k)
        if self.fit_intercept and start.size(0) == self.d:
            start = ch.cat([start, ch.zeros(1, k)])
        return start * self.covariate_scale

    def register_weight(self, weight: ch.Tensor) -> None:
        """Register the trained parameters from a start weight.

        Args:
            weight: The start, shaped (d + 1, k), in the descent's
                coordinates.
        """
        self.register_parameter("weight", torch.nn.Parameter(weight.clone()))

    def build_projection_set(
        self,
        X: ch.Tensor,  # pylint: disable=invalid-name
        y: ch.Tensor,
        radius: float,
    ) -> delphi.projection_sets.ProjectionSet:
        """Return the convex set the descent is kept in.

        Args:
            X: Design matrix, shaped (n, d + 1) with an intercept.
            y: Targets, shaped (n, k).
            radius: The set's radius.

        Returns:
            The whole space, unless a subclass constrains it.
        """
        del X, y
        return delphi.projection_sets.Unconstrained(self.emp_start, radius)

    def default_radius(self) -> float:
        """Return the first phase's radius absent the radius hyperparameter.

        Returns:
            D_{r,B}'s ``4 log(2 / alpha) + 7``, taking alpha = 1 when it is
            unknown.
        """
        alpha = 1.0 if self.alpha is None else self.alpha
        return 4.0 * math.log(2.0 / alpha) + 7.0

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
        self.covariate_scale = 1.0
        X = self.design_matrix(X)  # pylint: disable=invalid-name
        if self.hyperparameters.scale_covariates:
            # Daskalakis et al. (2020), (4.1): ||x||_2 <= 1 once divided by
            # B sqrt(d), with B the largest entry, the intercept's 1 included.
            self.covariate_scale = float(X.abs().max()) * math.sqrt(X.size(1))
            X = X / self.covariate_scale  # pylint: disable=invalid-name
        self._data = (X, y)
        self._calc_emp_model(X, y)
        self.criterion_params = self.criterion_parameters(X, y)
        self.projection_set, self.radius = None, None
        self.trainer = delphi.trainer.Trainer(self, self.hyperparameters)
        train_loader, val_loader = self._loaders(X, y, self.hyperparameters)
        self.procedure = self.trainer.train_model(
            train_loader, val_loader, rand_seed=self.rand_seed
        )
        self.refine(train_loader, rand_seed=self.rand_seed)
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

    def _loaders(self, X: ch.Tensor, y: ch.Tensor, config):  # pylint: disable=invalid-name
        """Return the training loader, and a validation loader if val > 0.

        Without a validation split, every sample feeds the descent and
        ``evaluate`` scores the iterates on the training set instead.
        """
        if config.val > 0.0:
            return datasets.make_train_and_val(config, X, y)
        batch_size = X.size(0) if config.batch_size == -1 else config.batch_size
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X, y),
            batch_size=batch_size,
            shuffle=config.shuffle,
        )
        return train_loader, None

    def pre_phase_hook(self) -> None:
        """Build the projection set on the first phase, resize it after."""
        self._projections = 0
        if self.projection_set is not None:
            self.projection_set.radius = self.radius
            return
        hp = self.hyperparameters
        radius = self.default_radius() if hp.radius is None else hp.radius
        if hp.projection_set == "default":
            X, y = linear_model.require_fitted(self._data, "data")  # pylint: disable=invalid-name
            self.projection_set = self.build_projection_set(X, y, radius)
        else:
            self.projection_set = delphi.projection_sets.Unconstrained(
                self.emp_start, radius
            )
        self.radius = self.projection_set.radius
        if "weight" in self._parameters:
            with ch.no_grad():
                self.weight.copy_(self.projection_set.project(self.weight))

    def step_post_hook(self, optimizer, args, kwargs) -> None:
        """Project the weight back onto the projection set."""
        projection_set = linear_model.require_fitted(
            self.projection_set, "projection_set"
        )
        with ch.no_grad():
            projected = projection_set.project(self.weight)
            if projected is not self.weight:
                self._projections += 1
                self.weight.copy_(projected)

    def post_phase_hook(
        self, result: delphi.trainer.RunResult
    ) -> tuple[bool, Optional[str]]:
        """Grow the projection radius when alpha is unknown, or stop."""
        if self.alpha is not None:
            return True, "alpha_known"
        if result.metrics["projections"] == 0:
            return True, "radius_not_binding"
        radius = linear_model.require_fitted(self.radius, "radius")
        if radius >= self.hyperparameters.max_radius:
            return True, "max_radius"
        self.radius = min(
            radius * self.hyperparameters.rate, self.hyperparameters.max_radius
        )
        return False, None

    def run_metrics(self) -> dict:
        """Return the run's projection radius and how often it projected."""
        return {"radius": self.radius, "projections": self._projections}

    def evaluate(self) -> dict:
        """Return the criterion at the current iterate on the training set.

        Every call draws the same noise, so ``best_`` compares iterates rather
        than Monte Carlo draws, and evaluation leaves the training RNG alone.
        """
        X, y = linear_model.require_fitted(self._data, "data")  # pylint: disable=invalid-name
        with ch.no_grad(), ch.random.fork_rng():
            ch.manual_seed(self.rand_seed)
            return {"loss": self.compute_loss((X, y)).sum()}

    @property
    def emp_coef_(self) -> Optional[ch.Tensor]:
        """Coefficients of emp_weight, the descent's start."""
        return linear_model.LinearModel.coefficients(self, self.emp_start.flatten())[0]

    @property
    def emp_intercept_(self) -> Optional[ch.Tensor]:
        """Intercept of emp_weight, the descent's start."""
        return linear_model.LinearModel.coefficients(self, self.emp_start.flatten())[1]

    @property
    def theta(self) -> ch.Tensor:
        """The trained weight, which FisherLR re-measures curvature at."""
        return self.weight

    def summary(self) -> str:
        """Return a plain-text report on the fit.

        Returns:
            The report, ready to print.

        Raises:
            RuntimeError: If called before fit.
        """
        if self.procedure is None:
            raise RuntimeError(
                f"{type(self).__name__} has not been fit; call fit first."
            )
        return summary.frame(
            str(self),
            [
                ("Data", self._summary_data()),
                ("Procedure", self._summary_procedure()),
                ("Estimates", self._summary_estimates()),
            ],
        )

    def _summary_data(self) -> list[tuple[str, str]]:
        X, _ = linear_model.require_fitted(self._data, "data")  # pylint: disable=invalid-name
        alpha = "unknown" if self.alpha is None else f"{self.alpha:.4f}"
        return [
            ("samples", str(X.size(0))),
            ("covariates", str(self.d)),
            ("outputs", str(self.k)),
            ("survival mass (alpha)", alpha),
        ]

    def _summary_procedure(self) -> list[tuple[str, str]]:
        procedure = linear_model.require_fitted(self.procedure, "procedure")
        rows = [
            (
                "descent",
                f"{procedure.iterations} steps over {len(procedure.phases)} "
                f"phase(s), {procedure.seconds:.2f}s; {procedure.stop_reason}",
            )
        ]
        if self.refinement is not None:
            rows.append(
                (
                    "refinement",
                    f"{self.refinement.iterations} Newton steps, "
                    f"{self.refinement.seconds:.2f}s",
                )
            )
        radius = "none" if self.radius is None else f"{self.radius:.4f}"
        rows.append(("projection radius", radius))
        rows.append(("projections", str(sum(procedure.series("projections")))))
        return rows

    def _summary_estimates(self) -> list[tuple[str, str]]:
        rows = []
        for name in linear_model.ESTIMATORS:
            rows.append(
                (f"{name}_coef_", summary.tensor(getattr(self, f"{name}_coef_")))
            )
            if self.fit_intercept:
                rows.append(
                    (
                        f"{name}_intercept_",
                        summary.tensor(getattr(self, f"{name}_intercept_")),
                    )
                )
        rows.append(("emp_coef_", summary.tensor(self.emp_coef_)))
        if self.fit_intercept:
            rows.append(("emp_intercept_", summary.tensor(self.emp_intercept_)))
        return rows
