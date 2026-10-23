# Author: pstefanou12@
"""Plain-text report on a fitted truncated linear model.

The report covers how the optimization went, not inference: there are no
standard errors or p-values, since the analysis the estimators follow gives
a rate, not an asymptotic distribution.
"""

from delphi.optim.lr_scheduler import fisher
from delphi.stats import linear_model
from delphi.truncated.distributions import summary as distribution_summary


def render(model) -> str:
    """Return the report for a fitted model.

    Args:
        model: A fitted ``TruncatedLinearModel``.

    Returns:
        The report, ready to print.

    Raises:
        RuntimeError: If the model has not been fit.
    """
    if model.procedure is None:
        raise RuntimeError(f"{type(model).__name__} has not been fit; call fit first.")
    avg = model.procedure.final.avg_params
    return distribution_summary.frame(
        str(model),
        [
            ("Model and data", _data(model, avg)),
            ("Optimization", distribution_summary.optimization(model.procedure)),
            ("Estimates", _estimates(model)),
            (
                "Projection set at the averaged estimate",
                model.projection_set.describe(avg.reshape(-1, model.k)),
            ),
            ("Curvature at the averaged estimate", _curvature(model, avg)),
        ],
    )


def _data(model, avg) -> list[tuple[str, str]]:
    return [
        ("covariates (d)", str(model.d)),
        ("outputs (k)", str(model.k)),
        ("intercept", "fit" if model.fit_intercept else "not fit"),
        ("observed samples", str(model.n_samples_)),
        ("noise", str(model.noise)),
        ("survival mass (alpha)", f"{model.alpha:.4f}"),
        ("mass on the set at estimate", f"{model.mass_on_set(avg):.4f}"),
    ]


def _estimates(model) -> list[tuple[str, str]]:
    rows = []
    for name in linear_model.ESTIMATORS:
        rows.append(
            (
                f"{name}_coef_",
                distribution_summary.tensor(getattr(model, f"{name}_coef_")),
            )
        )
        if model.fit_intercept:
            rows.append(
                (
                    f"{name}_intercept_",
                    distribution_summary.tensor(getattr(model, f"{name}_intercept_")),
                )
            )
    coef, intercept = model.coefficients(model.required_untruncated_weight.flatten())
    rows.append(("untruncated fit coef", distribution_summary.tensor(coef)))
    if intercept is not None:
        rows.append(
            ("untruncated fit intercept", distribution_summary.tensor(intercept))
        )
    return rows


def _curvature(model, avg) -> list[tuple[str, str]]:
    try:
        mu, smoothness = fisher.FisherLR.spectrum(
            model.hessian(avg.reshape(-1, model.k)), model.independent_coordinates
        )
    except ValueError as error:
        return [("curvature", f"skipped: {error}")]
    return [
        ("smallest curvature (mu)", f"{mu:.4e}"),
        ("largest curvature (L)", f"{smoothness:.4e}"),
        ("condition number (L / mu)", f"{smoothness / mu:.3e}"),
    ]
