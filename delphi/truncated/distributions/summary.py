# Author: pstefanou12@
"""Plain-text report on a fitted truncated exponential family distribution.

The report covers how the optimization went, not inference: there are no
standard errors or p-values, since neither analysis the library follows gives
an asymptotic distribution for the estimate.
"""

from delphi.optim.lr_scheduler import fisher

ESTIMATORS = ("avg", "ema", "final", "best")

# Longer vectors are cut to their first few entries.
MAX_ENTRIES = 6

WIDTH = 72


def render(model) -> str:
    """Return the report for a fitted model.

    Args:
        model: A fitted ``TruncatedExponentialFamilyDistribution``.

    Returns:
        The report, ready to print.

    Raises:
        RuntimeError: If the model has not been fit.
    """
    if model.trainer is None:
        raise RuntimeError(f"{type(model).__name__} has not been fit; call fit first.")
    mass = model.mass_on_set(model.avg_params)
    sections = [
        ("Model and data", _data(model)),
        ("Optimization", _optimization(model)),
        ("Estimates", _estimates(model)),
        ("Truncation and curvature at the averaged estimate", _curvature(model, mass)),
    ]
    lines = ["=" * WIDTH, f" {model}", "=" * WIDTH]
    for title, rows in sections:
        lines += ["", f" {title}", " " + "-" * (WIDTH - 1)]
        lines += [f"   {label:<30}{value}" for label, value in rows]
    lines.append("=" * WIDTH)
    return "\n".join(lines)


def _data(model) -> list[tuple[str, str]]:
    alpha = (
        "not given; radius from min_radius"
        if model.alpha is None
        else f"{model.alpha:.4f}"
    )
    return [
        ("dimensions", str(model.dims)),
        ("observed samples", str(model.n_samples_)),
        ("survival mass (alpha)", alpha),
        ("final radius", f"{model.radius:.4f}"),
    ]


def _optimization(model) -> list[tuple[str, str]]:
    hyperparameters, trainer = model.hyperparameters, model.trainer
    grad_norms = trainer.grad_norms
    rows = [
        ("optimizer", f"{hyperparameters.optimizer}, lr {hyperparameters.lr:g}"),
        ("scheduler", hyperparameters.scheduler or "constant"),
        ("steps", f"{model.steps_} over {len(model.radius_history)} phase(s)"),
        (
            "stop reason",
            f"{trainer.procedure_stop_reason}; last phase {trainer.stop_reason}",
        ),
        ("wall time", f"{model.fit_seconds_:.2f}s"),
        ("final train loss", _scalar(trainer.train_losses[-1])),
        ("final validation loss", _scalar(trainer.final_loss)),
    ]
    warm = trainer.warm_start_result
    if warm is not None:
        rows.insert(
            3, ("warm start", f"{warm.iterations} steps, stopped on {warm.stop_reason}")
        )
    if len(grad_norms):
        rows.append(("final gradient norm", f"{float(grad_norms[-1]):.4e}"))
    if hyperparameters.project:
        mean = model.projection_distance_ / max(model.projections_, 1)
        share = 100.0 * model.projections_ / max(model.steps_, 1)
        rows.append(
            (
                "projections",
                f"{model.projections_} ({share:.1f}% of steps), mean move {mean:.4f}",
            )
        )
    else:
        rows.append(("projections", "disabled"))
    return rows


def _estimates(model) -> list[tuple[str, str]]:
    rows = []
    for suffix in model.estimate_suffixes:
        for name in ESTIMATORS:
            rows.append(
                (f"{name}_{suffix}", _tensor(getattr(model, f"{name}_{suffix}")))
            )
    return rows


def _curvature(model, mass: float) -> list[tuple[str, str]]:
    rows = [("mass on the truncation set", f"{mass:.4f}")]
    if model.alpha is not None:
        rows.append(("  against alpha", f"{model.alpha:.4f}"))
    try:
        mu, smoothness = fisher.FisherLR.spectrum(
            model.hessian(model.avg_params), model.independent_coordinates
        )
    except ValueError as error:
        rows.append(("curvature", f"skipped: {error}"))
        return rows
    rows += [
        ("smallest curvature (mu)", f"{mu:.4e}"),
        ("largest curvature (L)", f"{smoothness:.4e}"),
        ("condition number (L / mu)", f"{smoothness / mu:.3e}" if mu > 0 else "inf"),
    ]
    return rows


def _scalar(value) -> str:
    return "n/a" if value is None else f"{float(value):.6f}"


def _tensor(value) -> str:
    if value is None:
        return "n/a"
    value = value.detach()
    if value.dim() == 2 and value.size(0) == value.size(1) and value.size(0) <= 4:
        rows = (", ".join(f"{float(x):.4f}" for x in row) for row in value)
        return "[" + "; ".join(rows) + "]"
    flat = value.flatten()
    shown = ", ".join(f"{float(x):.4f}" for x in flat[:MAX_ENTRIES])
    more = f", ... ({flat.numel()} total)" if flat.numel() > MAX_ENTRIES else ""
    return f"({shown}{more})"
