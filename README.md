# delphi

Statistical estimation from **truncated** samples: data where every
observation outside some set is missing entirely, so the sample is a biased
view of the distribution that produced it. Given a membership oracle for that
set, delphi recovers the distribution as it was before truncation.

## Supported distributions

Each model is fit by projected stochastic gradient descent on the truncated
negative log-likelihood, following Karatapanis, Kontonis and Tzamos, *Oracle
Efficient Truncated Statistics* (ICLR 2025).

| Model | Module | Constructor | Estimate |
|---|---|---|---|
| `TruncatedMultivariateNormal` | `truncated_multivariate_normal` | `(hyperparameters, phi, dims, alpha=None)` | `avg_loc_`, `avg_covariance_matrix_` |
| `TruncatedMultivariateNormalKnownCovariance` | `truncated_multivariate_normal_known_covariance` | `(hyperparameters, phi, dims, covariance_matrix, alpha=None)` | `avg_loc_` |
| `TruncatedNormal` | `truncated_normal` | `(hyperparameters, phi, alpha=None)` | `avg_loc_`, `avg_variance_` |
| `TruncatedNormalKnownVariance` | `truncated_normal_known_variance` | `(hyperparameters, phi, covariance_matrix, alpha=None)` | `avg_loc_` |
| `TruncatedExponential` | `truncated_exponential` | `(hyperparameters, phi, dims, alpha=None)` | `avg_lambda_` |
| `TruncatedPoisson` | `truncated_poisson` | `(hyperparameters, phi, dims, alpha=None)` | `avg_lambda_` |
| `TruncatedWeibull` | `truncated_weibull` | `(hyperparameters, phi, dims, k, alpha=None)` | `avg_lambda_` (scale; shape `k` known) |
| `TruncatedBooleanProduct` | `truncated_boolean_product` | `(hyperparameters, phi, dims, alpha=None)` | `avg_p_` |

All live in `delphi.truncated.distributions`. Each is tested on one-sided,
non-convex, disconnected and seeded random truncation sets, at survival mass
down to about 1%, with no truncation at all, and on pairs of distributions
whose truncated samples are indistinguishable without the oracle.

## Quickstart

A correlated Gaussian, observed only where `x0 + x1 > 1`:

```python
import torch as ch
from torch import distributions

from delphi.truncated.distributions import truncated_multivariate_normal

ch.manual_seed(0)

# Ground truth: a correlated 2-D Gaussian.
loc = ch.tensor([0.0, 0.0])
covariance_matrix = ch.tensor([[1.0, 0.5], [0.5, 1.0]])
population = distributions.MultivariateNormal(loc, covariance_matrix).sample([100000])


# The truncation set: only points with x0 + x1 > 1 are ever observed.
def phi(x):
    return x.sum(dim=-1) > 1.0


S = population[phi(population)][:10000]
alpha = phi(population).float().mean().item()

# Fit on standardized data, moving phi through the same transform.
mean, std = S.mean(0), S.std(0)
model = truncated_multivariate_normal.TruncatedMultivariateNormal(
    {"iterations": 2500, "batch_size": 10, "lr": 1e-2, "scheduler": "fisher"},
    lambda z: phi(z * std + mean),
    dims=2,
    alpha=alpha,
)
model.fit((S - mean) / std)

# Undo the transform on the way out.
recovered_loc = model.avg_loc_ * std + mean
recovered_cov = ch.diag(std) @ model.avg_covariance_matrix_ @ ch.diag(std)
```

| | mean | covariance |
|---|---|---|
| truth | (0, 0) | [[1, 0.5], [0.5, 1]] |
| sample mean and covariance | (1.042, 1.041) | [[0.447, −0.053], [−0.053, 0.436]] |
| recovered | (0.015, 0.051) | [[1.016, 0.488], [0.488, 0.959]] |

28% of the population survives. The sample's own statistics are off in every
entry, and even get the sign of the correlation wrong.

## Usage notes

**The truncation set is a plain callable.** `phi` maps a batch of shape
`(n, dims)` to a mask over it. Anything that answers "is this point observed?"
works; there is nothing to subclass.

**`alpha` is optional.** It is the survival mass, the fraction of the
population inside the set, and it sizes the region the optimizer is kept in.
Omit it if you cannot measure it, and the `min_radius` hyperparameter is used
instead. `alpha=1` is rejected: with nothing truncated there is nothing to
correct, so fit the distribution directly.

**Use `"scheduler": "fisher"`.** It sets the step size from the Hessian of the
truncated likelihood, starting at `1 / L` and decaying as `1 / (L + mu t)`, so
neither needs tuning. In the example above, a constant rate ends at KL 0.025
from the truth after the same 2500 steps, against 0.002.

**Standardize multivariate normals first.** Fit on `(S - mean) / std` with
`phi` composed with the inverse transform, as above, and map the estimates
back. On the example above, fitting the raw data instead ends at KL 0.19
after the same 2500 steps, against 0.002.

**Use the `avg_` estimates.** Every model exposes four: `avg_` (the mean of
the iterates after a burn-in, set by `avg_burn_in`), `ema_` (a bias-corrected
moving average), `final_` (the last iterate) and `best_` (the iterate with the
lowest validation loss). The averaged iterate is the one the convergence
guarantee is about; `best_` is the least reliable of the four in our tests.

**Hyperparameters are a plain dict**, validated against a pydantic config.
Keys the config does not declare are ignored with a warning, so a misspelled
key is audible rather than silent.

**Logging is off until you turn it on.** delphi logs through the standard
library under the `delphi` logger and installs no handler:

```python
import logging

logging.basicConfig(level=logging.INFO)
logging.getLogger("delphi.trainer").setLevel(logging.WARNING)  # per module
```

## Development

```bash
poetry install --with dev
just check                 # ruff format, ruff check, pylint --errors-only, mypy
just test                  # the full suite
just test -m "not slow"    # skip fits over a minute
just test <path> --log-cli-level=INFO   # show training logs
just cov                   # branch coverage of delphi/
```

## License

MIT.
