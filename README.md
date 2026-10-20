# delphi

Statistical estimation from **truncated** samples: data where every
observation outside some set is missing entirely, so the sample is a biased
view of the distribution that produced it. Given a membership oracle for that
set, delphi recovers the distribution as it was before truncation.

## Table of Contents

- [Install](#install)
- [Truncated Distributions](#truncated-distributions)
  - [Quickstart](#quickstart)
  - [Usage notes](#usage-notes)
- [Development](#development)
- [Citing](#citing)
- [License](#license)

## Install

delphi needs Python 3.10 to 3.14, [Poetry](https://python-poetry.org) and
[`just`](https://github.com/casey/just). Neither needs a version manager such
as asdf.

```bash
# Poetry
curl -sSL https://install.python-poetry.org | python3 -

# just, into ~/.local/bin (or: brew install just, cargo install just)
curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to ~/.local/bin
```

Then clone and install delphi:

```bash
git clone https://github.com/pstefanou12/delphi.ai.git
cd delphi.ai
just install    # poetry install --with dev
just hooks      # pre-push hooks running just check
```

## Truncated Distributions

Each model is fit by projected stochastic gradient descent on the truncated
negative log-likelihood. All live in `delphi.truncated.distributions`.

| Model | Module | Example |
|---|---|---|
| `TruncatedMultivariateNormal` | [`truncated_multivariate_normal`](delphi/truncated/distributions/truncated_multivariate_normal.py) | |
| `TruncatedMultivariateNormalKnownCovariance` | [`truncated_multivariate_normal_known_covariance`](delphi/truncated/distributions/truncated_multivariate_normal_known_covariance.py) | |
| `TruncatedNormal` | [`truncated_normal`](delphi/truncated/distributions/truncated_normal.py) | |
| `TruncatedNormalKnownVariance` | [`truncated_normal_known_variance`](delphi/truncated/distributions/truncated_normal_known_variance.py) | |
| `TruncatedExponential` | [`truncated_exponential`](delphi/truncated/distributions/truncated_exponential.py) | |
| `TruncatedPoisson` | [`truncated_poisson`](delphi/truncated/distributions/truncated_poisson.py) | |
| `TruncatedWeibull` | [`truncated_weibull`](delphi/truncated/distributions/truncated_weibull.py) | |
| `TruncatedBooleanProduct` | [`truncated_boolean_product`](delphi/truncated/distributions/truncated_boolean_product.py) | |
| `TruncatedGamma` | [`truncated_gamma`](delphi/truncated/distributions/truncated_gamma.py) | |
| `TruncatedPareto` | [`truncated_pareto`](delphi/truncated/distributions/truncated_pareto.py) | |
| `TruncatedBeta` | [`truncated_beta`](delphi/truncated/distributions/truncated_beta.py) | |

### Quickstart

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

### Usage notes

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

**`model.summary()` reports how the fit went**: the data and survival mass,
the optimizer, scheduler, steps, stop reason and final losses, all four
estimates, how often the projection fired, the fitted model's own mass on the
truncation set (a mismatch with `alpha` means the fit has gone wrong) and the
curvature at the estimate. It has no standard errors or p-values: neither
analysis this follows gives an asymptotic distribution for the estimate.
Estimates are reported in the coordinates the model was fit in, so after
standardizing, map them back as in the Quickstart.

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
just check                 # ruff format, ruff check, pylint --errors-only, mypy
just test                  # tests affected by your changes; all of them in CI
just test --testmon-noselect   # the full suite
just test -m "not slow"    # skip fits over a minute
just test <path> --log-cli-level=INFO   # show training logs
just cov                   # branch coverage of delphi/
```

## Citing

If you use delphi in your work, please cite it:

```bibtex
@software{stefanou_delphi,
  author  = {Stefanou, Patroklos},
  title   = {delphi: statistical estimation from truncated samples},
  url     = {https://github.com/pstefanou12/delphi.ai},
  version = {0.2.1.9},
  year    = {2026},
}
```

## License

MIT.
