# delphi

Statistical estimation from **truncated** samples — data where observations
outside some set are not merely unlabelled but absent entirely, so the sample
you hold is a biased view of the distribution that produced it.

Fitting such data with standard estimators gives confidently wrong answers:
ordinary least squares on left-truncated labels recovers a slope that reflects
the truncation, not the model. The algorithms here correct for a truncation set
you can query through a membership oracle.

## Status

This package is being brought to an MVP, and this README is rebuilt alongside
it: each algorithm gets its section as it lands, rather than documenting
intentions. The scope is truncated exponential-family distributions, unknown
truncation for multivariate normals via Hermite polynomials, truncated linear
regression with known and unknown noise variance, truncated lasso, confidence
intervals for truncated linear regression, and truncated classification.

Only what is documented below is supported today.

## Install

```bash
poetry install
```

Development uses [`just`](https://github.com/casey/just):

```bash
just check    # ruff format, ruff check, pylint --errors-only, mypy
just test     # the test suite
just cov      # branch coverage of delphi/
```

## Quickstart

Truncated linear regression with known noise variance. The data below is
left-truncated at zero: every pair whose label fell below the threshold is gone
from the sample, not hidden in it.

```python
import torch as ch
from delphi import oracle
from delphi.stats import TruncatedLinearRegression

ch.manual_seed(0)

# Ground truth: y = x, with unit-variance Gaussian noise.
noise_var = ch.ones(1, 1)
w = ch.ones(1, 1)
x = ch.rand(1000, 1)
y = x @ w.T + ch.sqrt(noise_var) * ch.randn(1000, 1)

# The truncation set, as a membership oracle: labels must be positive.
phi = oracle.Left_Regression(ch.zeros(1))
kept = phi(y).nonzero()[:, 0]
x_trunc, y_trunc = x[kept], y[kept]

# alpha is the survival probability: the fraction that made it through.
alpha = x_trunc.size(0) / x.size(0)

model = TruncatedLinearRegression(
    {"epochs": 5, "optimizer": "lbfgs", "batch_size": -1, "trials": 1},
    phi,
    alpha,
    fit_intercept=False,
    noise_var=noise_var,
)
model.fit(x_trunc, y_trunc)
print(model.coef_)  # 0.983
```

About 70% of the sample survives truncation here, and that is enough to matter:
ordinary least squares on the same truncated data recovers **1.583** against a
true weight of 1.0, because the discarded pairs are disproportionately the
low-`x` ones. The estimator above recovers 0.983 from the identical sample.

The first argument is a plain dict of hyperparameters, validated against a
pydantic config. Keys the config does not declare are ignored with a warning,
so a renamed or misspelled key is audible rather than silent.

## License

MIT.
