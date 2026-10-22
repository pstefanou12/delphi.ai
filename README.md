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
  - [Unknown truncation](#unknown-truncation)
- [Architecture](#architecture)
  - [The trainer](#the-trainer)
  - [Models](#models)
  - [Polynomials](#polynomials)
  - [Adding a distribution](#adding-a-distribution)
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
| `TruncatedLogNormal` | [`truncated_log_normal`](delphi/truncated/distributions/truncated_log_normal.py) | |
| `TruncatedBinomial` | [`truncated_binomial`](delphi/truncated/distributions/truncated_binomial.py) | |
| `TruncatedGeometric` | [`truncated_geometric`](delphi/truncated/distributions/truncated_geometric.py) | |
| `TruncatedNegativeBinomial` | [`truncated_negative_binomial`](delphi/truncated/distributions/truncated_negative_binomial.py) | |
| `TruncatedLaplace` | [`truncated_laplace`](delphi/truncated/distributions/truncated_laplace.py) | |
| `TruncatedChi2` | [`truncated_chi2`](delphi/truncated/distributions/truncated_chi2.py) | |
| `TruncatedContinuousBernoulli` | [`truncated_continuous_bernoulli`](delphi/truncated/distributions/truncated_continuous_bernoulli.py) | |
| `TruncatedVonMises` | [`truncated_von_mises`](delphi/truncated/distributions/truncated_von_mises.py) | |
| `TruncatedDirichlet` | [`truncated_dirichlet`](delphi/truncated/distributions/truncated_dirichlet.py) | |
| `TruncatedHalfNormal` | [`truncated_half_normal`](delphi/truncated/distributions/truncated_half_normal.py) | |
| `TruncatedInverseGamma` | [`truncated_inverse_gamma`](delphi/truncated/distributions/truncated_inverse_gamma.py) | |
| `TruncatedMultinomial` | [`truncated_multinomial`](delphi/truncated/distributions/truncated_multinomial.py) | |

Without a membership oracle, see [Unknown truncation](#unknown-truncation):

| Model | Module | Example |
|---|---|---|
| `UnknownTruncationMultivariateNormal` | [`unknown_truncated_multivariate_normal`](delphi/truncated/distributions/unknown_truncated_multivariate_normal.py) | |
| `UnknownTruncationMultivariateNormalKnownCovariance` | [`unknown_truncated_multivariate_normal`](delphi/truncated/distributions/unknown_truncated_multivariate_normal.py) | |
| `UnknownTruncationNormal` | [`unknown_truncated_normal`](delphi/truncated/distributions/unknown_truncated_normal.py) | |
| `UnknownTruncationNormalKnownVariance` | [`unknown_truncated_normal`](delphi/truncated/distributions/unknown_truncated_normal.py) | |

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
    {},  # the defaults: 2500 steps, lr 1e-2, the Fisher schedule
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
| recovered | (0.015, 0.047) | [[1.016, 0.491], [0.491, 0.963]] |

28% of the population survives. The sample's own statistics are off in every
entry, and even get the sign of the correlation wrong.

### Usage notes

**The truncation set is a plain callable.** `phi` maps a batch of shape
`(n, dims)` to a mask over it. Anything that answers "is this point observed?"
works; there is nothing to subclass.

**`alpha` is optional.** It is the survival mass, the fraction of the
population inside the set, and it sizes the region the optimizer is kept in.
Omit it if you cannot measure it: the budget then starts at `min_radius` and
grows by a factor of `rate` per phase, while the projection still binds, up to
`max_radius`. `alpha=1` is rejected: with nothing truncated there is nothing
to correct, so fit the distribution directly.

**The Fisher schedule is the default.** It sets the step size from the
Hessian of the truncated likelihood, starting at `1 / L` and decaying as
`1 / (L + mu t)`, and re-measures both as the fit moves, so neither needs
tuning. In the example above, a constant rate (`"scheduler": None`) ends at KL
0.026 from the truth after the same 2500 steps, against 0.002.

**Standardize multivariate normals first.** Fit on `(S - mean) / std` with
`phi` composed with the inverse transform, as above, and map the estimates
back. On the example above, fitting the raw data instead ends at KL 0.27
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

### Unknown truncation

When the set cannot be queried at all, a normal can still be recovered if the
set is not too complex, by the algorithm of Kontonis, Tzamos and Zampetakis
(2019), "Efficient Truncated Statistics with Unknown Truncation". It whitens
the sample, estimates the degree-`degree` Hermite expansion of a function that
mixes the set with the Gaussian, and minimizes a convex objective built from it
by projected L-BFGS over the whole sample (the paper uses SGD, whose
heavy-tailed minibatch gradients stalled above the minimum in ten dimensions).
These models take no `phi`:

```python
import torch as ch
from torch import distributions

from delphi.truncated.distributions import unknown_truncated_multivariate_normal

ch.manual_seed(0)

# Observed only where x0 > -0.5 and x1 > -0.5; the fit is never told so.
population = distributions.MultivariateNormal(ch.zeros(2), ch.eye(2)).sample([50000])
inside = (population[:, 0] > -0.5) & (population[:, 1] > -0.5)
S = population[inside][:20000]

model = unknown_truncated_multivariate_normal.UnknownTruncationMultivariateNormal(
    {}, dims=2  # the defaults: degree 124, the largest within 8000 terms
)
model.fit(S)

# The set the fit implies, as a membership oracle.
recovered = model.truncation_set()
```

| | mean | covariance | KL from truth |
|---|---|---|---|
| truth | (0, 0) | [[1, 0], [0, 1]] | 0 |
| sample mean and covariance | (0.506, 0.512) | [[0.490, 0.004], [0.004, 0.477]] | 0.469 |
| recovered | (−0.010, 0.070) | [[1.019, 0.037], [0.037, 0.926]] | 0.005 |

`recovered` disagrees with the true set on 3% of the population.

**The degree sets the accuracy.** The expansion of a set's indicator converges
slowly, and at a fixed degree the estimate keeps a bias that more samples do
not remove; the paper's bound shrinks as `degree ** (-1 / 4)`. A degree-k basis
in d dimensions has `C(k + d, d)` coefficients, so by default the degree is the
largest whose basis has at most `max_terms` (8000) terms, capped at
`max_degree` (200): 200 in one dimension, 124 in two, 34 in three, 5 in ten.
Low dimensions gain the most. On 20000 samples, a half-space through the mode
went from KL 0.087 at degree 10 to 0.009 at degree 100, after which the
coefficients' sampling noise overtook the shrinking bias. In ten dimensions the
basis outgrows the samples before the bias falls far.

**Know the covariance if you can.** With it, the sample is whitened by the
truth and only the mean is learned; the test suite's fits then land at KL
0.001 to 0.008 in two and three dimensions (0.025 on a ten-dimensional ball,
at degree 5), where the unknown-covariance ones land at 0.005 to 0.012. Both
degrade as the survival mass falls, which `alpha` does not change: here it
only sizes the region the descent is kept in.

## Architecture

delphi separates *how a model is fit*, which the trainer owns, from *what is
fit*, which each model describes.

### The trainer

`delphi/trainer.py` fits any model built on the `delphi` base class
(`delphi/delphi.py`). Its one entry point, `Trainer.train_model`, runs a
**procedure**:

1. **An optional warm start.** If the config sets a budget,
   `warm_start_iterations` or `warm_start_epochs`, the trainer first trains
   the model's `warm_start_criterion`. `warm_start_`-prefixed settings
   override the main run's stop criteria for this stage only. Its result is
   handed to the model's `post_warm_start_hook`.
2. **Phases.** Each phase trains the model's own `criterion` from the current
   parameters, with a fresh optimizer and scheduler. After each one, the
   model's `post_phase_hook` either stops the procedure, giving a reason, or
   changes the problem for the next phase, such as widening a projection
   budget. `max_phases` caps the number. A model that never changes its
   problem keeps the default hook and trains once.

Each run, warm start or phase, is one **stage**: a config, a criterion and
that criterion's Hessian, which the Fisher learning-rate schedule sizes its
steps from. `train_model` returns a `ProcedureResult` with the warm start,
the phases in order and why the procedure stopped. Each run is a `RunResult`
holding the config it ran under, its loss, gradient and iterate histories,
its stop reason and wall time, and any `metrics` the model reports through
`run_metrics`. The best, final, moving-average and averaged estimates are
computed from the iterates.

Within a run the trainer steps the optimizer and records every iterate. It
stops on the first of the iteration or epoch budget, the gradient or loss
tolerances, patience, or the model's own `should_stop`.

### Models

A model subclasses `delphi` and supplies:

| piece | role |
|---|---|
| `defaults` | its pydantic config class |
| `criterion`, `criterion_params` | the loss the phases train |
| `hessian()` | the criterion's curvature, needed by the Fisher schedule |
| `warm_start_criterion`, `warm_start_hessian` | the warm start's loss, its parameters bound, and its curvature |
| `pre_phase_hook`, `post_phase_hook` | prepare each phase; decide after it whether another runs |
| `post_warm_start_hook`, `post_training_hook` | adopt the warm start's result; finish after the whole procedure |
| `step_post_hook`, `run_metrics` | act after each step (e.g. project); report facts about a run |

The truncated distributions add two layers on top of this:

- **`delphi/exponential_family/<family>.py`** wraps the torch distribution in
  natural parameters: its sufficient statistic `T(x)`, `to_natural`,
  `to_canonical` and `log_prob`.
- **`delphi/truncated/distributions/truncated_<family>.py`** is the model. It
  subclasses `TruncatedExponentialFamilyDistribution`, which supplies the
  truncated NLL, its Hessian, the sublevel-set projection and the phases
  that widen its budget. The family adds its starting point
  (`_calc_emp_model`), the domain of its parameters (`_constraints`) and its
  estimates, named as torch names them (`avg_rate_`, `avg_loc_`, ...).

Truncation sets are plain callables. `delphi/oracle.py` holds the stateful
ones: `Identity`, which admits every sample, and `RecoveredSet`, the set an
unknown-truncation fit implies. The Hermite polynomials that fit uses live
in `delphi/torchnomials/`.

### Polynomials

`delphi/torchnomials/` holds polynomial families in torch, built on an
abstract `Polynomial` over total-degree multi-indices. `Hermite` gives the
physicist (default) or probabilist (`prob=True`) Hermite polynomials,
normalized under their weight measure with `norm=True`. Fitting to samples
expands their density ratio to that weight:

```python
from delphi.torchnomials import hermite

poly = hermite.Hermite(degree=10, dims=2, prob=True, norm=True)
poly.fit(samples)       # c_V = mean(P_V) / ||P_V||^2
values = poly(points)   # sum_V c_V P_V(points)
```

### Adding a distribution

1. Write the family class in `delphi/exponential_family/`, and check
   `<T(x), theta> + log h(x) - log_prob(x)` is constant across samples.
2. Write the truncated model. If the untruncated MLE has a closed form,
   `_calc_emp_model` returns it. If not, it returns a rough start and the
   family's config enables the warm start
   (`WarmStartedExponentialFamilyDistributionConfig`), which refines that
   start to the untruncated MLE.
3. Test it with `helpers.assert_family_recovers` and
   `helpers.assert_family_distinguishes`, as the existing
   `truncated_*_test.py` modules do.

## Development

```bash
just check                 # ruff format, ruff check, pylint --errors-only, mypy
just test                  # tests affected by your changes; all of them in CI
just test --testmon-noselect   # the full suite
just test -m "not slow"    # skip fits over a minute
just test <path> -n 0 --log-cli-level=INFO   # show training logs, serially
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
