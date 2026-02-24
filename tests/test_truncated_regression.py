"""
Test suite for truncated linear regression.
Includes:
    -Truncated regression with known variance
    -Truncated regression with unknown variance
    -Truncated regression with known regression and temporal dependencies
    -Truncated LASSO regression with known noise variance
"""

import numpy as np
import torch as ch
from torch.nn import MSELoss
from sklearn.linear_model import LinearRegression, LassoCV

from delphi import stats
from delphi import oracle
from delphi.utils.helpers import Parameters, calc_spectral_norm

# CONSTANTS
mse_loss = MSELoss()
seed = 69


# left truncated linear regression with known variance - 1 dimension
def test_known_truncated_regression_one_dimension_no_intercept():
    NUM_SAMPLES = 1000
    # generate ground truth
    NOISE_VAR = ch.ones(1, 1)
    W = ch.ones(1, 1)
    print(f"gt weight: {W}")
    print(f"gt noise var: {NOISE_VAR}")

    # generate data
    X = ch.rand(NUM_SAMPLES, 1)
    y = X @ W.T
    noised = y + ch.sqrt(NOISE_VAR) * ch.randn(y.size(0), 1)
    # generate ground-truth data
    phi = oracle.Left_Regression(ch.zeros(1))
    # truncate
    indices = phi(noised).nonzero()[:, 0]
    x_trunc, y_trunc = X[indices], noised[indices]
    alpha = x_trunc.size(0) / X.size(0)
    print(f"alpha: {alpha}")

    gt_norm = LinearRegression(fit_intercept=False)
    gt_norm.fit(X, y)
    gt_ = ch.from_numpy(np.concatenate([gt_norm.coef_.flatten()]))

    # calculate empirical noise variance for regression
    ols_trunc = LinearRegression(fit_intercept=False)
    ols_trunc.fit(x_trunc, y_trunc)
    emp_ = ch.from_numpy(np.concatenate([ols_trunc.coef_.flatten()]))
    print(f"empirical weights: {emp_}")
    emp_mse_loss = mse_loss(emp_, gt_)
    print(f"emp mse loss: {emp_mse_loss}")

    # scale y features
    y_trunc_scale = y_trunc / ch.sqrt(NOISE_VAR)
    phi_scale = oracle.Left_Regression(phi.left / ch.sqrt(NOISE_VAR))
    # train algorithm
    train_kwargs = Parameters(
        {
            "epochs": 5,
            "optimizer": "lbfgs",
            "batch_size": -1,
            "trials": 1,
            "verbose": True,
        }
    )
    trunc_reg = stats.TruncatedLinearRegression(
        train_kwargs, phi_scale, alpha, fit_intercept=False, noise_var=ch.ones(1, 1)
    )
    trunc_reg.fit(x_trunc, y_trunc_scale)
    w_ = ch.cat([(trunc_reg.best_coef_).flatten()]) * ch.sqrt(NOISE_VAR)
    print(f"estimated weights: {w_}")
    trunc_mse_loss = mse_loss(gt_, w_.flatten())
    print(f"truc mse loss: {trunc_mse_loss}")
    msg = f"trunc mse loss is larger than empirical mse loss. known mse loss is {trunc_mse_loss}, and empirical mse loss is: {emp_mse_loss}"
    assert trunc_mse_loss <= emp_mse_loss, msg
    msg = f"trunc mse loss: {trunc_mse_loss}, which is larger than 1e-1"
    assert trunc_mse_loss <= 1e-1, msg


# left truncated linear regression with known variance - 1 dimension
def test_known_truncated_regression_one_dimension():
    NUM_SAMPLES = 5000
    # generate ground truth
    NOISE_VAR = ch.ones(1, 1)
    W = ch.ones(1, 1)
    W0 = ch.ones(1, 1)

    print(f"gt weight: {W}")
    print(f"gt intercept: {W0}")
    print(f"gt noise var: {NOISE_VAR}")
    # generate data
    X = ch.rand(NUM_SAMPLES, 1)
    y = X @ W.T + W0
    noised = y + ch.sqrt(NOISE_VAR) * ch.randn(y.size(0), 1)
    # generate ground-truth data
    phi = oracle.Left_Regression(ch.ones(1))
    # truncate
    indices = phi(noised).nonzero()[:, 0]
    x_trunc, y_trunc = X[indices], noised[indices]
    alpha = x_trunc.size(0) / X.size(0)
    print(f"alpha: {alpha}")

    gt_norm = LinearRegression()
    gt_norm.fit(X, y)
    gt_ = ch.from_numpy(np.concatenate([gt_norm.coef_.flatten(), gt_norm.intercept_]))

    # calculate empirical noise variance for regression
    ols_trunc = LinearRegression()
    ols_trunc.fit(x_trunc, y_trunc)
    emp_ = ch.from_numpy(
        np.concatenate([ols_trunc.coef_.flatten(), ols_trunc.intercept_])
    )
    print(f"empirical weights: {emp_}")
    emp_mse_loss = mse_loss(emp_, gt_)
    print(f"emp mse loss: {emp_mse_loss}")

    # scale y features
    y_trunc_scale = y_trunc / ch.sqrt(NOISE_VAR)
    phi_scale = oracle.Left_Regression(phi.left / ch.sqrt(NOISE_VAR))
    # train algorithm
    train_kwargs = Parameters(
        {"epochs": 3, "lr": 1e-1, "batch_size": 10, "trials": 1, "verbose": True}
    )
    trunc_reg = stats.TruncatedLinearRegression(
        train_kwargs, phi_scale, alpha, fit_intercept=True, noise_var=ch.ones(1, 1)
    )
    trunc_reg.fit(x_trunc, y_trunc_scale)
    w_ = ch.cat(
        [(trunc_reg.best_coef_).flatten(), trunc_reg.best_intercept_]
    ) * ch.sqrt(NOISE_VAR)
    print(f"estimated weights: {w_}")
    trunc_mse_loss = mse_loss(gt_, w_.flatten())
    print(f"truc mse loss: {trunc_mse_loss}")
    msg = f"trunc mse loss is larger than empirical mse loss. known mse loss is {trunc_mse_loss}, and empirical mse loss is: {emp_mse_loss}"
    assert trunc_mse_loss <= emp_mse_loss, msg
    msg = f"trunc mse loss: {trunc_mse_loss}, which is larger than 1e-1"
    assert trunc_mse_loss <= 1e-1, msg


# left truncated linear regression with known variance - 20 dimensions
def test_known_truncated_regression_higher_dimensions():
    D = 20
    NUM_SAMPLES = 10000
    # generate ground truth
    NOISE_VAR = 3 * ch.ones(1, 1)
    W = ch.ones(1, D)
    W0 = ch.ones(1, 1)
    print(f"gt weight: {W}")
    print(f"gt intercept: {W0}")
    print(f"gt noise var: {NOISE_VAR}")

    # generate data
    X = ch.rand(NUM_SAMPLES, D)
    y = X @ W.T + W0
    noised = y + ch.sqrt(NOISE_VAR) * ch.randn(y.size(0), 1)
    # generate ground-truth data
    phi = oracle.Left_Regression(10 * ch.ones(1))
    # truncate
    indices = phi(noised).nonzero()[:, 0]
    x_trunc, y_trunc = X[indices], noised[indices]
    alpha = x_trunc.size(0) / X.size(0)
    print(f"alpha: {alpha}")

    gt_norm = LinearRegression()
    gt_norm.fit(X, y)
    gt_ = ch.from_numpy(np.concatenate([gt_norm.coef_.flatten(), gt_norm.intercept_]))

    # calculate empirical noise variance for regression
    ols_trunc = LinearRegression()
    ols_trunc.fit(x_trunc, y_trunc)
    emp_ = ch.from_numpy(
        np.concatenate([ols_trunc.coef_.flatten(), ols_trunc.intercept_])
    )
    print(f"empirical weights: {emp_}")
    emp_mse_loss = mse_loss(emp_, gt_)
    print(f"emp mse loss: {emp_mse_loss}")

    # scale y features
    y_trunc_scale = y_trunc / ch.sqrt(NOISE_VAR)
    phi_scale = oracle.Left_Regression(phi.left / ch.sqrt(NOISE_VAR))
    # train algorithm
    train_kwargs = Parameters(
        {
            "epochs": 5,
            "optimizer": "lbfgs",
            "batch_size": -1,
            "trials": 1,
            "verbose": True,
            "num_samples": 5000,
        }
    )
    trunc_reg = stats.TruncatedLinearRegression(
        train_kwargs, phi_scale, alpha, noise_var=ch.ones(1, 1)
    )
    trunc_reg.fit(x_trunc, y_trunc_scale)
    w_ = ch.cat(
        [(trunc_reg.best_coef_).flatten(), trunc_reg.best_intercept_[..., None]]
    ) * ch.sqrt(NOISE_VAR)
    print(f"estimated weights: {w_}")
    known_mse_loss = mse_loss(gt_, w_.flatten())
    print(f"known mse loss: {known_mse_loss}")
    msg = f"known mse loss is larger than empirical mse loss. known mse loss is {known_mse_loss}, and empirical mse loss is: {emp_mse_loss}"
    assert known_mse_loss <= emp_mse_loss, msg
    msg = f"known mse loss is larger than 1e-1. known mse loss is {known_mse_loss.item():.3f}"
    assert known_mse_loss <= 1e-1, msg


# left truncated regression with unknown noise variance in one dimension - no intercept
def test_unknown_variance_truncated_regression_one_dimension_no_intercept():
    D = 1
    NUM_SAMPLES = 10000
    # generate ground truth
    noise_var = ch.ones(1, 1)
    W = ch.ones(1, 1)

    print(f"gt weights: {W.tolist()}")
    print(f"gt noise var: {noise_var.item()}")
    # generate data
    X = ch.rand(NUM_SAMPLES, D)
    y = X @ W.T
    noised = y + ch.sqrt(noise_var) * ch.randn(y.size(0), 1)
    # generate ground-truth data
    phi = oracle.Left_Regression(ch.zeros(1))
    # phi = oracle.Identity()
    # truncate
    indices = phi(noised).nonzero()[:, 0]
    x_trunc, y_trunc = X[indices], noised[indices]
    alpha = x_trunc.size(0) / X.size(0)
    print(f"alpha: {alpha}")

    gt_norm = LinearRegression(fit_intercept=False)
    gt_norm.fit(X, noised)
    gt_ = ch.from_numpy(gt_norm.coef_)

    # calculate empirical noise variance for regression
    ols_trunc = LinearRegression(fit_intercept=False)
    ols_trunc.fit(x_trunc, y_trunc)
    emp_noise_var = ch.from_numpy(ols_trunc.predict(x_trunc) - y_trunc.numpy()).var(0)
    emp_ = ch.from_numpy(ols_trunc.coef_.flatten())
    print(f"emp weight estimates: {emp_.tolist()}")
    print(f"emp noise estimate: {emp_noise_var.item()}")
    emp_mse_loss = mse_loss(emp_, gt_)
    emp_var_l1 = float(ch.abs(emp_noise_var - noise_var))
    print(f"emp mse loss: {emp_mse_loss}")
    print(f"emp noise var l1: {emp_var_l1}")

    # scale y features by empirical noise variance
    y_trunc_emp_scale = y_trunc / ch.sqrt(emp_noise_var)
    # y_trunc_emp_scale = y_trunc
    phi_emp_scale = oracle.Left_Regression(phi.left / ch.sqrt(emp_noise_var))
    # train algorithm
    train_kwargs = Parameters(
        {
            "optimizer": "lbfgs",
            "trials": 1,
            "epochs": 5,
            "batch_size": -1,
            "var_lr": None,
            "verbose": True,
        }
    )
    unknown_trunc_reg = stats.TruncatedLinearRegression(
        train_kwargs, phi_emp_scale, alpha, fit_intercept=False
    )
    unknown_trunc_reg.fit(x_trunc, y_trunc_emp_scale)
    w_ = unknown_trunc_reg.best_coef_.flatten() * ch.sqrt(emp_noise_var)
    noise_var_ = unknown_trunc_reg.best_variance_ * emp_noise_var
    print(f"estimated_weights: {w_.tolist()}")
    print(f"estimated noise variance: {noise_var_.item()}")
    unknown_mse_loss = mse_loss(gt_, w_.flatten())
    print(f"unknown mse loss: {unknown_mse_loss}")
    unknown_var_l1 = float(ch.abs(noise_var - noise_var_))
    print(f"unknown var l1: {unknown_var_l1}")
    assert unknown_mse_loss <= 1e-1, (
        f"unknown mse loss: {unknown_mse_loss} is larger than: 1e-1"
    )
    assert unknown_var_l1 <= 1e-1, (
        f"unknown var l1: {unknown_var_l1} is larger than 1e-1"
    )


# left truncated regression with unknown noise variance in one dimension
def test_unknown_variance_truncated_regression_one_dimension():
    D = 1
    NUM_SAMPLES = 10000
    # generate ground truth
    noise_var = ch.ones(1, 1)
    W = ch.ones(1, D)
    W0 = ch.ones(1, 1)

    print(f"gt weights: {W.tolist()}")
    print(f"gt bias: {W0.tolist()}")
    print(f"gt noise var: {noise_var.item()}")
    # generate data
    X = ch.rand(NUM_SAMPLES, D)
    y = X @ W.T + W0
    noised = y + ch.sqrt(noise_var) * ch.randn(y.size(0), 1)
    # generate ground-truth data
    phi = oracle.Left_Regression(ch.ones(1))
    # phi = oracle.Identity()
    # truncate
    indices = phi(noised).nonzero()[:, 0]
    x_trunc, y_trunc = X[indices], noised[indices]
    alpha = x_trunc.size(0) / X.size(0)
    print(f"alpha: {alpha}")

    gt_norm = LinearRegression()
    gt_norm.fit(X, noised)
    gt_ = ch.from_numpy(np.concatenate([gt_norm.coef_.flatten(), gt_norm.intercept_]))

    # calculate empirical noise variance for regression
    ols_trunc = LinearRegression()
    ols_trunc.fit(x_trunc, y_trunc)
    emp_noise_var = ch.from_numpy(ols_trunc.predict(x_trunc) - y_trunc.numpy()).var(0)
    emp_ = ch.from_numpy(
        np.concatenate([ols_trunc.coef_.flatten(), ols_trunc.intercept_])
    )
    print(f"emp weight estimates: {emp_.tolist()}")
    print(f"emp noise estimate: {emp_noise_var.item()}")
    emp_mse_loss = mse_loss(emp_, gt_)
    emp_var_l1 = float(ch.abs(emp_noise_var - noise_var))
    print(f"emp mse loss: {emp_mse_loss}")
    print(f"emp noise var l1: {emp_var_l1}")

    # scale y features by empirical noise variance
    y_trunc_emp_scale = (y_trunc - ols_trunc.intercept_) / ch.sqrt(emp_noise_var)
    # y_trunc_emp_scale = y_trunc
    phi_emp_scale = oracle.Left_Regression(
        (phi.left - ols_trunc.intercept_) / ch.sqrt(emp_noise_var)
    )
    # train algorithm
    train_kwargs = Parameters(
        {
            "optimizer": "sgd",
            "trials": 1,
            "batch_size": 100,
            "var_lr": 1e-2,
            "verbose": True,
            "step_lr_gamma": 1.0,
            "early_stopping": True,
            "gradient_steps": 1,
        }
    )
    unknown_trunc_reg = stats.TruncatedLinearRegression(
        train_kwargs, phi_emp_scale, alpha
    )
    unknown_trunc_reg.fit(x_trunc, y_trunc_emp_scale)
    w_ = ch.cat(
        [
            (unknown_trunc_reg.best_coef_).flatten(),
            unknown_trunc_reg.best_intercept_ + ols_trunc.intercept_,
        ]
    ) * ch.sqrt(emp_noise_var)
    noise_var_ = unknown_trunc_reg.best_variance_ * emp_noise_var
    print(f"estimated_weights: {w_.tolist()}")
    print(f"estimated noise variance: {noise_var_.item()}")
    unknown_mse_loss = mse_loss(gt_, w_.flatten())
    print(f"unknown mse loss: {unknown_mse_loss}")
    unknown_var_l1 = float(ch.abs(noise_var - noise_var_))
    print(f"unknown var l1: {unknown_var_l1}")
    assert unknown_mse_loss <= 1e-1, (
        f"unknown mse loss: {unknown_mse_loss} is larger than 1e-1"
    )
    assert unknown_var_l1 <= 1e-1, (
        f"unknown var l1: {unknown_var_l1} is larger than 1e-1"
    )


# left truncated regression with unknown noise variance in one dimension with noise variance 3
def test_unknown_variance_truncated_regression_one_dimension_with_noise_var_3():
    D = 1
    NUM_SAMPLES = 3000
    # generate ground truth
    noise_var = 3 * ch.ones(1, 1)
    W = ch.ones(1, D)
    W0 = ch.ones(1, 1)

    print(f"gt weights: {W.tolist()}")
    print(f"gt bias: {W0.tolist()}")
    print(f"gt noise var: {noise_var.item()}")
    # generate data
    # X = ch.rand(NUM_SAMPLES, D)
    a, b = -5, 5
    X = a + (b - a) * ch.randn(NUM_SAMPLES, D)
    y = X @ W.T + W0
    noised = y + ch.sqrt(noise_var) * ch.randn(y.size(0), 1)
    # generate ground-truth data
    phi = oracle.Left_Regression(ch.ones(1))
    # phi = oracle.Identity()
    # truncate
    indices = phi(noised).nonzero()[:, 0]
    x_trunc, y_trunc = X[indices], noised[indices]
    alpha = x_trunc.size(0) / X.size(0)
    print(f"alpha: {alpha}")

    gt_norm = LinearRegression()
    gt_norm.fit(X, noised)
    gt_ = ch.from_numpy(np.concatenate([gt_norm.coef_.flatten(), gt_norm.intercept_]))

    # calculate empirical noise variance for regression
    ols_trunc = LinearRegression()
    ols_trunc.fit(x_trunc, y_trunc)
    emp_noise_var = ch.from_numpy(ols_trunc.predict(x_trunc) - y_trunc.numpy()).var(0)
    emp_ = ch.from_numpy(
        np.concatenate([ols_trunc.coef_.flatten(), ols_trunc.intercept_])
    )
    print(f"emp weight estimates: {emp_.tolist()}")
    print(f"emp noise estimate: {emp_noise_var.item()}")
    emp_mse_loss = mse_loss(emp_, gt_)
    emp_var_l1 = float(ch.abs(emp_noise_var - noise_var))
    print(f"emp mse loss: {emp_mse_loss}")
    print(f"emp noise var l1: {emp_var_l1}")

    # scale y features by empirical noise variance
    # train algorithm
    args = Parameters(
        {
            "trials": 1,
            "epochs": 10,
            "optimizer": "lbfgs",
            "batch_size": -1,
            "var_lr": None,
            "verbose": True,
            "num_samples": 5000,
        }
    )
    unknown_trunc_reg = stats.TruncatedLinearRegression(args, phi, alpha)
    unknown_trunc_reg.fit(x_trunc, y_trunc)
    w_ = ch.cat(
        [
            (unknown_trunc_reg.best_coef_).flatten(),
            unknown_trunc_reg.best_intercept_[..., None],
        ]
    )
    noise_var_ = unknown_trunc_reg.best_variance_
    print(f"estimated_weights: {w_.tolist()}")
    print(f"estimated noise variance: {noise_var_.item()}")
    unknown_mse_loss = mse_loss(gt_, w_.flatten())
    print(f"unknown mse loss: {unknown_mse_loss}")
    unknown_var_l1 = float(ch.abs(noise_var - noise_var_))
    print(f"unknown var l1: {unknown_var_l1}")
    assert unknown_mse_loss <= 1e-1, (
        f"unknown mse loss: {unknown_mse_loss} is larger than 1e-1"
    )
    assert unknown_var_l1 <= 1e-1, (
        f"unknown var l1: {unknown_var_l1} is larger than 1e-1"
    )


def test_unknown_variance_truncated_regression_ten_dimensions_no_intercept():
    D = 10
    NUM_SAMPLES = 20000
    # generate ground truth
    noise_var = 3 * ch.ones(1, 1)
    W = ch.ones(1, D)

    print(f"gt weights: {W.tolist()}")
    print(f"gt noise var: {noise_var.item()}")
    # generate data
    X = ch.rand(NUM_SAMPLES, D)
    y = X @ W.T
    noised = y + ch.sqrt(noise_var) * ch.randn(y.size(0), 1)
    # generate ground-truth data
    phi = oracle.Left_Regression(5 * ch.ones(1))
    # truncate
    indices = phi(noised).nonzero()[:, 0]
    x_trunc, y_trunc = X[indices], noised[indices]
    alpha = x_trunc.size(0) / X.size(0)
    print(f"alpha: {alpha}")

    gt_norm = LinearRegression(fit_intercept=False)
    gt_norm.fit(X, noised)
    gt_ = ch.from_numpy(gt_norm.coef_.flatten())

    # calculate empirical noise variance for regression
    ols_trunc = LinearRegression(fit_intercept=False)
    ols_trunc.fit(x_trunc, y_trunc)
    emp_noise_var = ch.from_numpy(ols_trunc.predict(x_trunc) - y_trunc.numpy()).var(0)
    emp_ = ch.from_numpy(ols_trunc.coef_.flatten())
    print(f"emp weight estimates: {emp_.tolist()}")
    print(f"emp noise estimate: {emp_noise_var.item()}")
    emp_mse_loss = mse_loss(emp_, gt_)
    emp_var_l1 = float(ch.abs(emp_noise_var - noise_var))
    print(f"emp mse loss: {emp_mse_loss}")
    print(f"emp noise var l1: {emp_var_l1}")
    # train algorithm
    train_kwargs = Parameters(
        {
            "epochs": 5,
            "optimizer": "lbfgs",
            "trials": 1,
            "batch_size": -1,
            "var_lr": None,
            "verbose": True,
            "early_stopping": True,
            "gradient_steps": 500,
            # 'lbfgs_lr': 2.0
        }
    )
    unknown_trunc_reg = stats.TruncatedLinearRegression(
        train_kwargs, phi, alpha, fit_intercept=False
    )
    unknown_trunc_reg.fit(x_trunc, y_trunc)
    # unknown_trunc_reg.fit(x_trunc.repeat(1, 1), y_trunc_emp_scale.repeat(1, 1))
    w_ = unknown_trunc_reg.best_coef_.flatten()
    noise_var_ = unknown_trunc_reg.best_variance_
    print(f"estimated_weights: {w_.tolist()}")
    print(f"estimated noise variance: {noise_var_.item()}")
    unknown_mse_loss = mse_loss(gt_, w_.flatten())
    print(f"unknown mse loss: {unknown_mse_loss}")
    unknown_var_l1 = float(ch.abs(noise_var - noise_var_))
    print(f"unknown var l1: {unknown_var_l1}")
    assert unknown_mse_loss <= 1e-1, (
        f"unknown mse loss: {unknown_mse_loss} is larger than 1e-1"
    )
    assert unknown_var_l1 <= 1e-1, (
        f"unknown var l1: {unknown_var_l1} is larger than 1e-1"
    )


# left truncated regression with unknown noise variance in ten dimensions
def test_unknown_variance_truncated_regression_ten_dimensions():
    D = 10
    NUM_SAMPLES = 10000
    # generate ground truth
    noise_var = 3 * ch.ones(1, 1)
    W = ch.ones(1, D)
    W0 = ch.ones(1, 1)

    print(f"gt weights: {ch.cat([W, W0], dim=1).tolist()}")
    print(f"gt noise var: {noise_var.item()}")
    # generate data
    X = ch.rand(NUM_SAMPLES, D)
    y = X @ W.T + W0
    noised = y + ch.sqrt(noise_var) * ch.randn(y.size(0), 1)
    # generate ground-truth data
    phi = oracle.Left_Regression(6 * ch.ones(1))
    # truncate
    indices = phi(noised).nonzero()[:, 0]
    x_trunc, y_trunc = X[indices], noised[indices]
    alpha = x_trunc.size(0) / X.size(0)
    print(f"alpha: {alpha}")

    gt_norm = LinearRegression()
    gt_norm.fit(X, noised)
    gt_ = ch.from_numpy(np.concatenate([gt_norm.coef_.flatten(), gt_norm.intercept_]))

    # calculate empirical noise variance for regression
    ols_trunc = LinearRegression()
    ols_trunc.fit(x_trunc, y_trunc)
    emp_noise_var = ch.from_numpy(ols_trunc.predict(x_trunc) - y_trunc.numpy()).var(0)
    emp_ = ch.from_numpy(
        np.concatenate([ols_trunc.coef_.flatten(), ols_trunc.intercept_])
    )
    print(f"emp weight estimates: {emp_.tolist()}")
    print(f"emp noise estimate: {emp_noise_var.item()}")
    emp_mse_loss = mse_loss(emp_, gt_)
    emp_var_l1 = float(ch.abs(emp_noise_var - noise_var))
    print(f"emp mse loss: {emp_mse_loss}")
    print(f"emp noise var l1: {emp_var_l1}")
    # train algorithm
    train_kwargs = Parameters(
        {
            "epochs": 10,
            "optimizer": "lbfgs",
            "trials": 1,
            "batch_size": -1,
            "var_lr": None,
            "verbose": True,
            "early_stopping": True,
            "gradient_steps": 500,
        }
    )
    unknown_trunc_reg = stats.TruncatedLinearRegression(train_kwargs, phi, alpha)
    unknown_trunc_reg.fit(x_trunc, y_trunc)
    w_ = ch.cat(
        [
            (unknown_trunc_reg.best_coef_).flatten(),
            unknown_trunc_reg.best_intercept_[..., None],
        ]
    ) * ch.sqrt(emp_noise_var)
    noise_var_ = unknown_trunc_reg.best_variance_ * emp_noise_var
    print(f"estimated_weights: {w_.tolist()}")
    print(f"estimated noise variance: {noise_var_.item()}")
    unknown_mse_loss = mse_loss(gt_, w_.flatten())
    print(f"unknown mse loss: {unknown_mse_loss}")
    unknown_var_l1 = float(ch.abs(noise_var - noise_var_))
    print(f"unknown var l1: {unknown_var_l1}")
    assert unknown_mse_loss <= 1e-1, (
        f"unknown mse loss: {unknown_mse_loss} is larger than 1e-1"
    )
    assert unknown_var_l1 <= 1e-1, (
        f"unknown var l1: {unknown_var_l1} is larger than 1e-1"
    )


# left truncated regression with unknown noise variance in ten dimensions
def test_unknown_variance_truncated_regression_fifty_dimensions():
    D = 50
    NUM_SAMPLES = 100000
    # generate ground truth
    noise_var = 5 * ch.ones(1, 1)
    W = ch.ones(1, D)
    W0 = ch.ones(1, 1)

    print(f"gt weights: {ch.cat([W, W0], dim=1).tolist()}")
    print(f"gt noise var: {noise_var.item()}")
    # generate data
    X = ch.rand(NUM_SAMPLES, D)
    y = X @ W.T + W0
    noised = y + ch.sqrt(noise_var) * ch.randn(y.size(0), 1)
    # generate ground-truth data
    phi = oracle.Left_Regression(25 * ch.ones(1))
    # truncate
    indices = phi(noised).nonzero()[:, 0]
    x_trunc, y_trunc = X[indices], noised[indices]
    alpha = x_trunc.size(0) / X.size(0)
    print(f"alpha: {alpha}")

    gt_norm = LinearRegression()
    gt_norm.fit(X, noised)
    gt_ = ch.from_numpy(np.concatenate([gt_norm.coef_.flatten(), gt_norm.intercept_]))

    # calculate empirical noise variance for regression
    ols_trunc = LinearRegression()
    ols_trunc.fit(x_trunc, y_trunc)
    emp_noise_var = ch.from_numpy(ols_trunc.predict(x_trunc) - y_trunc.numpy()).var(0)
    emp_ = ch.from_numpy(
        np.concatenate([ols_trunc.coef_.flatten(), ols_trunc.intercept_])
    )
    print(f"emp weight estimates: {emp_.tolist()}")
    print(f"emp noise estimate: {emp_noise_var.item()}")
    emp_mse_loss = mse_loss(emp_, gt_)
    emp_var_l1 = float(ch.abs(emp_noise_var - noise_var))
    print(f"emp mse loss: {emp_mse_loss}")
    print(f"emp noise var l1: {emp_var_l1}")
    # scale y features by empirical noise variance
    y_trunc_emp_scale = y_trunc / ch.sqrt(emp_noise_var)
    phi_emp_scale = oracle.Left_Regression(phi.left / ch.sqrt(emp_noise_var))
    # train algorithm
    train_kwargs = Parameters(
        {
            "epochs": 15,
            "optimizer": "sgd",
            "trials": 1,
            "batch_size": 100,
            "lr": 1e-1,
            # 'var_lr': 1e-2,
            "verbose": True,
            "early_stopping": True,
            "gradient_steps": 2500,
            "step_lr_gamma": 1.0,
        }
    )
    unknown_trunc_reg = stats.TruncatedLinearRegression(
        train_kwargs, phi_emp_scale, alpha
    )
    unknown_trunc_reg.fit(x_trunc, y_trunc_emp_scale)
    w_ = ch.cat(
        [
            (unknown_trunc_reg.best_coef_).flatten(),
            unknown_trunc_reg.best_intercept_[..., None],
        ]
    ) * ch.sqrt(emp_noise_var)
    noise_var_ = unknown_trunc_reg.best_variance_ * emp_noise_var
    print(f"estimated_weights: {w_.tolist()}")
    print(f"estimated noise variance: {noise_var_.item()}")
    unknown_mse_loss = mse_loss(gt_, w_.flatten())
    print(f"unknown mse loss: {unknown_mse_loss}")
    unknown_var_l1 = float(ch.abs(noise_var - noise_var_))
    print(f"unknown var l1: {unknown_var_l1}")
    assert unknown_mse_loss <= 1e-1, (
        f"unknown mse loss: {unknown_mse_loss} is larger than 1e-1"
    )
    assert unknown_var_l1 <= 1e-1, (
        f"unknown var l1: {unknown_var_l1} is larger than 1e-1"
    )


def test_truncated_lasso_regression_one_dimension_no_intercept():
    L1 = 0.1
    NUM_SAMPLES = 1000
    # generate ground truth
    NOISE_VAR = ch.ones(1, 1)
    W = ch.ones(1, 1)
    print(f"gt weight: {W}")
    print(f"gt noise var: {NOISE_VAR}")

    # generate data
    X = ch.rand(NUM_SAMPLES, 1)
    y = X @ W.T
    noised = y + ch.sqrt(NOISE_VAR) * ch.randn(y.size(0), 1)
    # generate ground-truth data
    phi = oracle.Left_Regression(ch.zeros(1))
    # truncate
    indices = phi(noised).nonzero()[:, 0]
    x_trunc, y_trunc = X[indices], noised[indices]
    alpha = x_trunc.size(0) / X.size(0)
    print(f"alpha: {alpha}")

    gt_lasso = LassoCV(alphas=[L1], fit_intercept=False)
    gt_lasso.fit(X, y)
    gt_ = ch.from_numpy(np.concatenate([gt_lasso.coef_.flatten()]))
    print(f"ground truth weights: {gt_}")

    # calculate empirical noise variance for regression
    lasso_trunc = LassoCV(alphas=[L1], fit_intercept=False)
    lasso_trunc.fit(x_trunc, y_trunc)
    emp_ = ch.from_numpy(np.concatenate([lasso_trunc.coef_.flatten()]))
    print(f"empirical weights: {emp_}")
    emp_mse_loss = mse_loss(emp_, gt_)
    print(f"emp mse loss: {emp_mse_loss}")

    # scale y features
    y_trunc_scale = y_trunc / ch.sqrt(NOISE_VAR)
    phi_scale = oracle.Left_Regression(phi.left / ch.sqrt(NOISE_VAR))
    # train algorithm
    train_kwargs = Parameters(
        {
            "lr": 1e-1,
            "batch_size": 10,
            "gradient_steps": 1000,
            "trials": 1,
            "verbose": True,
        }
    )
    trunc_reg = stats.TruncatedLassoRegression(
        train_kwargs,
        phi_scale,
        alpha,
        l1=L1,
        fit_intercept=False,
        noise_var=ch.ones(1, 1),
    )
    trunc_reg.fit(x_trunc, y_trunc_scale)
    w_ = ch.cat([(trunc_reg.best_coef_).flatten()]) * ch.sqrt(NOISE_VAR)
    print(f"estimated weights: {w_}")
    trunc_mse_loss = mse_loss(gt_, w_.flatten())
    print(f"trunc mse loss: {trunc_mse_loss}")
    msg = f"trunc mse loss is larger than empirical mse loss. known mse loss is {trunc_mse_loss}, and empirical mse loss is: {emp_mse_loss}"
    assert trunc_mse_loss <= emp_mse_loss, msg
    msg = f"trunc mse loss: {trunc_mse_loss}, which is larger than 1e-1"
    assert trunc_mse_loss <= 1e-1, msg


def test_truncated_lasso_regression_one_dimension():
    L1 = 0.1
    NUM_SAMPLES = 1000
    # generate ground truth
    NOISE_VAR = ch.ones(1, 1)
    W = ch.ones(1, 1)
    W0 = ch.ones(1, 1)
    print(f"gt weight: {W}")
    print(f"gt bias: {W0}")
    print(f"gt noise var: {NOISE_VAR}")

    # generate data
    X = ch.rand(NUM_SAMPLES, 1)
    y = X @ W.T + W0
    noised = y + ch.sqrt(NOISE_VAR) * ch.randn(y.size(0), 1)
    # generate ground-truth data
    phi = oracle.Left_Regression(2)
    # truncate
    indices = phi(noised).nonzero()[:, 0]
    x_trunc, y_trunc = X[indices], noised[indices]
    alpha = x_trunc.size(0) / X.size(0)
    print(f"alpha: {alpha}")

    gt_lasso = LassoCV(alphas=[L1])
    gt_lasso.fit(X, y)
    gt_ = ch.from_numpy(
        np.concatenate(
            [
                gt_lasso.coef_.flatten(),
                gt_lasso.intercept_.reshape(
                    1,
                ),
            ]
        )
    )
    print(f"ground truth weights: {gt_}")

    # calculate empirical noise variance for regression
    lasso_trunc = LassoCV(alphas=[L1])
    lasso_trunc.fit(x_trunc, y_trunc)
    emp_ = ch.from_numpy(
        np.concatenate(
            [
                lasso_trunc.coef_.flatten(),
                lasso_trunc.intercept_.reshape(
                    1,
                ),
            ]
        )
    )
    print(f"empirical weights: {emp_}")
    emp_mse_loss = mse_loss(emp_, gt_)
    print(f"emp mse loss: {emp_mse_loss}")

    # scale y features
    y_trunc_scale = y_trunc / ch.sqrt(NOISE_VAR)
    phi_scale = oracle.Left_Regression(phi.left / ch.sqrt(NOISE_VAR))
    # train algorithm
    train_kwargs = Parameters(
        {
            "lr": 1e-1,
            "batch_size": 10,
            "gradient_steps": 1000,
            "trials": 1,
            "verbose": True,
        }
    )
    trunc_reg = stats.TruncatedLassoRegression(
        train_kwargs, phi_scale, alpha, l1=L1, noise_var=ch.ones(1, 1)
    )
    trunc_reg.fit(x_trunc, y_trunc_scale)
    w_ = ch.cat(
        [(trunc_reg.best_coef_).flatten(), trunc_reg.best_intercept_]
    ) * ch.sqrt(NOISE_VAR)
    print(f"estimated weights: {w_}")
    trunc_mse_loss = mse_loss(gt_, w_.flatten())
    print(f"trunc mse loss: {trunc_mse_loss}")
    msg = f"trunc mse loss is larger than empirical mse loss. known mse loss is {trunc_mse_loss}, and empirical mse loss is: {emp_mse_loss}"
    assert trunc_mse_loss <= emp_mse_loss, msg
    msg = f"trunc mse loss: {trunc_mse_loss}, which is larger than 1e-1"
    assert trunc_mse_loss <= 1e-1, msg


def test_truncated_lasso_regression_ten_dimensions():
    L1 = 0.1
    NUM_SAMPLES = 10000
    D = 10
    # generate ground truth
    NOISE_VAR = ch.ones(1, 1)
    W = ch.ones(1, D)
    W0 = ch.ones(1, 1)
    print(f"gt weight: {W}")
    print(f"gt bias: {W0}")
    print(f"gt noise var: {NOISE_VAR}")

    # generate data
    X = ch.rand(NUM_SAMPLES, D)
    y = X @ W.T + W0
    noised = y + ch.sqrt(NOISE_VAR) * ch.randn(y.size(0), 1)
    # generate ground-truth data
    phi = oracle.Left_Regression(5)
    # truncate
    indices = phi(noised).nonzero()[:, 0]
    x_trunc, y_trunc = X[indices], noised[indices]
    alpha = x_trunc.size(0) / X.size(0)
    print(f"alpha: {alpha}")

    gt_lasso = LassoCV(alphas=[L1])
    gt_lasso.fit(X, y)
    gt_ = ch.from_numpy(
        np.concatenate(
            [
                gt_lasso.coef_.flatten(),
                gt_lasso.intercept_.reshape(
                    1,
                ),
            ]
        )
    )
    print(f"ground truth weights: {gt_}")

    # calculate empirical noise variance for regression
    lasso_trunc = LassoCV(alphas=[L1])
    lasso_trunc.fit(x_trunc, y_trunc)
    emp_ = ch.from_numpy(
        np.concatenate(
            [
                lasso_trunc.coef_.flatten(),
                lasso_trunc.intercept_.reshape(
                    1,
                ),
            ]
        )
    )
    print(f"empirical weights: {emp_}")
    emp_mse_loss = mse_loss(emp_, gt_)
    print(f"emp mse loss: {emp_mse_loss}")

    # scale y features
    y_trunc_scale = y_trunc / ch.sqrt(NOISE_VAR)
    phi_scale = oracle.Left_Regression(phi.left / ch.sqrt(NOISE_VAR))
    # train algorithm
    train_kwargs = Parameters(
        {
            "lr": 1e-1,
            "batch_size": 10,
            "gradient_steps": 1000,
            "trials": 1,
            "verbose": True,
        }
    )
    trunc_reg = stats.TruncatedLassoRegression(
        train_kwargs, phi_scale, alpha, l1=L1, noise_var=ch.ones(1, 1)
    )
    trunc_reg.fit(x_trunc, y_trunc_scale)
    w_ = ch.cat(
        [(trunc_reg.best_coef_).flatten(), trunc_reg.best_intercept_]
    ) * ch.sqrt(NOISE_VAR)
    print(f"estimated weights: {w_}")
    trunc_mse_loss = mse_loss(gt_, w_.flatten())
    print(f"trunc mse loss: {trunc_mse_loss}")
    msg = f"trunc mse loss is larger than empirical mse loss. known mse loss is {trunc_mse_loss}, and empirical mse loss is: {emp_mse_loss}"
    assert trunc_mse_loss <= emp_mse_loss, msg
    msg = f"trunc mse loss: {trunc_mse_loss}, which is larger than 1e-1"
    assert trunc_mse_loss <= 1e-1, msg


def test_truncated_dependent_regression():
    D = 10  # number of dimensions for A_{*} matrix
    T = 10000  # uncensored system trajectory length

    spectral_norm = float("inf")
    while spectral_norm > 1.0:
        A = 0.25 * ch.randn((D, D))
        spectral_norm = calc_spectral_norm(A)

    spectral_norm = calc_spectral_norm(A)
    print(f"A spectral norm: {calc_spectral_norm(A)}")

    phi = oracle.LogitBall(4.0)

    X, Y = ch.Tensor([]), ch.Tensor([])
    NOISE_VAR = ch.eye(D)
    M = ch.distributions.MultivariateNormal(ch.zeros(D), NOISE_VAR)
    x_t = ch.zeros((1, D))
    for i in range(T):
        noise = M.sample()
        y_t = x_t @ A + noise
        if phi(y_t):  # returns a boolean
            X = ch.cat([X, x_t])
            Y = ch.cat([Y, y_t])
        x_t = y_t

    alpha = X.size(0) / T

    print(f"alpha: {alpha}")

    train_kwargs = Parameters(
        {
            "c_eta": 0.5,
            "epochs": 5,
            "trials": 1,
            "batch_size": 1,
            "num_samples": 100,
            "T": X.size(0),
            "c_s": 10.0,
            "alpha": alpha,
            "tol": 1e-1,
            "c_gamma": 2.0,
        }
    )
    trunc_lds = stats.TruncatedLinearRegression(  # pylint: disable=no-value-for-parameter
        phi, train_kwargs, noise_var=NOISE_VAR, dependent=True, rand_seed=seed
    )
    trunc_lds.fit(X, Y)
    A_ = trunc_lds.coef_
    A0_ = trunc_lds.ols_coef_
    trunc_spec_norm = calc_spectral_norm(A - A_)
    emp_spec_norm = calc_spectral_norm(A - A0_)

    print(f"alpha: {alpha}")
    print(f"A spectral norm: {spectral_norm}")
    print(f"truncated spectral norm: {trunc_spec_norm}")
    print(f"ols spectral norm: {emp_spec_norm}")

    assert trunc_spec_norm <= emp_spec_norm, (
        f"truncated spectral norm {trunc_spec_norm}, while OLS spectral norm is: {emp_spec_norm}"
    )
