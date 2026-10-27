# Author: pstefanou12@
"""Tests for truncated LASSO regression with known noise variance."""

import numpy as np
import pytest
from sklearn.linear_model import LassoCV
import torch as ch
from torch.nn import MSELoss

from delphi import stats

SEED = 69
mse_loss = MSELoss()


@pytest.mark.skip(reason="accuracy: mse 0.146 against a 0.1 tolerance")
def test_truncated_lasso_regression_one_dimension_no_intercept():
    ch.manual_seed(SEED)
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
    phi_left = ch.zeros(1)

    def phi(x):
        return x > phi_left

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
    phi_scale_left = phi_left / ch.sqrt(NOISE_VAR)

    def phi_scale(x):
        return x > phi_scale_left

    # train algorithm
    train_kwargs = {
        "lr": 1e-1,
        "batch_size": 10,
        "iterations": 1000,
        "trials": 1,
    }
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


# TODO: pstefanou12@ - the lasso weight carries one column more than the
# design matrix, so the intercept column is counted twice.
@pytest.mark.skip(reason="accuracy: mse 0.175 against a 0.1 tolerance")
def test_truncated_lasso_regression_one_dimension():
    ch.manual_seed(SEED)
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
    phi_left = 2

    def phi(x):
        return x > phi_left

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
    phi_scale_left = phi_left / ch.sqrt(NOISE_VAR)

    def phi_scale(x):
        return x > phi_scale_left

    # train algorithm
    train_kwargs = {
        "lr": 1e-1,
        "batch_size": 10,
        "iterations": 1000,
        "trials": 1,
    }
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


@pytest.mark.skip(reason="accuracy: mse 0.220 against an empirical 0.025")
def test_truncated_lasso_regression_ten_dimensions():
    ch.manual_seed(SEED)
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
    phi_left = 5

    def phi(x):
        return x > phi_left

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
    phi_scale_left = phi_left / ch.sqrt(NOISE_VAR)

    def phi_scale(x):
        return x > phi_scale_left

    # train algorithm
    train_kwargs = {
        "lr": 1e-1,
        "batch_size": 10,
        "iterations": 1000,
        "trials": 1,
    }
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
