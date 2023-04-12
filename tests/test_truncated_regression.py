"""
Test suite for truncated linear regression. 
Includes: 
    -Truncated regression with known variance
    -Truncated regression with unknown variance 
    -Truncated regression with known regression and temporal dependencies
"""
import numpy as np
import torch as ch
from torch import Tensor
from torch.distributions import Uniform, Gumbel
from torch.nn import MSELoss
from sklearn.linear_model import LinearRegression

from delphi import stats 
from delphi import oracle
from delphi.utils.helpers import Parameters, calc_spectral_norm

# CONSTANTS
mse_loss =  MSELoss()
seed = 69 

# left truncated linear regression
def test_known_truncated_regression():
    D, K = 100, 1
    SAMPLES = 10000
    w_ = Uniform(-1, 1)
    M = Uniform(-10, 10)
    # generate ground truth
    NOISE_VAR = 20*ch.ones(1, 1)
    W = w_.sample([K, D])
    W0 = w_.sample([1, 1])

    print(f"gt weight: {W}")
    print(f"gt intercept: {W0}")
    print(f"gt noise var: {NOISE_VAR}")
    # generate data
    X = M.sample(ch.Size([SAMPLES, D])) if isinstance(M, Uniform) else M.sample(ch.Size([SAMPLES]))
    y = X@W.T + W0 
    noised = y + ch.sqrt(NOISE_VAR) * ch.randn(y.size(0), 1)
    # generate ground-truth data
    phi = oracle.Left_Regression(ch.zeros(1))
    # truncate
    indices = phi(noised).nonzero()[:,0]
    x_trunc, y_trunc = X[indices], noised[indices]
    alpha = x_trunc.size(0) / X.size(0)
    print(f'alpha: {alpha}')

    gt_norm = LinearRegression()
    gt_norm.fit(X, y)
    gt_ = ch.from_numpy(np.concatenate([gt_norm.coef_.flatten(), gt_norm.intercept_]))

    # calculate empirical noise variance for regression 
    ols_trunc = LinearRegression()
    ols_trunc.fit(x_trunc, y_trunc)
    emp_ = ch.from_numpy(np.concatenate([ols_trunc.coef_.flatten(), ols_trunc.intercept_]))
    print(f'empirical weights: {emp_}')
    emp_mse_loss = mse_loss(emp_, gt_)
    print(f'emp mse loss: {emp_mse_loss}')

    # scale y features
    y_trunc_scale = y_trunc / ch.sqrt(NOISE_VAR)
    phi_scale = oracle.Left_Regression(phi.left / ch.sqrt(NOISE_VAR))
    # train algorithm
    train_kwargs = Parameters({'phi': phi_scale, 
                            'alpha': alpha,
                            'epochs': 10,
                            'lr': 5e-1,
                            'num_samples': 10,
                            'batch_size': 1,
                            'trials': 1,
                            'constant': True,
                            'noise_var': ch.ones(1, 1)}) 
    trunc_reg = stats.TruncatedLinearRegression(train_kwargs)
    trunc_reg.fit(x_trunc, y_trunc_scale)
    w_ = ch.cat([(trunc_reg.best_coef_).flatten(), trunc_reg.best_intercept_]) * ch.sqrt(NOISE_VAR)
    print(f'estimated weights: {w_}')
    known_mse_loss = mse_loss(gt_, w_.flatten())
    print(f'known mse loss: {known_mse_loss}')
    msg = f'known mse loss is larger than empirical mse loss. known mse loss is {known_mse_loss}, and empirical mse loss is: {emp_mse_loss}'
    assert known_mse_loss <= emp_mse_loss, msg
        
    avg_w_ = ch.cat([(trunc_reg.avg_coef_).flatten(), trunc_reg.avg_intercept_]) * ch.sqrt(NOISE_VAR)
    avg_known_mse_loss = mse_loss(gt_, avg_w_.flatten())
    print(f'avg known mse loss: {avg_known_mse_loss}')
    msg = f'avg known mse loss is larger than empirical mse loss. avg known mse loss is {avg_known_mse_loss}, and empirical mse loss is: {emp_mse_loss}'
    assert avg_known_mse_loss <= emp_mse_loss, msg
    
def test_unknown_truncated_regression():
    D, K = 10, 1
    SAMPLES = 1000
    w_ = Uniform(-1, 1)
    M = Uniform(-10, 10)
    # generate ground truth
    noise_var = 10*ch.ones(1, 1)
    W = w_.sample([K, D])
    W0 = w_.sample([1, 1])

    print(f"gt weight: {W}")
    print(f"gt intercept: {W0}")
    print(f"gt noise var: {noise_var}")
    # generate data
    X = M.sample(ch.Size([SAMPLES, D])) if isinstance(M, Uniform) else M.sample(ch.Size([SAMPLES]))
    y = X@W.T + W0 
    noised = y + ch.sqrt(noise_var) * ch.randn(y.size(0), 1)
    # generate ground-truth data
    phi = oracle.Left_Regression(ch.zeros(1))
    # truncate
    indices = phi(noised).nonzero()[:,0]
    x_trunc, y_trunc = X[indices], noised[indices]
    alpha = x_trunc.size(0) / X.size(0)
    print(f'alpha: {alpha}')

    gt_norm = LinearRegression()
    gt_norm.fit(X, noised)
    gt_ = ch.from_numpy(np.concatenate([gt_norm.coef_.flatten(), gt_norm.intercept_]))

    # calculate empirical noise variance for regression 
    ols_trunc = LinearRegression()
    ols_trunc.fit(x_trunc, y_trunc)
    emp_noise_var = ch.from_numpy(ols_trunc.predict(X) - noised.numpy()).var(0)
    emp_ = ch.from_numpy(np.concatenate([ols_trunc.coef_.flatten(), ols_trunc.intercept_]))
    emp_mse_loss = mse_loss(emp_, gt_)
    emp_var_l1 = float(ch.abs(emp_noise_var - noise_var))
    print(f'emp mse loss: {emp_mse_loss}')
    print(f'emp noise var l1: {emp_var_l1}')

    # scale y features by empirical noise variance
    y_trunc_emp_scale = y_trunc / ch.sqrt(emp_noise_var)
    phi_emp_scale = oracle.Left_Regression(phi.left / ch.sqrt(emp_noise_var))
    # train algorithm
    train_kwargs = Parameters({'phi': phi_emp_scale, 
                                'alpha': alpha,
                                'trials': 1,
                                'batch_size': 10,
                                'var_lr': 1e-2,})
    unknown_trunc_reg = stats.TruncatedLinearRegression(train_kwargs)
    unknown_trunc_reg.fit(x_trunc.repeat(100, 1), y_trunc_emp_scale.repeat(100, 1))
    w_ = ch.cat([(unknown_trunc_reg.best_coef_).flatten(), unknown_trunc_reg.best_intercept_]) * ch.sqrt(emp_noise_var)
    noise_var_ = unknown_trunc_reg.variance_ * emp_noise_var
    unknown_mse_loss = mse_loss(gt_, w_.flatten())
    print(f'unknown mse loss: {unknown_mse_loss}')
    unknown_var_l1 = float(ch.abs(noise_var - noise_var_))
    print(f'unknown var l1: {unknown_var_l1}')
    assert unknown_mse_loss <= emp_mse_loss, f'unknown mse loss: {unknown_mse_loss}'
    assert unknown_var_l1 <= emp_var_l1, f'unknown var l1: {unknown_var_l1}'

def test_truncated_dependent_regression(): 
    D = 10 # number of dimensions for A_{*} matrix
    T = 10000 # uncensored system trajectory length 

    spectral_norm = float('inf')
    while spectral_norm > 1.0: 
        A = .25 * ch.randn((D, D))
        spectral_norm = calc_spectral_norm(A)

    spectral_norm = calc_spectral_norm(A)
    print(f'A spectral norm: {calc_spectral_norm(A)}')

    phi = oracle.LogitBall(4.0)

    X, Y = ch.Tensor([]), ch.Tensor([])
    NOISE_VAR = ch.eye(D)
    M = ch.distributions.MultivariateNormal(ch.zeros(D), NOISE_VAR) 
    x_t = ch.zeros((1, D))
    for i in range(T): 
        noise = M.sample()
        y_t = x_t@A + noise
        if phi(y_t): # returns a boolean 
            X = ch.cat([X, x_t])
            Y = ch.cat([Y, y_t])
        x_t = y_t

    alpha = X.size(0) / T

    print(f'alpha: {alpha}')

    train_kwargs = Parameters({
        'phi': phi, 
        'c_eta': .5,
        'epochs': 5, 
        'trials': 1, 
        'batch_size': 1,
        'num_samples': 100,
        'T': X.size(0),
        'trials': 1,
        'c_s': 10.0,
        'alpha': alpha,
        'tol': 1e-1,
        'noise_var': NOISE_VAR, 
        'c_gamma': 2.0,
    })
    trunc_lds = stats.TruncatedLinearRegression(train_kwargs, 
                                                dependent=True, 
                                                rand_seed=seed)
    trunc_lds.fit(X, Y)
    A_ = trunc_lds.best_coef_
    A0_ = trunc_lds.ols_coef_
    A_avg = trunc_lds.avg_coef_
    trunc_spec_norm = calc_spectral_norm(A - A_)
    emp_spec_norm = calc_spectral_norm(A - A0_)
    avg_trunc_spec_norm = calc_spectral_norm(A - A_avg)

    print(f'alpha: {alpha}')
    print(f'A spectral norm: {spectral_norm}')
    print(f'truncated spectral norm: {trunc_spec_norm}')
    print(f'average truncated spectral norm: {avg_trunc_spec_norm}')
    print(f'ols spectral norm: {emp_spec_norm}')

    assert trunc_spec_norm <= emp_spec_norm, f"truncated spectral norm {trunc_spec_norm}, while OLS spectral norm is: {emp_spec_norm}"
    assert avg_trunc_spec_norm <= emp_spec_norm, f"average truncated spectral norm {avg_trunc_spec_norm}, while OLS spectral norm is: {emp_spec_norm}"