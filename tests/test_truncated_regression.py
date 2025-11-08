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
    y = X@W.T  
    noised = y + ch.sqrt(NOISE_VAR) * ch.randn(y.size(0), 1)
    # generate ground-truth data
    phi = oracle.Left_Regression(ch.zeros(1))
    # truncate
    indices = phi(noised).nonzero()[:,0]
    x_trunc, y_trunc = X[indices], noised[indices]
    alpha = x_trunc.size(0) / X.size(0)
    print(f'alpha: {alpha}')

    gt_norm = LinearRegression(fit_intercept=False)
    gt_norm.fit(X, y)
    gt_ = ch.from_numpy(np.concatenate([gt_norm.coef_.flatten()]))

    # calculate empirical noise variance for regression 
    ols_trunc = LinearRegression(fit_intercept=False)
    ols_trunc.fit(x_trunc, y_trunc)
    emp_ = ch.from_numpy(np.concatenate([ols_trunc.coef_.flatten()]))
    print(f'empirical weights: {emp_}')
    emp_mse_loss = mse_loss(emp_, gt_)
    print(f'emp mse loss: {emp_mse_loss}')

    # scale y features
    y_trunc_scale = y_trunc / ch.sqrt(NOISE_VAR)
    phi_scale = oracle.Left_Regression(phi.left / ch.sqrt(NOISE_VAR))
    # train algorithm
    train_kwargs = Parameters({'epochs': 2,
                                'lr': 1e-1,
                                'batch_size': 10,
                                'trials': 1,
                                'verbose': True
                            }) 
    trunc_reg = stats.TruncatedLinearRegression(train_kwargs,
                                                phi_scale, 
                                                alpha, 
                                                fit_intercept=False,
                                                noise_var=ch.ones(1, 1))
    trunc_reg.fit(x_trunc, y_trunc_scale)
    w_ = ch.cat([(trunc_reg.best_coef_).flatten()]) * ch.sqrt(NOISE_VAR)
    print(f'estimated weights: {w_}')
    trunc_mse_loss = mse_loss(gt_, w_.flatten())
    print(f'truc mse loss: {trunc_mse_loss}')
    msg = f'trunc mse loss is larger than empirical mse loss. known mse loss is {trunc_mse_loss}, and empirical mse loss is: {emp_mse_loss}'
    assert trunc_mse_loss <= emp_mse_loss, msg
    msg = f'trunc mse loss: {trunc_mse_loss}, which is larger than 1e-1'
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
    y = X@W.T + W0 
    noised = y + ch.sqrt(NOISE_VAR) * ch.randn(y.size(0), 1)
    # generate ground-truth data
    phi = oracle.Left_Regression(ch.ones(1))
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
    train_kwargs = Parameters({'epochs': 3,
                                'lr': 1e-1,
                                'batch_size': 10,
                                'trials': 1,
                                'verbose': True
                            }) 
    trunc_reg = stats.TruncatedLinearRegression(train_kwargs,
                                                phi_scale, 
                                                alpha, 
                                                fit_intercept=True,
                                                noise_var=ch.ones(1, 1))
    trunc_reg.fit(x_trunc, y_trunc_scale)
    w_ = ch.cat([(trunc_reg.best_coef_).flatten(), trunc_reg.best_intercept_]) * ch.sqrt(NOISE_VAR)
    print(f'estimated weights: {w_}')
    trunc_mse_loss = mse_loss(gt_, w_.flatten())
    print(f'truc mse loss: {trunc_mse_loss}')
    msg = f'trunc mse loss is larger than empirical mse loss. known mse loss is {trunc_mse_loss}, and empirical mse loss is: {emp_mse_loss}'
    assert trunc_mse_loss <= emp_mse_loss, msg
    msg = f'trunc mse loss: {trunc_mse_loss}, which is larger than 1e-1'
    assert trunc_mse_loss <= 1e-1, msg

# left truncated linear regression with known variance - 20 dimensions
def test_known_truncated_regression_higher_dimensions():
    D = 20
    NUM_SAMPLES = 10000
    # generate ground truth
    NOISE_VAR = 3*ch.ones(1, 1)
    W = ch.ones(1, D)
    W0 = ch.ones(1, 1)
    print(f"gt weight: {W}")
    print(f"gt intercept: {W0}")
    print(f"gt noise var: {NOISE_VAR}")

    # generate data
    X = ch.rand(NUM_SAMPLES, D)
    y = X@W.T + W0 
    noised = y + ch.sqrt(NOISE_VAR) * ch.randn(y.size(0), 1)
    # generate ground-truth data
    phi = oracle.Left_Regression(10*ch.ones(1))
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
    train_kwargs = Parameters({'epochs': 5,
                                'lr': 1e-1,
                                'batch_size': 10,
                                'trials': 1,
                                'verbose': True
                            }) 
    trunc_reg = stats.TruncatedLinearRegression(train_kwargs,
                                                phi_scale, 
                                                alpha, 
                                                noise_var=ch.ones(1, 1))
    trunc_reg.fit(x_trunc, y_trunc_scale)
    w_ = ch.cat([(trunc_reg.best_coef_).flatten(), trunc_reg.best_intercept_]) * ch.sqrt(NOISE_VAR)
    print(f'estimated weights: {w_}')
    known_mse_loss = mse_loss(gt_, w_.flatten())
    print(f'known mse loss: {known_mse_loss}')
    msg = f'known mse loss is larger than empirical mse loss. known mse loss is {known_mse_loss}, and empirical mse loss is: {emp_mse_loss}'
    assert known_mse_loss <= emp_mse_loss, msg
    msg = f'known mse loss is larger than 1e-1. known mse loss is {known_mse_loss.item():.3f}'
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
    y = X@W.T
    noised = y + ch.sqrt(noise_var) * ch.randn(y.size(0), 1)
    # generate ground-truth data
    phi = oracle.Left_Regression(ch.zeros(1))
    # phi = oracle.Identity()
    # truncate
    indices = phi(noised).nonzero()[:,0]
    x_trunc, y_trunc = X[indices], noised[indices]
    alpha = x_trunc.size(0) / X.size(0)
    print(f'alpha: {alpha}')

    gt_norm = LinearRegression(fit_intercept=False)
    gt_norm.fit(X, noised)
    gt_ = ch.from_numpy(gt_norm.coef_)

    # calculate empirical noise variance for regression 
    ols_trunc = LinearRegression(fit_intercept=False)
    ols_trunc.fit(x_trunc, y_trunc)
    emp_noise_var = ch.from_numpy(ols_trunc.predict(x_trunc) - y_trunc.numpy()).var(0)
    emp_ = ch.from_numpy(ols_trunc.coef_.flatten())
    print(f'emp weight estimates: {emp_.tolist()}')
    print(f'emp noise estimate: {emp_noise_var.item()}')
    emp_mse_loss = mse_loss(emp_, gt_)
    emp_var_l1 = float(ch.abs(emp_noise_var - noise_var))
    print(f'emp mse loss: {emp_mse_loss}')
    print(f'emp noise var l1: {emp_var_l1}')

    # scale y features by empirical noise variance
    y_trunc_emp_scale = y_trunc / ch.sqrt(emp_noise_var)
    # y_trunc_emp_scale = y_trunc
    phi_emp_scale = oracle.Left_Regression(phi.left / ch.sqrt(emp_noise_var))
    # train algorithm
    train_kwargs = Parameters({
                                'trials': 1,
                                'epochs': 2,
                                'batch_size': 10,
                                'var_lr': 1e-2, 
                                'verbose': True,
                                # 'step_lr': 500,
                                'step_lr_gamma': 1.0,
                            })
    unknown_trunc_reg = stats.TruncatedLinearRegression(train_kwargs,
                                                        phi_emp_scale,
                                                        alpha,
                                                        fit_intercept=False)
    unknown_trunc_reg.fit(x_trunc, y_trunc_emp_scale)
    w_ = unknown_trunc_reg.best_coef_.flatten() * ch.sqrt(emp_noise_var)
    noise_var_ = unknown_trunc_reg.best_variance_  * emp_noise_var
    print(f'estimated_weights: {w_.tolist()}')
    print(f'estimated noise variance: {noise_var_.item()}')
    unknown_mse_loss = mse_loss(gt_, w_.flatten())
    print(f'unknown mse loss: {unknown_mse_loss}')
    unknown_var_l1 = float(ch.abs(noise_var - noise_var_))
    print(f'unknown var l1: {unknown_var_l1}')
    assert unknown_mse_loss <= 1e-1, f'unknown mse loss: {unknown_mse_loss} is larger than: 1e-1'
    assert unknown_var_l1 <= 1e-1, f'unknown var l1: {unknown_var_l1} is larger than 1e-1'

# left truncated regression with unknown noise variance in one dimension 
def test_unknown_variance_truncated_regression_one_dimension():
    D  = 1
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
    y = X@W.T + W0 
    noised = y + ch.sqrt(noise_var) * ch.randn(y.size(0), 1)
    # generate ground-truth data
    phi = oracle.Left_Regression(ch.ones(1))
    # phi = oracle.Identity()
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
    emp_noise_var = ch.from_numpy(ols_trunc.predict(x_trunc) - y_trunc.numpy()).var(0)
    emp_ = ch.from_numpy(np.concatenate([ols_trunc.coef_.flatten(), ols_trunc.intercept_]))
    print(f'emp weight estimates: {emp_.tolist()}')
    print(f'emp noise estimate: {emp_noise_var.item()}')
    emp_mse_loss = mse_loss(emp_, gt_)
    emp_var_l1 = float(ch.abs(emp_noise_var - noise_var))
    print(f'emp mse loss: {emp_mse_loss}')
    print(f'emp noise var l1: {emp_var_l1}')

    # scale y features by empirical noise variance
    y_trunc_emp_scale = (y_trunc - ols_trunc.intercept_) / ch.sqrt(emp_noise_var)
    # y_trunc_emp_scale = y_trunc
    phi_emp_scale = oracle.Left_Regression((phi.left - ols_trunc.intercept_) / ch.sqrt(emp_noise_var))
    # train algorithm
    train_kwargs = Parameters({
                                'trials': 1,
                                'epochs': 10,
                                'batch_size': 10,
                                'var_lr': 1e-1, 
                                'verbose': True,
                            })
    unknown_trunc_reg = stats.TruncatedLinearRegression(train_kwargs,
                                                        phi_emp_scale,
                                                        alpha)
    unknown_trunc_reg.fit(x_trunc, y_trunc_emp_scale)
    # unknown_trunc_reg.fit(x_trunc.repeat(1, 1), y_trunc_emp_scale.repeat(1, 1))
    w_ = ch.cat([(unknown_trunc_reg.best_coef_).flatten(), unknown_trunc_reg.best_intercept_ + ols_trunc.intercept_]) * ch.sqrt(emp_noise_var)
    noise_var_ = unknown_trunc_reg.best_variance_ * emp_noise_var
    print(f'estimated_weights: {w_.tolist()}')
    print(f'estimated noise variance: {noise_var_.item()}')
    unknown_mse_loss = mse_loss(gt_, w_.flatten())
    print(f'unknown mse loss: {unknown_mse_loss}')
    unknown_var_l1 = float(ch.abs(noise_var - noise_var_))
    print(f'unknown var l1: {unknown_var_l1}')
    assert unknown_mse_loss <= emp_mse_loss, f'unknown mse loss: {unknown_mse_loss}'
    assert unknown_var_l1 <= emp_var_l1, f'unknown var l1: {unknown_var_l1}'


# left truncated regression with unknown noise variance in ten dimensions 
def test_unknown_variance_truncated_regression_ten_dimensions():
    D = 10
    NUM_SAMPLES = 20000
    # generate ground truth
    noise_var = ch.ones(1, 1)
    W = ch.ones(1, D)
    W0 = ch.ones(1, 1) 

    print(f"gt weights: {ch.cat([W, W0], dim=1).tolist()}")
    print(f"gt noise var: {noise_var.item()}")
    # generate data
    X = ch.rand(NUM_SAMPLES, D) 
    y = X@W.T + W0 
    noised = y + ch.sqrt(noise_var) * ch.randn(y.size(0), 1)
    # generate ground-truth data
    phi = oracle.Left_Regression(5*ch.ones(1))
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
    emp_noise_var = ch.from_numpy(ols_trunc.predict(x_trunc) - y_trunc.numpy()).var(0)
    emp_ = ch.from_numpy(np.concatenate([ols_trunc.coef_.flatten(), ols_trunc.intercept_]))
    print(f'emp weight estimates: {emp_.tolist()}')
    print(f'emp noise estimate: {emp_noise_var.item()}')
    emp_mse_loss = mse_loss(emp_, gt_)
    emp_var_l1 = float(ch.abs(emp_noise_var - noise_var))
    print(f'emp mse loss: {emp_mse_loss}')
    print(f'emp noise var l1: {emp_var_l1}')
    # scale y features by empirical noise variance
    y_trunc_emp_scale = (y_trunc - ols_trunc.intercept_) / ch.sqrt(emp_noise_var)
    phi_emp_scale = oracle.Left_Regression((phi.left - ols_trunc.intercept_) / ch.sqrt(emp_noise_var))
    # train algorithm
    train_kwargs = Parameters({
                                'trials': 1,
                                'epochs': 10,
                                'batch_size': 10,
                                'var_lr': 1e-2, 
                                'verbose': True,
                            })
    unknown_trunc_reg = stats.TruncatedLinearRegression(train_kwargs,
                                                        phi_emp_scale,
                                                        alpha)
    unknown_trunc_reg.fit(x_trunc, y_trunc_emp_scale)
    # unknown_trunc_reg.fit(x_trunc.repeat(1, 1), y_trunc_emp_scale.repeat(1, 1))
    w_ = ch.cat([(unknown_trunc_reg.best_coef_).flatten(), unknown_trunc_reg.best_intercept_ + ols_trunc.intercept_]) * ch.sqrt(emp_noise_var)
    noise_var_ = unknown_trunc_reg.best_variance_ * emp_noise_var
    print(f'estimated_weights: {w_.tolist()}')
    print(f'estimated noise variance: {noise_var_.item()}')
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
        'c_gamma': 2.0,
    })
    trunc_lds = stats.TruncatedLinearRegression(phi,
                                                train_kwargs,
                                                noise_var=NOISE_VAR, 
                                                dependent=True, 
                                                rand_seed=seed)
    trunc_lds.fit(X, Y)
    A_ = trunc_lds.coef_
    A0_ = trunc_lds.ols_coef_
    trunc_spec_norm = calc_spectral_norm(A - A_)
    emp_spec_norm = calc_spectral_norm(A - A0_)

    print(f'alpha: {alpha}')
    print(f'A spectral norm: {spectral_norm}')
    print(f'truncated spectral norm: {trunc_spec_norm}')
    print(f'ols spectral norm: {emp_spec_norm}')

    assert trunc_spec_norm <= emp_spec_norm, f"truncated spectral norm {trunc_spec_norm}, while OLS spectral norm is: {emp_spec_norm}"


def test_gradient_contours_1d_regression():
    from delphi.grad import TruncatedUnknownVarianceMSE 
    """Test loss landscape and gradients for 1D truncated regression"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Simple 1D case for clear visualization
    D, SAMPLES = 1, 1000
    true_w = ch.tensor([[2.0]])  # True weight
    true_w0 = ch.tensor([[1.0]])  # True intercept
    NOISE_VAR = ch.ones(1, 1)
    
    # Generate data
    M = Uniform(-5, 5)
    X = M.sample([SAMPLES, D])
    y = X @ true_w.T + true_w0
    noised = y + ch.sqrt(NOISE_VAR) * ch.randn(y.size(0), 1)
    
    # Truncate (left truncation at 0)
    phi = oracle.Left_Regression(ch.zeros(1))
    indices = phi(noised).nonzero()[:,0]
    x_trunc, y_trunc = X[indices], noised[indices]
    
    print(f"True weights: w={true_w.item()}, w0={true_w0.item()}")
    print(f"Truncated samples: {x_trunc.shape[0]}/{SAMPLES}")
    print(f"Truncated data range: x=[{x_trunc.min():.2f}, {x_trunc.max():.2f}], y=[{y_trunc.min():.2f}, {y_trunc.max():.2f}]")
    
    # Test different weight values
    w_range = np.linspace(0.0, 4.0, 30)  # Around true w=2.0
    w0_range = np.linspace(-1.0, 3.0, 30)  # Around true w0=1.0
    
    losses = np.zeros((len(w_range), len(w0_range)))
    grad_w = np.zeros((len(w_range), len(w0_range)))
    grad_w0 = np.zeros((len(w_range), len(w0_range)))
    
    phi_scale = oracle.Left_Regression(phi.left / ch.sqrt(NOISE_VAR))
    y_trunc_scale = y_trunc / ch.sqrt(NOISE_VAR)
    
    for i, w_val in enumerate(w_range):
        for j, w0_val in enumerate(w0_range):
            # Create parameters with requires_grad
            w = ch.tensor([[w_val]]).float().detach()
            w0 = ch.tensor([[w0_val]]).float().detach()

            w.requires_grad, w0.requires_grad = True, True
            # Compute predictions
            pred = x_trunc @ w.T + w0
            
            # Compute loss using your TruncatedMSE function
            loss = TruncatedUnknownVarianceMSE.apply(
                pred, y_trunc_scale, NOISE_VAR.inverse(), phi_scale, 
                50
            )
            
            losses[i, j] = loss.item()
            
            # Compute gradients
            loss.backward()
            grad_w[i, j] = w.grad.item() * NOISE_VAR if w.grad is not None else 0.0
            grad_w0[i, j] = w0.grad.item() * NOISE_VAR if w0.grad is not None else 0.0
            
            # Zero gradients for next iteration
            w.grad = None
            w0.grad = None

    # test_ws = [0.5, 2.0, 3.5]
    
    # for w_val in test_ws:
    #     w = ch.tensor([[w_val]], requires_grad=True)
    #     pred = x_trunc @ w.T + w0
        
    #     # Compute each term separately
    #     quadratic_term = -0.5 * (y_trunc - pred).pow(2).mean()
        
    #     # Your 'out' term approximation
    #     stacked = pred[None, ...].repeat(50, 1, 1)
    #     import math
    #     noised = stacked + math.sqrt(NOISE_VAR) * ch.randn(stacked.size())        
    #     filtered = phi(noised)
    #     z = (filtered * noised).sum(dim=0) / (filtered.sum(dim=0) + 1e-3)
    #     out_term = -0.5 * (z.pow(2) + z * pred).mean()
        
    #     total_loss = quadratic_term + out_term
        
    #     print(f"w={w_val}: quadratic={quadratic_term.item():.3f}, out_term={out_term.item():.3f}, total={total_loss.item():.3f}")

    #     test_ws = [0.5, 2.0, 3.5]
    
    # for sign in [1, -1]:
    #     print(f"\n=== Testing out_term sign: {sign} ===")
        
    #     for w_val in test_ws:
    #         w = ch.tensor([[w_val]], requires_grad=True)
    #         pred = x_trunc @ w.T + w0
            
    #         quadratic_term = -0.5 * (y_trunc - pred).pow(2).mean()
            
    #         # Your out_term with sign parameter
    #         stacked = pred[None, ...].repeat(50, 1, 1)
    #         noised = stacked + math.sqrt(NOISE_VAR) * ch.randn(stacked.size())        
    #         filtered = phi(noised)
    #         z = (filtered * noised).sum(dim=0) / (filtered.sum(dim=0) + 1e-3)
    #         out_term = sign * 0.5 * (z.pow(2) + z * pred).mean()
            
    #         total_loss = quadratic_term + out_term
            
    #         print(f"w={w_val}: total={total_loss.item():.3f}")

    def compute_log_truncation_prob(pred, phi, noise_var, num_samples=100):
        """
        Properly compute log P(z ∈ S | pred) using Monte Carlo
        """
        # Sample from conditional distribution p(z | pred)
        stacked = pred[None, ...].repeat(num_samples, 1, num_samples)
        z_samples = stacked + math.sqrt(noise_var) * ch.randn(stacked.size())
    
        # Compute the integrand: exp(-1/2 (z - pred)²)
        integrand = ch.exp(-0.5 * (z_samples - pred).pow(2))
    
        # Mask with truncation set
        in_set = phi(z_samples)
        masked_integrand = integrand * in_set
    
        # Monte Carlo estimate of the integral
        integral_estimate = masked_integrand.mean(dim=0)
    
        # log(integral)
        eps = 1e-10
        log_integral = ch.log(integral_estimate + eps)
    
        return log_integral
    
    # test_ws = [0.5, 2.0, 3.5]
    
    # for w_val in test_ws:
    #     w = ch.tensor([[w_val]], requires_grad=True)
    #     pred = x_trunc @ w.T + w0
        
    #     quadratic_term = -0.5 * (y_trunc - pred).pow(2).mean()
    #     log_trunc_prob = compute_log_truncation_prob(pred, phi, NOISE_VAR)
    #     out_term = -log_trunc_prob.mean()
        
    #     total_loss = quadratic_term + out_term
        
    #     print(f"w={w_val}: quadratic={quadratic_term.item():.3f}, "
    #           f"out_term={out_term.item():.3f}, total={total_loss.item():.3f}")

    print("=== GRADIENT ANALYSIS ===")
    print(f"True parameters: w={true_w.item():.3f}, w0={true_w0.item():.3f}")
    
    # Find minimum
    min_idx = np.unravel_index(np.argmin(losses), losses.shape)
    min_w, min_w0 = w_range[min_idx[0]], w0_range[min_idx[1]]
    print(f"Minimum loss at: w={min_w:.3f}, w0={min_w0:.3f}")
    
    # Check gradients at several test points
    test_points = [
        (0.5, 0.0, "Too small"),
        (2.0, 1.0, "True params"), 
        (3.5, 2.0, "Too large")
    ]
    
    for w_val, w0_val, desc in test_points:
        i = np.argmin(np.abs(w_range - w_val))
        j = np.argmin(np.abs(w0_range - w0_val))
        grad_w_val = grad_w[i, j]
        grad_w0_val = grad_w0[i, j]
        
        print(f"\nAt w={w_val:.1f}, w0={w0_val:.1f} ({desc}):")
        print(f"  Loss: {losses[i, j]:.4f}")
        print(f"  Gradient dw: {grad_w_val:.4f} (should be {'POSITIVE' if w_val < true_w.item() else 'NEGATIVE' if w_val > true_w.item() else 'ZERO'})")
        print(f"  Gradient dw0: {grad_w0_val:.4f} (should be {'POSITIVE' if w0_val < true_w0.item() else 'NEGATIVE' if w0_val > true_w0.item() else 'ZERO'})")
    
    # Check convexity around optimum
    opt_i = min_idx[0]
    opt_j = min_idx[1]
    print(f"\nConvexity check around optimum:")
    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        if 0 <= opt_i + di < len(w_range) and 0 <= opt_j + dj < len(w0_range):
            neighbor_loss = losses[opt_i + di, opt_j + dj]
            print(f"  Loss increases to {neighbor_loss:.4f} when moving from optimum")



    # test regression 


    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss contour
    W, W0 = np.meshgrid(w0_range, w_range)
    contour1 = axes[0, 0].contourf(W, W0, losses, levels=50, cmap='viridis')
    axes[0, 0].contour(W, W0, losses, levels=10, colors='black', alpha=0.3)
    axes[0, 0].scatter([true_w0.item()], [true_w.item()], color='red', marker='*', s=200, label='True params')
    axes[0, 0].set_xlabel('Intercept (w0)')
    axes[0, 0].set_ylabel('Weight (w)')
    axes[0, 0].set_title('Loss Landscape')
    axes[0, 0].legend()
    plt.colorbar(contour1, ax=axes[0, 0])
    
    # Gradient field
    axes[0, 1].quiver(W, W0, grad_w0, grad_w, scale=50, color='blue', alpha=0.6)
    axes[0, 1].scatter([true_w0.item()], [true_w.item()], color='red', marker='*', s=200, label='True params')
    axes[0, 1].set_xlabel('Intercept (w0)')
    axes[0, 1].set_ylabel('Weight (w)')
    axes[0, 1].set_title('Gradient Field')
    axes[0, 1].legend()
    
    # 1D slices through true parameters
    # Fix w0 at true value, vary w
    w0_fixed = true_w0.item()
    j_fixed = np.argmin(np.abs(w0_range - w0_fixed))
    axes[1, 0].plot(w_range, losses[:, j_fixed])
    axes[1, 0].axvline(x=true_w.item(), color='red', linestyle='--', label='True w')
    axes[1, 0].set_xlabel('Weight (w)')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title(f'Loss vs w (w0 fixed at {w0_fixed})')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Fix w at true value, vary w0
    w_fixed = true_w.item()
    i_fixed = np.argmin(np.abs(w_range - w_fixed))
    axes[1, 1].plot(w0_range, losses[i_fixed, :])
    axes[1, 1].axvline(x=true_w0.item(), color='red', linestyle='--', label='True w0')
    axes[1, 1].set_xlabel('Intercept (w0)')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title(f'Loss vs w0 (w fixed at {w_fixed})')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print diagnostic information
    min_idx = np.unravel_index(np.argmin(losses), losses.shape)
    min_w, min_w0 = w_range[min_idx[0]], w0_range[min_idx[1]]
    print(f"Minimum loss at: w={min_w:.3f}, w0={min_w0:.3f}")
    print(f"True parameters: w={true_w.item():.3f}, w0={true_w0.item():.3f}")
    print(f"Distance from optimum: {np.sqrt((min_w - true_w.item())**2 + (min_w0 - true_w0.item())**2):.3f}")
    
    # Check gradient behavior around optimum
    opt_i = min_idx[0]
    opt_j = min_idx[1]
    print(f"Gradient at optimum: dw={grad_w[opt_i, opt_j]:.6f}, dw0={grad_w0[opt_i, opt_j]:.6f}")
    print(f"Gradient should be near zero at optimum")

# Also add a simpler 1D version for quick testing
def test_simple_gradient_direction():
    from delphi.grad import TruncatedMSE
    """Simple test to check if gradients point in right direction"""
    # Simple case: true line y = 2x + 1, truncated at y > 0
    true_w, true_w0 = 2.0, 1.0
    X = ch.tensor([[1.0], [2.0], [3.0]])
    y = X * true_w + true_w0
    
    phi = oracle.Left_Regression(ch.zeros(1))
    
    # Test at wrong parameters
    test_w, test_w0 = 1.0, 0.0  # Too small
    
    w = ch.tensor([[test_w]], requires_grad=True)
    w0 = ch.tensor([[test_w0]], requires_grad=True)
    
    pred = X @ w.T + w0
    loss = TruncatedMSE.apply(pred, y, phi, 1.0, num_samples=50)
    loss.backward()
    
    print(f"Test at w={test_w}, w0={test_w0}")
    print(f"Gradient dw={w.grad.item():.4f}, should be POSITIVE to increase w")
    print(f"Gradient dw0={w0.grad.item():.4f}, should be POSITIVE to increase w0")


def test_3d_gradient_contours():
    """Test loss landscape and gradients for weights, intercept, and variance"""
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D
    from delphi.grad import TruncatedUnknownVarianceMSE
    
    # Set random seed for reproducibility
    ch.manual_seed(42)
    np.random.seed(42)
    
    # True parameters
    true_w = 2.0
    true_w0 = 1.0
    true_var = ch.Tensor([1.0])  # True variance
    true_lambda = 1.0 / true_var  # λ = 1/σ²
    
    # Generate data
    SAMPLES = 2000
    M = Uniform(-5, 5)
    X = M.sample([SAMPLES, 1])
    y = X * true_w + true_w0
    noised = y + ch.sqrt(true_var) * ch.randn(y.size(0), 1)
    
    # Truncate (left truncation at 0)
    phi = oracle.Left_Regression(ch.zeros(1))
    indices = phi(noised).nonzero()[:,0]
    x_trunc, y_trunc = X[indices], noised[indices]
    
    print(f"=== 3D GRADIENT CONTOURS ===")
    print(f"True: w={true_w}, w0={true_w0}, var={true_var}, λ={true_lambda}")
    print(f"Truncated: {x_trunc.shape[0]}/{SAMPLES} samples")
    
    # Test ranges
    w_range = np.linspace(0.5, 3.5, 8)      # Around true_w=2.0
    w0_range = np.linspace(-0.5, 2.5, 8)    # Around true_w0=1.0  
    lambda_range = np.linspace(0.3, 3.0, 10) # Around true_lambda=1.0 (var=0.33 to 1.0)
    
    # Storage for results
    losses = np.zeros((len(w_range), len(w0_range), len(lambda_range)))
    grad_w = np.zeros((len(w_range), len(w0_range), len(lambda_range)))
    grad_w0 = np.zeros((len(w_range), len(w0_range), len(lambda_range))) 
    grad_lambda = np.zeros((len(w_range), len(w0_range), len(lambda_range)))
    
    for i, w_val in enumerate(w_range):
        for j, w0_val in enumerate(w0_range):
            for k, lambda_val in enumerate(lambda_range):
                # Create parameters
                w = ch.tensor([[w_val]], requires_grad=True)
                w0 = ch.tensor([[w0_val]], requires_grad=True) 
                lambda_param = ch.tensor([lambda_val], requires_grad=True)
                
                # Compute predictions (with your reparameterization)
                pred = x_trunc @ w.T.float() + w0.float()
                
                # Compute loss
                loss = TruncatedUnknownVarianceMSE.apply(
                    pred, y_trunc, lambda_param, phi, 100
                )

                losses[i, j, k] = loss.item()
                
                # Compute gradients
                loss.backward()
                
                grad_w[i, j, k] = w.grad.item() if w.grad is not None else 0.0
                grad_w0[i, j, k] = w0.grad.item() if w0.grad is not None else 0.0
                grad_lambda[i, j, k] = lambda_param.grad.item() if lambda_param.grad is not None else 0.0

                
                # Zero gradients
                w.grad = None
                w0.grad = None
                lambda_param.grad = None

    print(f'grad lambda: {grad_lambda}')
    
    # Analyze results
    print(f"\n=== GRADIENT ANALYSIS ===")
    
    # Find global minimum
    min_idx = np.unravel_index(np.argmin(losses), losses.shape)
    min_w, min_w0, min_lambda = w_range[min_idx[0]], w0_range[min_idx[1]], lambda_range[min_idx[2]]
    min_var = 1.0 / min_lambda
    
    print(f"Global minimum at: w={min_w:.3f}, w0={min_w0:.3f}, λ={min_lambda:.3f} (var={min_var:.3f})")
    print(f"True parameters:   w={true_w:.3f}, w0={true_w0:.3f}, λ={true_lambda.item():.3f} (var={true_var.item():.3f})")
    print(f"Distance from optimum: {np.sqrt((min_w-true_w)**2 + (min_w0-true_w0)**2 + (min_lambda-true_lambda).item()**2):.3f}")
    
    # Check gradients at key points
    test_points = [
        (1.0, 0.0, 0.5, "All too small"),
        (2.0, 1.0, 1.0, "True parameters"),
        (3.0, 2.0, 2.0, "All too large")
    ]
    
    for w_val, w0_val, lambda_val, desc in test_points:
        i = np.argmin(np.abs(w_range - w_val))
        j = np.argmin(np.abs(w0_range - w0_val)) 
        k = np.argmin(np.abs(lambda_range - lambda_val))
        
        print(f"\nAt {desc}:")
        print(f"  w={w_val:.1f}, w0={w0_val:.1f}, λ={lambda_val:.1f} (var={1.0/lambda_val:.2f})")
        print(f"  Loss: {losses[i, j, k]:.4f}")
        print(f"  Grad w: {grad_w[i, j, k]:.4f} (should be {'POSITIVE' if w_val < true_w else 'NEGATIVE' if w_val > true_w else 'ZERO'})")
        print(f"  Grad w0: {grad_w0[i, j, k]:.4f} (should be {'POSITIVE' if w0_val < true_w0 else 'NEGATIVE' if w0_val > true_w0 else 'ZERO'})")
        print(f"  Grad λ: {grad_lambda[i, j, k]:.4f} (should be {'POSITIVE' if lambda_val < true_lambda else 'NEGATIVE' if lambda_val > true_lambda else 'ZERO'})")
    
    # Create 2D slice plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Fix lambda at true value, vary w and w0
    lambda_idx = np.argmin(np.abs(lambda_range - true_lambda.item()))
    W, W0 = np.meshgrid(w0_range, w_range)
    
    contour1 = axes[0, 0].contourf(W, W0, losses[:, :, lambda_idx], levels=20, cmap='viridis')
    axes[0, 0].contour(W, W0, losses[:, :, lambda_idx], levels=10, colors='black', alpha=0.3)
    axes[0, 0].scatter([true_w0], [true_w], color='red', marker='*', s=200, label='True params')
    axes[0, 0].set_xlabel('Intercept (w0)')
    axes[0, 0].set_ylabel('Weight (w)')
    axes[0, 0].set_title(f'Loss (λ fixed at {true_lambda.item():.1f})')
    axes[0, 0].legend()
    plt.colorbar(contour1, ax=axes[0, 0])
    
    # Fix w0 at true value, vary w and lambda
    w0_idx = np.argmin(np.abs(w0_range - true_w0))
    W, L = np.meshgrid(lambda_range, w_range)
    
    contour2 = axes[0, 1].contourf(W, L, losses[:, w0_idx, :], levels=20, cmap='viridis')
    axes[0, 1].contour(W, L, losses[:, w0_idx, :], levels=10, colors='black', alpha=0.3)
    axes[0, 1].scatter([true_lambda.item()], [true_w], color='red', marker='*', s=200, label='True params')
    axes[0, 1].set_xlabel('Lambda (λ)')
    axes[0, 1].set_ylabel('Weight (w)')
    axes[0, 1].set_title(f'Loss (w0 fixed at {true_w0:.1f})')
    axes[0, 1].legend()
    plt.colorbar(contour2, ax=axes[0, 1])
    
    # Fix w at true value, vary w0 and lambda
    w_idx = np.argmin(np.abs(w_range - true_w))
    W0, L = np.meshgrid(lambda_range, w0_range)
    
    contour3 = axes[0, 2].contourf(W0, L, losses[w_idx, :, :], levels=20, cmap='viridis')
    axes[0, 2].contour(W0, L, losses[w_idx, :, :], levels=10, colors='black', alpha=0.3)
    axes[0, 2].scatter([true_lambda.item()], [true_w0], color='red', marker='*', s=200, label='True params')
    axes[0, 2].set_xlabel('Lambda (λ)')
    axes[0, 2].set_ylabel('Intercept (w0)')
    axes[0, 2].set_title(f'Loss (w fixed at {true_w:.1f})')
    axes[0, 2].legend()
    plt.colorbar(contour3, ax=axes[0, 2])
    
    # Gradient magnitude analysis
    grad_magnitudes = np.sqrt(grad_w**2 + grad_w0**2 + grad_lambda**2)
    
    # Plot maximum gradient magnitude per parameter
    max_grad_w = np.max(np.abs(grad_w), axis=(1, 2))
    max_grad_w0 = np.max(np.abs(grad_w0), axis=(0, 2))
    max_grad_lambda = np.max(np.abs(grad_lambda), axis=(0, 1))
    
    axes[1, 0].plot(w_range, max_grad_w)
    axes[1, 0].axvline(x=true_w, color='red', linestyle='--', label='True w')
    axes[1, 0].set_xlabel('Weight (w)')
    axes[1, 0].set_ylabel('Max |Gradient|')
    axes[1, 0].set_title('Max Weight Gradient vs w')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(w0_range, max_grad_w0)
    axes[1, 1].axvline(x=true_w0, color='red', linestyle='--', label='True w0')
    axes[1, 1].set_xlabel('Intercept (w0)')
    axes[1, 1].set_ylabel('Max |Gradient|')
    axes[1, 1].set_title('Max Intercept Gradient vs w0')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    axes[1, 2].plot(lambda_range, max_grad_lambda)
    axes[1, 2].axvline(x=true_lambda.item(), color='red', linestyle='--', label='True λ')
    axes[1, 2].set_xlabel('Lambda (λ)')
    axes[1, 2].set_ylabel('Max |Gradient|')
    axes[1, 2].set_title('Max Lambda Gradient vs λ')
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Stability analysis
    print(f"\n=== STABILITY ANALYSIS ===")
    print(f"Max gradient magnitude: {np.max(grad_magnitudes):.4f}")
    print(f"Mean gradient magnitude: {np.mean(grad_magnitudes):.4f}")
    print(f"Gradient magnitude at true params: {grad_magnitudes[min_idx]:.4f}")
    
    # Check if gradients are reasonable
    if np.max(grad_magnitudes) > 1000:
        print("⚠️  WARNING: Very large gradients detected - potential instability!")
    elif np.max(grad_magnitudes) < 0.001:
        print("⚠️  WARNING: Very small gradients detected - potential optimization issues!")
    else:
        print("✓ Gradient magnitudes look reasonable")

def test_debug_lambda_gradients():
    """Debug the lambda gradients specifically"""
    import torch
    from delphi.grad import TruncatedUnknownVarianceMSE
    
    # Use a simpler setup with just a few points
    ch.manual_seed(42)
    
    # True parameters
    true_w = 2.0
    true_w0 = 1.0
    true_var = ch.Tensor([1.0])
    true_lambda = 1.0 / true_var
    
    # Generate smaller dataset for debugging
    SAMPLES = 100
    X = Uniform(-5, 5).sample([SAMPLES, 1])
    y = X * true_w + true_w0
    noised = y + ch.sqrt(true_var) * ch.randn(y.size(0), 1)
    
    # Truncate
    phi = oracle.Left_Regression(ch.zeros(1))
    indices = phi(noised).nonzero()[:,0]
    x_trunc, y_trunc = X[indices], noised[indices]
    
    print(f"Debug: {x_trunc.shape[0]}/{SAMPLES} truncated samples")
    
    # Test specific lambda values around the true value
    lambda_test_vals = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
    
    for lambda_val in lambda_test_vals:
        w = ch.tensor([[2.0]], requires_grad=True)  # Fix at true w
        w0 = ch.tensor([[1.0]], requires_grad=True)  # Fix at true w0
        lambda_param = ch.tensor([lambda_val], requires_grad=True)
        
        pred = x_trunc @ w.T.float() + w0.float()
        
        # Compute loss with detailed debugging
        loss = TruncatedUnknownVarianceMSE.apply(
            pred, y_trunc, lambda_param, phi, 100
        )
        
        loss.backward()
        
        print(f"λ={lambda_val:.2f}: loss={loss.item():.4f}, grad_λ={lambda_param.grad.item():.4f}")
        
        # Expected: grad_λ should be positive when λ < true_λ (1.0), negative when λ > true_λ
        expected_sign = "positive" if lambda_val < 1.0 else "negative" if lambda_val > 1.0 else "zero"
        actual_sign = "positive" if lambda_param.grad.item() > 0 else "negative" if lambda_param.grad.item() < 0 else "zero"
        
        print(f"  Expected sign: {expected_sign}, Actual: {actual_sign}")
        
        # Reset gradients
        w.grad = None
        w0.grad = None
        lambda_param.grad = None

# Also add this to check the reparameterization:
def check_reparameterization():
    """Verify the λ = 1/σ² relationship is handled correctly"""
    print("\n=== REPARAMETERIZATION CHECK ===")
    
    # Test if the loss function correctly handles the λ parameterization
    test_vars = [0.5, 1.0, 2.0]  # σ² values
    test_lambdas = [1/v for v in test_vars]  # λ = 1/σ²
    
    for var, lam in zip(test_vars, test_lambdas):
        print(f"σ²={var:.2f}, λ={lam:.2f}, relationship: λ = 1/σ² = {1/var:.2f}")


def test_gradient_consistency():
    """Test if gradients are consistent across different sample sizes"""
    import torch
    
    ch.manual_seed(42)
    
    true_w = 2.0
    true_w0 = 1.0  
    true_var = 1.0
    true_lambda = 1.0 / true_var
    
    # Test different sample sizes
    sample_sizes = [3, 10, 30, 55, 100]
    
    for n_samples in sample_sizes:
        print(f"\n=== Testing with {n_samples} samples ===")
        
        # Generate data
        X = Uniform(-5, 5).sample([n_samples, 1])
        y = X * true_w + true_w0
        noised = y + ch.sqrt(ch.Tensor([true_var])) * ch.randn(y.size(0), 1)
        
        # Truncate
        phi = oracle.Left_Regression(ch.zeros(1))
        indices = phi(noised).nonzero()[:,0]
        x_trunc, y_trunc = X[indices], noised[indices]
        
        actual_samples = x_trunc.shape[0]
        print(f"After truncation: {actual_samples} samples")
        
        # Test multiple lambda values
        for lambda_val in [0.5, 1.0, 1.5]:
            w = ch.tensor([[2.0]], requires_grad=True)
            w0 = ch.tensor([[1.0]], requires_grad=True) 
            lambda_param = ch.tensor([lambda_val], requires_grad=True)
            
            pred = x_trunc @ w.T.float() + w0.float()
            from delphi.grad import TruncatedUnknownVarianceMSE
            loss = TruncatedUnknownVarianceMSE.apply(pred, y_trunc, lambda_param, phi, 100)
            
            loss.backward()
            grad_lambda = lambda_param.grad.item()
            
            expected_sign = "positive" if lambda_val < 1.0 else "negative" if lambda_val > 1.0 else "zero"
            actual_sign = "positive" if grad_lambda > 0 else "negative" if grad_lambda < 0 else "zero"
            
            print(f"  λ={lambda_val:.1f}: grad={grad_lambda:8.4f}, expected={expected_sign}, got={actual_sign}")
            
            # Clean up
            w.grad = None
            w0.grad = None
            lambda_param.grad = None

def test_monte_carlo_stability():
    """Test if Monte Carlo sampling is causing the issue"""
    import torch
    
    ch.manual_seed(42)
    
    # Fixed simple case
    X = ch.tensor([[1.0], [2.0], [3.0]])
    y = ch.tensor([[3.0], [5.0], [7.0]]) + ch.randn(3, 1) * 0.1  # Small noise
    
    phi = oracle.Left_Regression(ch.zeros(1))
    
    w = ch.tensor([[2.0]], requires_grad=True)
    w0 = ch.tensor([[1.0]], requires_grad=True) 
    lambda_param = ch.tensor([1.5], requires_grad=True)
    
    pred = X @ w.T.float() + w0.float()
    
    print("\n=== MONTE CARLO STABILITY TEST ===")
    
    # Test different numbers of Monte Carlo samples
    for mc_samples in [10, 50, 100, 500, 1000]:
        lambda_param.grad = None

        from delphi.grad import TruncatedUnknownVarianceMSE
        
        loss = TruncatedUnknownVarianceMSE.apply(pred, y, lambda_param, phi, mc_samples)
        loss.backward()
        
        print(f"MC samples: {mc_samples:4d}, loss: {loss.item():8.4f}, grad_λ: {lambda_param.grad.item():8.4f}")

def test_monte_carlo_stability():
    """Test if Monte Carlo sampling is causing the issue"""
    import torch
    
    ch.manual_seed(42)
    
    # Use the 55-sample case where we saw the sign flip
    X = Uniform(-5, 5).sample([55, 1])
    y = X * 2.0 + 1.0
    noised = y + ch.randn(y.size(0), 1)  # var=1.0
    
    phi = oracle.Left_Regression(ch.zeros(1))
    indices = phi(noised).nonzero()[:,0]
    x_trunc, y_trunc = X[indices], noised[indices]
    
    w = ch.tensor([[2.0]], requires_grad=True)
    w0 = ch.tensor([[1.0]], requires_grad=True) 
    lambda_param = ch.tensor([1.5], requires_grad=True)
    
    pred = x_trunc @ w.T.float() + w0.float()
    
    print(f"Using {x_trunc.shape[0]} truncated samples")
    print("\n=== MONTE CARLO STABILITY TEST ===")
    
    # Test different numbers of Monte Carlo samples
    for mc_samples in [10, 50, 100, 500, 1000]:
        import pdb; pdb.set_trace()

        lambda_param = ch.tensor([1.5], requires_grad=True)
        pred = pred.clone()
        from delphi.grad import TruncatedUnknownVarianceMSE 
        loss = TruncatedUnknownVarianceMSE.apply(pred, y_trunc, lambda_param, phi, mc_samples)
        loss.backward()
        
        print(f"MC samples: {mc_samples:4d}, loss: {loss.item():8.4f}, grad_λ: {lambda_param.grad.item():8.4f}")


def test_corrected_gradient():
    """Test the corrected gradient implementation"""
    import torch
    
    ch.manual_seed(42)
    
    # Simple test case
    X = ch.tensor([[1.0], [2.0], [3.0]])
    y = X * 2.0 + 1.0 + ch.randn(3, 1) * 0.1
    
    phi = oracle.Left_Regression(ch.zeros(1))
    
    # Test different lambda values
    test_lambdas = [0.5, 1.0, 1.5, 2.0]
    
    for lambda_val in test_lambdas:
        print(f"\n=== Testing λ={lambda_val} ===")
        
        pred = ch.tensor([[3.0], [5.0], [7.0]], requires_grad=True)
        lambda_param = ch.tensor([lambda_val], requires_grad=True)
        
        # Single forward-backward pass
        from delphi.grad import TruncatedUnknownVarianceMSE
        loss = TruncatedUnknownVarianceMSE.apply(pred, y, lambda_param, phi, 100)
        print(f"Loss: {loss.item():.4f}")
        
        loss.backward()
        
        print(f"Grad pred: {pred.grad.norm().item():.4f}")
        print(f"Grad lambda: {lambda_param.grad.item():.4f}")
        
        # Expected behavior check
        true_lambda = 1.0  # Since true variance = 1.0
        if lambda_val < true_lambda:
            print(f"Expected grad_lambda: POSITIVE (λ too small)")
        elif lambda_val > true_lambda:
            print(f"Expected grad_lambda: NEGATIVE (λ too large)")
        else:
            print(f"Expected grad_lambda: ZERO (optimal λ)")
        
        # Reset
        pred.grad = None
        lambda_param.grad = None

def test_debug_gradient_components():
    """Debug each component of the gradient computation"""
    import torch
    
    ch.manual_seed(42)
    
    # Simple test case
    X = ch.tensor([[1.0], [2.0], [3.0]])
    y = X * 2.0 + 1.0 + ch.randn(3, 1) * 0.1
    phi = oracle.Left_Regression(ch.zeros(1))
    
    lambda_val = 0.5  # Too small - should have POSITIVE gradient
    pred = ch.tensor([[3.0], [5.0], [7.0]], requires_grad=False)
    lambda_param = ch.tensor([lambda_val], requires_grad=False)
    
    # Manual computation to see what's happening
    sigma = 1.0 / ch.sqrt(lambda_param)
    batch_size = pred.shape[0]
    
    # Generate Monte Carlo samples
    stacked = pred.unsqueeze(-1).repeat(1, 1, 100)
    noised = stacked + sigma * ch.randn_like(stacked)
    filtered = phi(noised)
    out = noised * filtered
    
    # Compute E[z|z∈S] and E[z²|z∈S]
    valid_counts = filtered.sum(dim=-1, keepdim=True) + 1e-5
    z = out.sum(dim=-1, keepdim=True) / valid_counts
    z_2 = ((out.pow(2)).sum(dim=-1, keepdim=True) / valid_counts).flatten()
    
    print("=== GRADIENT COMPONENT ANALYSIS ===")
    print(f"True lambda: 1.0, Test lambda: {lambda_val}")
    print(f"Sigma: {sigma.item():.4f}")
    print(f"Batch size: {batch_size}")
    print()
    
    # Analyze each term
    term1 = 0.5 * (y.pow(2).flatten() - z_2)
    term2 = pred.flatten() * (z.flatten() - y.flatten())
    
    print("Component shapes:")
    print(f"y: {y.shape}, z: {z.shape}, z_2: {z_2.shape}")
    print(f"pred: {pred.shape}")
    print()
    
    print("Term 1 (0.5*(y² - z_2)):")
    print(f"  y²: {y.pow(2).squeeze()}")
    print(f"  z_2: {z_2.squeeze()}")
    print(f"  y² - z_2: {(y.pow(2).flatten() - z_2).squeeze()}")
    print(f"  term1: {term1.squeeze()}")
    print()
    
    print("Term 2 (pred*(z - y)):")
    print(f"  z: {z.squeeze()}")
    print(f"  y: {y.squeeze()}")
    print(f"  z - y: {(z.flatten() - y.flatten()).squeeze()}")
    print(f"  pred: {pred.squeeze()}")
    print(f"  term2: {term2.squeeze()}")
    print()
    
    grad_lambda = term1 + term2
    print("Total gradient components:")
    print(f"  term1 mean: {term1.mean().item():.4f}")
    print(f"  term2 mean: {term2.mean().item():.4f}")
    print(f"  total mean: {grad_lambda.mean().item():.4f}")
    
    # Check if z and z_2 make sense
    print()
    print("=== MONTECARLO SAMPLING CHECK ===")
    print(f"z range: [{z.min().item():.3f}, {z.max().item():.3f}]")
    print(f"z_2 range: [{z_2.min().item():.3f}, {z_2.max().item():.3f}]")
    print(f"y range: [{y.min().item():.3f}, {y.max().item():.3f}]")
    print(f"pred range: [{pred.min().item():.3f}, {pred.max().item():.3f}]")


def test_corrected_gradient_v2():
    """Test with the potentially corrected gradient formula"""
    import torch
    
    ch.manual_seed(42)
    
    class TruncatedUnknownVarianceMSE_v2(ch.autograd.Function):
        @staticmethod
        def forward(ctx, pred, targ, lambda_, phi, num_samples=10, eps=1e-5):
            sigma = 1.0 / ch.sqrt(lambda_)
            batch_size = pred.shape[0]
            
            stacked = pred.unsqueeze(-1).repeat(1, 1, num_samples)
            noised = stacked + sigma * ch.randn_like(stacked)
            filtered = phi(noised)
            out = noised * filtered
            
            valid_counts = filtered.sum(dim=-1, keepdim=True) + eps
            z = out.sum(dim=-1, keepdim=True) / valid_counts
            z_2 = (out.pow(2)).sum(dim=-1, keepdim=True) / valid_counts
            
            # Negative log-likelihood
            nll = -0.5 * lambda_ * targ.pow(2) + lambda_ * targ * pred
            const = -0.5 * lambda_ * z_2 + lambda_ * z * pred
            
            loss = (nll - const).mean()
            
            ctx.save_for_backward(pred, targ, lambda_, z, z_2)
            return loss

        @staticmethod
        def backward(ctx, grad_output):
            pred, targ, lambda_, z, z_2 = ctx.saved_tensors
            batch_size = pred.shape[0]
            
            # POTENTIAL FIX: Try different gradient formulations
            
            # Option 1: Original (seems wrong)
            # grad_lambda = 0.5 * (targ.pow(2) - z_2) + pred * (z - targ)
            
            # Option 2: Flipped signs
            # grad_lambda = 0.5 * (z_2 - targ.pow(2)) + pred * (targ - z)
            
            # Option 3: Paper equation 12 exactly
            # ∂L/∂λ = ½(y² - E[z²|z∈S]) + μ(z - E[z|z∈S])
            # But careful: our 'z' is already E[z|z∈S], so this becomes:
            grad_lambda = 0.5 * (targ.pow(2) - z_2) + pred * (z - targ)
            
            # Option 4: Maybe we need to consider the NLL formulation differently
            # L = -log p(y|μ,σ) + log p(z∈S|μ,σ)
            # So ∂L/∂λ = -∂/∂λ log p(y) + ∂/∂λ log p(z∈S)
            
            grad_pred = lambda_ * (targ - z)
            
            grad_pred = grad_output * grad_pred.mean(dim=0, keepdim=True) / batch_size
            grad_lambda = grad_output * grad_lambda.mean() / batch_size
            
            return grad_pred, None, grad_lambda[...,None], None, None, None
    
    # Test the new version
    X = ch.tensor([[1.0], [2.0], [3.0]])
    y = X * 2.0 + 1.0 + ch.randn(3, 1) * 0.1
    phi = oracle.Left_Regression(ch.zeros(1))
    
    test_lambdas = [0.5, 1.0, 1.5]
    
    for lambda_val in test_lambdas:
        print(f"\n=== Testing λ={lambda_val} with v2 ===")
        
        pred = ch.tensor([[3.0], [5.0], [7.0]], requires_grad=True)
        lambda_param = ch.tensor([lambda_val], requires_grad=True)
        
        loss = TruncatedUnknownVarianceMSE_v2.apply(pred, y, lambda_param, phi, 100)
        loss.backward()
        
        print(f"Loss: {loss.item():.4f}")
        print(f"Grad lambda: {lambda_param.grad.item():.4f}")
        
        true_lambda = 1.0
        if lambda_val < true_lambda:
            expected = "POSITIVE"
        elif lambda_val > true_lambda:
            expected = "NEGATIVE" 
        else:
            expected = "ZERO"
            
        actual = "POSITIVE" if lambda_param.grad.item() > 0 else "NEGATIVE"
        print(f"Expected: {expected}, Got: {actual}")
        
        pred.grad = None
        lambda_param.grad = None

def test_most_likely_fix():
    """Test the most likely correction to the gradient"""
    import torch
    
    class TruncatedUnknownVarianceMSE_Fixed(ch.autograd.Function):
        @staticmethod
        def forward(ctx, pred, targ, lambda_, phi, num_samples=100, eps=1e-5):
            sigma = 1.0 / ch.sqrt(lambda_)
            batch_size = pred.shape[0]
            
            stacked = pred.unsqueeze(-1).repeat(1, 1, num_samples)
            noised = stacked + sigma * ch.randn_like(stacked)
            filtered = phi(noised)
            out = noised * filtered
            
            valid_counts = filtered.sum(dim=-1, keepdim=True) + eps
            z = out.sum(dim=-1, keepdim=True) / valid_counts
            z_2 = (out.pow(2)).sum(dim=-1, keepdim=True) / valid_counts
            
            # Current: L = -log p(y) + log p(z∈S)
            # But maybe it should be: L = -[log p(y) - log p(z∈S)] = -log p(y) + log p(z∈S)
            # Our current implementation seems mathematically correct...
            nll = -0.5 * lambda_ * targ.pow(2) + lambda_ * targ * pred
            const = -0.5 * lambda_ * z_2 + lambda_ * z * pred
            
            loss = (nll - const).mean()
            
            ctx.save_for_backward(pred, targ, lambda_, z, z_2)
            return loss

        @staticmethod 
        def backward(ctx, grad_output):
            pred, targ, lambda_, z, z_2 = ctx.saved_tensors
            batch_size = pred.shape[0]
            
            # THE MOST LIKELY FIX: Flip the gradient sign
            # If all gradients are negative, maybe we computed -∂L/∂λ instead of ∂L/∂λ
            grad_lambda = 0.5 * (z_2 - targ.pow(2)) + pred * (targ - z)
            
            grad_pred = lambda_ * (targ - z)
            
            grad_pred = grad_output * grad_pred.mean(dim=0, keepdim=True)
            grad_lambda = grad_output * grad_lambda.mean()
            
            return grad_pred, None, grad_lambda[...,None], None, None, None
    
    # Test the fixed version
    ch.manual_seed(42)
    X = ch.tensor([[1.0], [2.0], [3.0]])
    y = X * 2.0 + 1.0 + ch.randn(3, 1) * 0.1
    phi = oracle.Left_Regression(ch.zeros(1))
    
    test_lambdas = [0.5, 1.0, 1.5]
    
    for lambda_val in test_lambdas:
        print(f"\n=== Testing λ={lambda_val} with FLIPPED GRADIENT ===")
        
        pred = ch.tensor([[3.0], [5.0], [7.0]], requires_grad=True)
        lambda_param = ch.tensor([lambda_val], requires_grad=True)
        
        loss = TruncatedUnknownVarianceMSE_Fixed.apply(pred, y, lambda_param, phi, 100)
        loss.backward()
        
        print(f"Loss: {loss.item():.4f}")
        print(f"Grad lambda: {lambda_param.grad.item():.4f}")
        
        true_lambda = 1.0
        if lambda_val < true_lambda:
            expected = "POSITIVE"
        elif lambda_val > true_lambda:
            expected = "NEGATIVE" 
        else:
            expected = "ZERO"
            
        actual = "POSITIVE" if lambda_param.grad.item() > 0 else "NEGATIVE"
        print(f"Expected: {expected}, Got: {actual}")
        
        pred.grad = None
        lambda_param.grad = None

class TruncatedUnknownVarianceMSE_Correct(ch.autograd.Function):
    """
    Correct implementation based on Equation 4.2 from the paper
    """
    @staticmethod
    def forward(ctx, pred, targ, lambda_, phi, num_samples=100, eps=1e-5):
        sigma = 1.0 / ch.sqrt(lambda_)
        batch_size = pred.shape[0]
        
        # Generate Monte Carlo samples for the truncation term
        stacked = pred.unsqueeze(-1).repeat(1, 1, num_samples)
        noised = stacked + sigma * ch.randn_like(stacked)
        filtered = phi(noised)
        out = noised * filtered
        
        # Compute E[z|z∈S] and E[z²|z∈S] 
        valid_counts = filtered.sum(dim=-1, keepdim=True) + eps
        z = out.sum(dim=-1, keepdim=True) / valid_counts
        z_2 = (out.pow(2)).sum(dim=-1, keepdim=True) / valid_counts
        
        # Loss function based on Equation 4.2
        # L = E[ℓ(y)|y∈S] - E[ℓ(z)|z∈S]
        # where ℓ(y) = -log p(y|μ,σ) = λ(y-μ)²/2 - ½log(λ) + const
        
        # First term: E[ℓ(y)|y∈S]
        term1 = 0.5 * lambda_ * (targ - pred).pow(2) - 0.5 * ch.log(lambda_)
        
        # Second term: E[ℓ(z)|z∈S]  
        term2 = 0.5 * lambda_ * (z - pred).pow(2) - 0.5 * ch.log(lambda_)
        
        loss = (term1 - term2).mean()
        
        ctx.save_for_backward(pred, targ, lambda_, z, z_2)
        return loss

    @staticmethod 
    def backward(ctx, grad_output):
        pred, targ, lambda_, z, z_2 = ctx.saved_tensors
        batch_size = pred.shape[0]
        
        # CORRECT GRADIENT BASED ON EQUATION 4.2:
        # ∂L/∂μ = λ[E[z|z∈S] - E[y|y∈S]]
        grad_pred = lambda_ * (z - targ)
        
        # ∂L/∂λ = ½[E[(y-μ)²|y∈S] - E[(z-μ)²|z∈S]]
        grad_lambda = 0.5 * ((targ - pred).pow(2) - (z - pred).pow(2))
        
        # Average over batch
        grad_pred = grad_output * grad_pred.mean(dim=0, keepdim=True)
        grad_lambda = grad_output * grad_lambda.mean()
        
        return grad_pred, None, grad_lambda[...,None], None, None, None

def test_equation_4_2():
    """Test the correct implementation based on Equation 4.2"""
    import torch
    
    ch.manual_seed(42)
    
    X = ch.tensor([[1.0], [2.0], [3.0]])
    y = X * 2.0 + 1.0 + ch.randn(3, 1) * 0.1
    phi = oracle.Left_Regression(ch.zeros(1))
    
    print("=== TESTING EQUATION 4.2 IMPLEMENTATION ===")
    
    test_lambdas = [0.5, 1.0, 1.5]
    
    for lambda_val in test_lambdas:
        print(f"\n=== Testing λ={lambda_val} ===")
        
        pred = ch.tensor([[3.0], [5.0], [7.0]], requires_grad=True)
        lambda_param = ch.tensor([lambda_val], requires_grad=True)
        
        loss = TruncatedUnknownVarianceMSE_Correct.apply(pred, y, lambda_param, phi, 100)
        loss.backward()
        
        print(f"Loss: {loss.item():.4f}")
        print(f"Grad lambda: {lambda_param.grad.item():.4f}")
        
        true_lambda = 1.0
        if lambda_val < true_lambda:
            expected = "POSITIVE"
        elif lambda_val > true_lambda:
            expected = "NEGATIVE" 
        else:
            expected = "ZERO"
            
        actual = "POSITIVE" if lambda_param.grad.item() > 0 else "NEGATIVE"
        print(f"Expected: {expected}, Got: {actual}")
        
        pred.grad = None
        lambda_param.grad = None

def test_debug_monte_carlo_sampling_detailed():
    """Detailed debug of the Monte Carlo sampling to understand what's happening"""
    import torch
    
    ch.manual_seed(42)
    
    X = ch.tensor([[1.0], [2.0], [3.0]])
    y = X * 2.0 + 1.0 + ch.randn(3, 1) * 0.1
    phi = oracle.Left_Regression(ch.zeros(1))
    pred = ch.tensor([[3.0], [5.0], [7.0]])
    
    lambda_val = 0.5  # Too small - should have POSITIVE gradient
    lambda_param = ch.tensor([lambda_val])
    sigma = 1.0 / ch.sqrt(lambda_param)
    
    print("=== DETAILED MONTE CARLO DEBUG ===")
    print(f"True lambda: 1.0, Test lambda: {lambda_val}")
    print(f"Sigma: {sigma.item():.4f}")
    print(f"Pred: {pred.squeeze()}")
    print(f"Target: {y.squeeze()}")
    print()
    
    # Generate samples
    stacked = pred.unsqueeze(-1).repeat(1, 1, 100)
    noise = sigma * ch.randn_like(stacked)
    noised = stacked + noise
    
    print(f"Noise stats: mean={noise.mean().item():.4f}, std={noise.std().item():.4f}")
    print(f"Noised range: [{noised.min().item():.3f}, {noised.max().item():.3f}]")
    
    # Apply truncation
    filtered = phi(noised)
    out = noised * filtered
    
    print(f"Filtered acceptance: {filtered.sum().item()}/{filtered.numel()} samples")
    
    # Check what values are being accepted/rejected
    for i in range(3):
        sample_mask = filtered[i, 0, :].bool()
        accepted = noised[i, 0, sample_mask]
        rejected = noised[i, 0, ~sample_mask]
        print(f"Sample {i}: {accepted.numel()}/100 accepted")
        if accepted.numel() > 0:
            print(f"  Accepted: [{accepted.min().item():.3f}, {accepted.max().item():.3f}]")
            print(f"  Accepted mean: {accepted.mean().item():.3f}")
        if rejected.numel() > 0:
            print(f"  Rejected: [{rejected.min().item():.3f}, {rejected.max().item():.3f}]")
    
    # Compute expectations
    valid_counts = filtered.sum(dim=-1, keepdim=True) + 1e-5
    z = out.sum(dim=-1, keepdim=True) / valid_counts
    z_sq = (out.pow(2)).sum(dim=-1, keepdim=True) / valid_counts
    
    print(f"\nz (E[z|z∈S]): {z.squeeze()}")
    print(f"z_sq (E[z²|z∈S]): {z_sq.squeeze()}")
    
    # Check the variance relationships
    empirical_var = ((y - pred).pow(2)).mean()
    sampled_var = ((z - pred).pow(2)).mean()
    
    print(f"\nVariance analysis:")
    print(f"Empirical variance E[(y-μ)²]: {empirical_var.item():.4f}")
    print(f"Sampled variance E[(z-μ)²]: {sampled_var.item():.4f}")
    print(f"Difference: {empirical_var.item() - sampled_var.item():.4f}")
    
    # What SHOULD the relationship be?
    print(f"\nExpected behavior:")
    print(f"If lambda is too small (σ too large), E[(z-μ)²] should be LARGER than E[(y-μ)²]")
    print(f"Because truncation removes the extreme values, reducing variance")
    print(f"So E[(y-μ)²] - E[(z-μ)²] should be NEGATIVE")
    print(f"Which would give NEGATIVE gradient (to decrease lambda even more!)")
    print(f"Wait... this might explain our consistently negative gradients!")


def test_with_imperfect_predictions():
    """Test with imperfect predictions (more realistic training scenario)"""
    import torch
    
    ch.manual_seed(42)
    
    X = ch.tensor([[1.0], [2.0], [3.0]])
    true_w = 2.0
    true_w0 = 1.0
    y = X * true_w + true_w0 + ch.randn(3, 1) * 1.0  # Larger noise
    
    phi = oracle.Left_Regression(ch.zeros(1))
    
    print("=== TESTING WITH IMPERFECT PREDICTIONS ===")
    print("Using WRONG predictions to simulate early training")
    
    # Use deliberately wrong predictions
    wrong_preds = [ch.tensor([[2.5]], requires_grad=True),  # too small
                   ch.tensor([[5.5]], requires_grad=True),  # too large  
                   ch.tensor([[4.0]], requires_grad=True)]  # too small
    
    test_lambdas = [0.5, 1.0, 1.5]
    
    for i, pred in enumerate(wrong_preds):
        print(f"\n--- Sample {i} ---")
        print(f"True: μ={X[i].item()*2+1:.1f}, y={y[i].item():.3f}")
        print(f"Pred: μ={pred.item():.1f}")
        
        for lambda_val in test_lambdas:
            lambda_param = ch.tensor([lambda_val], requires_grad=True)

            from delphi.grad import TruncatedUnknownVarianceMSE 
            loss = TruncatedUnknownVarianceMSE.apply(pred, y[i:i+1], lambda_param, phi, 100)
            loss.backward()
            
            print(f"  λ={lambda_val:.1f}: loss={loss.item():.4f}, grad_λ={lambda_param.grad.item():.4f}")
            
            lambda_param.grad = None

def test_realistic_training_scenario():
    """Test a more realistic training scenario"""
    import torch
    
    ch.manual_seed(42)
    
    # Generate more realistic data
    n_samples = 50
    X = Uniform(-3, 3).sample([n_samples, 1])
    true_w = 2.0
    true_w0 = 1.0
    true_var = 1.0
    y = X * true_w + true_w0 + ch.sqrt(ch.Tensor([true_var])) * ch.randn(n_samples, 1)
    
    # Apply truncation (like real data)
    phi = oracle.Left_Regression(ch.zeros(1))
    indices = phi(y).nonzero()[:, 0]
    X_trunc, y_trunc = X[indices], y[indices]
    
    print(f"\n=== REALISTIC TRAINING SCENARIO ===")
    print(f"Using {X_trunc.shape[0]}/{n_samples} truncated samples")
    
    # Test with different prediction qualities
    test_cases = [
        ("Good predictions", X_trunc * true_w + true_w0),
        ("Bad predictions", X_trunc * 1.0 + 0.0),  # Wrong parameters
        ("Random predictions", ch.randn_like(y_trunc) * 3.0)
    ]
    
    for desc, pred in test_cases:
        print(f"\n--- {desc} ---")
        
        for lambda_val in [0.5, 1.0, 1.5]:
            lambda_param = ch.tensor([lambda_val], requires_grad=True)
            pred_tensor = pred.clone().requires_grad_(True)

            from delphi.grad import TruncatedUnknownVarianceMSE 
            loss = TruncatedUnknownVarianceMSE.apply(pred_tensor, y_trunc, lambda_param, phi, 100)
            loss.backward()
            
            print(f"  λ={lambda_val:.1f}: loss={loss.item():.4f}, grad_λ={lambda_param.grad.item():.4f}")
            
            lambda_param.grad = None
            pred_tensor.grad = None


def test_convexity():
    """Test that the loss function is convex in lambda"""
    import torch
    
    ch.manual_seed(42)
    
    # Realistic training scenario
    n_samples = 100
    X = Uniform(-3, 3).sample([n_samples, 1])
    true_w = 2.0
    true_w0 = 1.0
    true_var = ch.Tensor([1.0])
    y = X * true_w + true_w0 + ch.sqrt(true_var) * ch.randn(n_samples, 1)
    
    phi = oracle.Left_Regression(ch.zeros(1))
    indices = phi(y).nonzero()[:, 0]
    X_trunc, y_trunc = X[indices], y[indices]
    
    # Good but not perfect predictions
    pred = X_trunc * 1.8 + 1.2  # Close but not perfect
    
    print(f"=== CONVEXITY TEST ===")
    print(f"Using {X_trunc.shape[0]}/{n_samples} truncated samples")
    print(f"Predictions are close to true values but not perfect")
    
    lambda_range = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 2.0, 3.0, 4.0, 5.0]
    losses = []
    gradients = []
    
    for lambda_val in lambda_range:
        lambda_param = ch.tensor([lambda_val], requires_grad=True)
        pred_tensor = pred.clone().requires_grad_(True)

        from delphi.grad import TruncatedUnknownVarianceMSE 
        loss = TruncatedUnknownVarianceMSE.apply(pred_tensor, y_trunc, lambda_param, phi, 100)
        losses.append(loss.item())
        
        loss.backward()
        gradients.append(lambda_param.grad.item())
        
        print(f"λ={lambda_val:.1f}: loss={loss.item():.4f}, grad={lambda_param.grad.item():.4f}")
        
        lambda_param.grad = None
        pred_tensor.grad = None
    
    # Find minimum
    min_idx = ch.tensor(losses).argmin()
    optimal_lambda = lambda_range[min_idx]
    
    print(f"\nOptimal lambda: {optimal_lambda:.2f} (true: 1.0)")
    print(f"Minimum loss: {losses[min_idx]:.4f}")
    
    # Check convexity: gradients should change from negative to positive
    negative_grads = sum(1 for g in gradients if g < 0)
    positive_grads = sum(1 for g in gradients if g > 0)
    
    print(f"Negative gradients: {negative_grads}, Positive gradients: {positive_grads}")
    
    if negative_grads > 0 and positive_grads > 0:
        print("✓ Loss function appears convex in lambda!")
    else:
        print("⚠️  Potential non-convexity detected")

def test_optimization_convergence():
    """Test that optimization actually converges to the right lambda"""
    import torch
    
    ch.manual_seed(42)
    
    # Realistic setup
    n_samples = 200
    X = Uniform(-3, 3).sample([n_samples, 1])
    true_w = 2.0
    true_w0 = 1.0
    true_var = 1.0
    y = X * true_w + true_w0 + ch.sqrt(true_var) * ch.randn(n_samples, 1)
    
    phi = oracle.Left_Regression(ch.zeros(1))
    indices = phi(y).nonzero()[:, 0]
    X_trunc, y_trunc = X[indices], y[indices]
    
    # Fixed reasonable predictions
    pred = X_trunc * 1.9 + 0.9  # Close to true
    
    print(f"\n=== OPTIMIZATION CONVERGENCE TEST ===")
    print(f"Using {X_trunc.shape[0]}/{n_samples} truncated samples")
    
    # Initialize lambda wrong
    lambda_param = ch.tensor([0.3], requires_grad=True)  # Start too small
    
    optimizer = ch.optim.Adam([lambda_param], lr=0.01)
    
    print("Iter | Lambda | Loss | Gradient")
    print("-----|--------|------|----------")
    
    for i in range(50):
        optimizer.zero_grad()
        
        pred_tensor = pred.clone().requires_grad_(True)
        loss = TruncatedUnknownVarianceMSE.apply(pred_tensor, y_trunc, lambda_param, phi, 100)
        
        loss.backward()
        
        if i % 10 == 0:
            print(f"{i:4d} | {lambda_param.item():.3f} | {loss.item():.4f} | {lambda_param.grad.item():.4f}")
        
        optimizer.step()
        
        # Clamp lambda to reasonable values
        with ch.no_grad():
            lambda_param.clamp_(0.1, 10.0)
    
    print(f"Final lambda: {lambda_param.item():.3f} (true: 1.0)")
    print(f"Error: {abs(lambda_param.item() - 1.0):.3f}")


import torch as ch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

class TruncatedUnknownVarianceMSEOriginal(ch.autograd.Function):
    """Your original implementation"""
    @staticmethod
    def forward(ctx, pred, targ, lambda_, phi, num_samples=100, eps=1e-5):
        sigma = 1 / ch.sqrt(lambda_)
        stacked = pred[..., None].repeat(1, num_samples, 1)
        noised = stacked + sigma * ch.randn(stacked.size())
        filtered = phi(noised)
        out = noised * filtered
        z = out.sum(dim=1) / (filtered.sum(dim=1) + eps)
        z_2 = out.pow(2).sum(dim=1) / (filtered.sum(dim=1) + eps)
        nll = 0.5 * lambda_ * targ.pow(2) - lambda_ * targ * pred
        const = -0.5 * lambda_ * z_2 + z * pred * lambda_
        ctx.save_for_backward(pred, targ, lambda_, z, z_2)
        return -(nll + const).mean(0)

    @staticmethod
    def backward(ctx, grad_output):
        pred, targ, lambda_, z, z_2 = ctx.saved_tensors
        lambda_grad = .5 * (targ.pow(2) - z_2)
        return lambda_ * (z - targ) / pred.size(0), None, lambda_grad / pred.size(0), None, None, None


class TruncatedUnknownVarianceMSECorrected1(ch.autograd.Function):
    """Corrected version with full Gaussian NLL"""
    @staticmethod
    def forward(ctx, pred, targ, lambda_, phi, num_samples=100, eps=1e-5):
        sigma = 1 / ch.sqrt(lambda_)
        stacked = pred[..., None].repeat(1, num_samples, 1)
        noised = stacked + sigma * ch.randn(stacked.size())
        filtered = phi(noised)
        out = noised * filtered
        z = out.sum(dim=1) / (filtered.sum(dim=1) + eps)
        z_2 = out.pow(2).sum(dim=1) / (filtered.sum(dim=1) + eps)
        
        # Full Gaussian NLL: -(1/2)log(λ) + (λ/2)(y - pred)²
        nll = -0.5 * ch.log(lambda_) + 0.5 * lambda_ * (targ - pred).pow(2)
        
        # Expectation correction: E_q[log p(z)] 
        const = -0.5 * lambda_ * z_2 + lambda_ * z * pred - 0.5 * lambda_ * pred.pow(2)
        
        ctx.save_for_backward(pred, targ, lambda_, z, z_2)
        return (nll + const).mean(0)

    @staticmethod
    def backward(ctx, grad_output):
        pred, targ, lambda_, z, z_2 = ctx.saved_tensors
        
        # ∇pred: λ(pred - targ) - λ(z - pred) = λ(2*pred - targ - z)
        pred_grad = grad_output * lambda_ * (pred - targ - (z - pred)) / pred.size(0)
        
        # ∇lambda: -1/(2λ) + (1/2)(targ-pred)² - (1/2)(z² - 2z*pred + pred²)
        lambda_grad = grad_output * (-0.5 / lambda_ + 0.5 * (targ - pred).pow(2) 
                                     - 0.5 * (z_2 - 2*z*pred + pred.pow(2))) / pred.size(0)
        
        return pred_grad, None, lambda_grad, None, None, None


class TruncatedUnknownVarianceMSECorrected2(ch.autograd.Function):
    """Alternative corrected version"""
    @staticmethod
    def forward(ctx, pred, targ, lambda_, phi, num_samples=100, eps=1e-5):
        sigma = 1 / ch.sqrt(lambda_)
        stacked = pred[..., None].repeat(1, num_samples, 1)
        noised = stacked + sigma * ch.randn(stacked.size())
        filtered = phi(noised)
        out = noised * filtered
        z = out.sum(dim=1) / (filtered.sum(dim=1) + eps)
        z_2 = out.pow(2).sum(dim=1) / (filtered.sum(dim=1) + eps)
        
        # Direct from paper: -log p(y_obs) + E_q[log p(z)]
        log_lik_obs = -0.5 * ch.log(lambda_) + 0.5 * lambda_ * (targ - pred).pow(2)
        exp_log_lik_z = -0.5 * ch.log(lambda_) + 0.5 * lambda_ * (z_2 - 2*z*pred + pred.pow(2))
        
        ctx.save_for_backward(pred, targ, lambda_, z, z_2)
        return (log_lik_obs - exp_log_lik_z).mean(0)

    @staticmethod
    def backward(ctx, grad_output):
        pred, targ, lambda_, z, z_2 = ctx.saved_tensors
        
        # Simplified gradient
        pred_grad = grad_output * lambda_ * (targ - z) / pred.size(0)
        
        lambda_grad = grad_output * (0.5 * ((targ - pred).pow(2) - (z_2 - 2*z*pred + pred.pow(2))) / lambda_) / pred.size(0)
        
        return pred_grad, None, lambda_grad, None, None, None


def oracle_positive(y):
    """Oracle: indicator for y > 0"""
    return (y > 0).float()


def numerical_gradient(loss_fn, pred, targ, lambda_, phi, h=1e-5, num_samples=10000):
    """Compute numerical gradients using finite differences"""
    
    # Gradient w.r.t pred
    pred_plus = pred.clone()
    pred_plus[0, 0] += h
    pred_minus = pred.clone()
    pred_minus[0, 0] -= h
    
    loss_plus = loss_fn(pred_plus, targ, lambda_, phi, num_samples)
    loss_minus = loss_fn(pred_minus, targ, lambda_, phi, num_samples)
    grad_pred = (loss_plus - loss_minus) / (2 * h)
    
    # Gradient w.r.t lambda
    lambda_plus = lambda_ + h
    lambda_minus = lambda_ - h
    
    loss_plus = loss_fn(pred, targ, lambda_plus, phi, num_samples)
    loss_minus = loss_fn(pred, targ, lambda_minus, phi, num_samples)
    grad_lambda = (loss_plus - loss_minus) / (2 * h)
    
    return grad_pred.item(), grad_lambda.item()


def test_gradients_single_point():
    """Test gradients at a single point"""
    print("=" * 60)
    print("Testing Gradients at Single Point")
    print("=" * 60)
    
    pred = ch.tensor([[1.0]], requires_grad=True)
    targ = ch.tensor([[0.5]])
    lambda_ = ch.tensor(1.0, requires_grad=True)
    phi = oracle_positive
    
    # Test each implementation
    implementations = [
        ("Original", TruncatedUnknownVarianceMSEOriginal),
        ("Corrected1", TruncatedUnknownVarianceMSECorrected1),
        ("Corrected2", TruncatedUnknownVarianceMSECorrected2),
    ]
    
    for name, impl in implementations:
        print(f"\n{name} Implementation:")
        
        # Compute loss and analytical gradients
        loss = impl.apply(pred, targ, lambda_, phi, num_samples=1000)
        loss.backward()
        
        ana_grad_pred = pred.grad.item()
        ana_grad_lambda = lambda_.grad.item()
        
        print(f"  Loss: {loss.item():.6f}")
        print(f"  Analytical ∇pred: {ana_grad_pred:.6f}")
        print(f"  Analytical ∇lambda: {ana_grad_lambda:.6f}")
        
        # Reset gradients
        pred.grad = None
        lambda_.grad = None
        
        # Compute numerical gradients
        num_grad_pred, num_grad_lambda = numerical_gradient(
            lambda p, t, l, phi, ns: impl.apply(p, t, l, phi, ns),
            pred, targ, lambda_, phi, num_samples=10000
        )
        
        print(f"  Numerical ∇pred: {num_grad_pred:.6f}")
        print(f"  Numerical ∇lambda: {num_grad_lambda:.6f}")
        print(f"  Error ∇pred: {abs(ana_grad_pred - num_grad_pred):.6e}")
        print(f"  Error ∇lambda: {abs(ana_grad_lambda - num_grad_lambda):.6e}")


def test_gradient_field():
    """Test gradients over a grid of points"""
    print("\n" + "=" * 60)
    print("Testing Gradient Field")
    print("=" * 60)
    
    pred_vals = np.linspace(-2, 2, 15)
    targ_vals = np.linspace(-2, 2, 15)
    lambda_ = ch.tensor(1.0)
    phi = oracle_positive
    
    implementations = [
        ("Original", TruncatedUnknownVarianceMSEOriginal),
        ("Corrected2", TruncatedUnknownVarianceMSECorrected2),
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for impl_idx, (name, impl) in enumerate(implementations):
        print(f"\nTesting {name}...")
        
        losses = np.zeros((len(pred_vals), len(targ_vals)))
        ana_grads = np.zeros((len(pred_vals), len(targ_vals)))
        num_grads = np.zeros((len(pred_vals), len(targ_vals)))
        errors = np.zeros((len(pred_vals), len(targ_vals)))
        
        for i, pred_val in enumerate(pred_vals):
            for j, targ_val in enumerate(targ_vals):
                pred = ch.tensor([[pred_val]], requires_grad=True)
                targ = ch.tensor([[targ_val]])
                
                # Analytical gradient
                loss = impl.apply(pred, targ, lambda_, phi, num_samples=500)
                loss.backward()
                ana_grad = pred.grad.item()
                
                # Numerical gradient
                pred.grad = None
                num_grad, _ = numerical_gradient(
                    lambda p, t, l, phi, ns: impl.apply(p, t, l, phi, ns),
                    pred, targ, lambda_, phi, 5000
                )
                
                losses[i, j] = loss.item()
                ana_grads[i, j] = ana_grad
                num_grads[i, j] = num_grad
                errors[i, j] = abs(ana_grad - num_grad)
        
        # Plot loss surface
        ax = axes[impl_idx, 0]
        im = ax.contourf(targ_vals, pred_vals, losses, levels=20, cmap='viridis')
        ax.set_xlabel('Target')
        ax.set_ylabel('Prediction')
        ax.set_title(f'{name}: Loss')
        plt.colorbar(im, ax=ax)
        
        # Plot analytical gradient
        ax = axes[impl_idx, 1]
        im = ax.contourf(targ_vals, pred_vals, ana_grads, levels=20, cmap='RdBu_r')
        ax.set_xlabel('Target')
        ax.set_ylabel('Prediction')
        ax.set_title(f'{name}: Analytical ∇pred')
        plt.colorbar(im, ax=ax)
        
        # Plot gradient error
        ax = axes[impl_idx, 2]
        im = ax.contourf(targ_vals, pred_vals, errors, levels=20, cmap='Reds')
        ax.set_xlabel('Target')
        ax.set_ylabel('Prediction')
        ax.set_title(f'{name}: |Ana - Num| Error')
        plt.colorbar(im, ax=ax)
        
        max_error = np.max(errors)
        mean_error = np.mean(errors)
        print(f"  Max error: {max_error:.6e}")
        print(f"  Mean error: {mean_error:.6e}")
    
    plt.tight_layout()
    plt.savefig('gradient_test_results.png', dpi=150)
    print("\nPlot saved as 'gradient_test_results.png'")


def test_1d_sweep():
    """Test gradients along a 1D sweep for visualization"""
    print("\n" + "=" * 60)
    print("1D Gradient Sweep")
    print("=" * 60)
    
    pred_vals = np.linspace(-3, 3, 50)
    targ = ch.tensor([[1.0]])
    lambda_ = ch.tensor(1.0)
    phi = oracle_positive
    
    implementations = [
        ("Original", TruncatedUnknownVarianceMSEOriginal),
        ("Corrected2", TruncatedUnknownVarianceMSECorrected2),
    ]
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    for name, impl in implementations:
        losses = []
        ana_grads = []
        num_grads = []
        
        for pred_val in pred_vals:
            pred = ch.tensor([[pred_val]], requires_grad=True)
            
            # Analytical
            loss = impl.apply(pred, targ, lambda_, phi, 500)
            loss.backward()
            ana_grad = pred.grad.item()
            
            # Numerical
            pred.grad = None
            num_grad, _ = numerical_gradient(
                lambda p, t, l, phi, ns: impl.apply(p, t, l, phi, ns),
                pred, targ, lambda_, phi, 5000
            )

            # import pdb; pdb.set_trace()
            
            losses.append(loss.item())
            ana_grads.append(ana_grad)
            num_grads.append(num_grad)
        
        # Plot loss
        axes[0].plot(pred_vals, losses, label=name, linewidth=2)
        
        # Plot gradients
        # axes[1].plot(pred_vals, num_grads, label=f'{name} (Numerical)', linewidth=2)
        axes[1].plot(pred_vals, ana_grads, '--', label=f'{name} (Analytical)', linewidth=2)
    
    axes[0].set_xlabel('Prediction')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss vs Prediction (Target=1.0, Lambda=1.0)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Prediction')
    axes[1].set_ylabel('Gradient ∇pred')
    axes[1].set_title('Gradient vs Prediction')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gradient_sweep_1d.png', dpi=150)
    print("\nPlot saved as 'gradient_sweep_1d.png'")
    
    plt.show()





