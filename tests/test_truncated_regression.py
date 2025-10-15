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
def test_known_truncated_regression_one_dimension():
    D, K = 1, 1
    SAMPLES = 1000
    w_ = Uniform(-1, 1)
    M = Uniform(-10, 10)
    # generate ground truth
    NOISE_VAR = ch.ones(1, 1)
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
    train_kwargs = Parameters({'alpha': alpha,
                                'epochs': 2,
                                'lr': 1e-1,
                                'num_samples': 50,
                                'batch_size': 5,
                                'trials': 1,
                                'constant': True, 
                                'verbose': True
                            }) 
    trunc_reg = stats.TruncatedLinearRegression(phi_scale, 
                                                train_kwargs, 
                                                noise_var=ch.ones(1, 1))
    trunc_reg.fit(x_trunc, y_trunc_scale)
    w_ = ch.cat([(trunc_reg.coef_).flatten(), trunc_reg.intercept_]) * ch.sqrt(NOISE_VAR)
    print(f'estimated weights: {w_}')
    trunc_mse_loss = mse_loss(gt_, w_.flatten())
    print(f'truc mse loss: {trunc_mse_loss}')
    msg = f'trunc mse loss is larger than empirical mse loss. known mse loss is {trunc_mse_loss}, and empirical mse loss is: {emp_mse_loss}'
    assert trunc_mse_loss <= emp_mse_loss, msg
    msg = f'trunc mse loss: {trunc_mse_loss}, which is larger than 1e-1'
    assert trunc_mse_loss <= 1e-1, msg

# left truncated linear regression with known variance - 100 dimensions
def test_known_truncated_regression_higher_dimensions():
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
    train_kwargs = Parameters({'alpha': alpha,
                                'epochs': 10,
                                'lr': 1e-1,
                                'num_samples': 10,
                                'batch_size': 1,
                                'trials': 1,
                                'constant': True, 
                                'verbose': True
                            }) 
    trunc_reg = stats.TruncatedLinearRegression(phi_scale, 
                                                train_kwargs, 
                                                noise_var=ch.ones(1, 1))
    trunc_reg.fit(x_trunc, y_trunc_scale)
    w_ = ch.cat([(trunc_reg.coef_).flatten(), trunc_reg.intercept_]) * ch.sqrt(NOISE_VAR)
    print(f'estimated weights: {w_}')
    known_mse_loss = mse_loss(gt_, w_.flatten())
    print(f'known mse loss: {known_mse_loss}')
    msg = f'known mse loss is larger than empirical mse loss. known mse loss is {known_mse_loss}, and empirical mse loss is: {emp_mse_loss}'
    assert known_mse_loss <= emp_mse_loss, msg

# left truncated regression with unknown noise variance in one dimension 
def test_unknown_variance_truncated_regression_one_dimension():
    D, K = 1, 1
    SAMPLES = 1000
    w_ = Uniform(-1, 1)
    M = Uniform(-10, 10)
    # generate ground truth
    noise_var = ch.ones(1, 1)
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
    phi = oracle.Identity()
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
    # phi_emp_scale = oracle.Left_Regression(phi.left / ch.sqrt(emp_noise_var))
    # train algorithm
    train_kwargs = Parameters({
                                'alpha': alpha,
                                'trials': 1,
                                'batch_size': 10,
                                'var_lr': 1e-2, 
                                'constant': False,
                            })
    unknown_trunc_reg = stats.TruncatedLinearRegression(phi,
                                                        train_kwargs)
    unknown_trunc_reg.fit(x_trunc.repeat(1, 1), y_trunc_emp_scale.repeat(1, 1))
    w_ = ch.cat([(unknown_trunc_reg.coef_).flatten(), unknown_trunc_reg.intercept_]) * ch.sqrt(emp_noise_var)
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
    from delphi.grad import TruncatedMSE
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
            loss = TruncatedMSE.apply(
                pred, y_trunc_scale, phi_scale, 
                float(NOISE_VAR), 50
            )
            
            losses[i, j] = loss.item()
            
            # Compute gradients
            loss.backward()
            grad_w[i, j] = w.grad.item() if w.grad is not None else 0.0
            grad_w0[i, j] = w0.grad.item() if w0.grad is not None else 0.0
            
            # Zero gradients for next iteration
            w.grad = None
            w0.grad = None

        test_ws = [0.5, 2.0, 3.5]
    
    for w_val in test_ws:
        w = ch.tensor([[w_val]], requires_grad=True)
        pred = x_trunc @ w.T + w0
        
        # Compute each term separately
        quadratic_term = -0.5 * (y_trunc - pred).pow(2).mean()
        
        # Your 'out' term approximation
        stacked = pred[None, ...].repeat(50, 1, 1)
        import math
        noised = stacked + math.sqrt(NOISE_VAR) * ch.randn(stacked.size())        
        filtered = phi(noised)
        z = (filtered * noised).sum(dim=0) / (filtered.sum(dim=0) + 1e-3)
        out_term = -0.5 * (z.pow(2) + z * pred).mean()
        
        total_loss = quadratic_term + out_term
        
        print(f"w={w_val}: quadratic={quadratic_term.item():.3f}, out_term={out_term.item():.3f}, total={total_loss.item():.3f}")

        test_ws = [0.5, 2.0, 3.5]
    
    for sign in [1, -1]:
        print(f"\n=== Testing out_term sign: {sign} ===")
        
        for w_val in test_ws:
            w = ch.tensor([[w_val]], requires_grad=True)
            pred = x_trunc @ w.T + w0
            
            quadratic_term = -0.5 * (y_trunc - pred).pow(2).mean()
            
            # Your out_term with sign parameter
            stacked = pred[None, ...].repeat(50, 1, 1)
            noised = stacked + math.sqrt(NOISE_VAR) * ch.randn(stacked.size())        
            filtered = phi(noised)
            z = (filtered * noised).sum(dim=0) / (filtered.sum(dim=0) + 1e-3)
            out_term = sign * 0.5 * (z.pow(2) + z * pred).mean()
            
            total_loss = quadratic_term + out_term
            
            print(f"w={w_val}: total={total_loss.item():.3f}")

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
    
    test_ws = [0.5, 2.0, 3.5]
    
    for w_val in test_ws:
        w = ch.tensor([[w_val]], requires_grad=True)
        pred = x_trunc @ w.T + w0
        
        quadratic_term = -0.5 * (y_trunc - pred).pow(2).mean()
        log_trunc_prob = compute_log_truncation_prob(pred, phi, NOISE_VAR)
        out_term = -log_trunc_prob.mean()
        
        total_loss = quadratic_term + out_term
        
        print(f"w={w_val}: quadratic={quadratic_term.item():.3f}, "
              f"out_term={out_term.item():.3f}, total={total_loss.item():.3f}")

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