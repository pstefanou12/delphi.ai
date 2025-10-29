# distribution tests 

import torch as ch
from torch import Tensor
from torch.distributions import MultivariateNormal, Uniform
from torch.distributions.kl import kl_divergence
from torch.distributions.multivariate_normal import _batch_mahalanobis
from torch.linalg import cholesky 
from scipy.linalg import sqrtm

from delphi import distributions 
from delphi import oracle
from delphi.utils.helpers import Parameters, cov, is_psd


class UnNorm_Sphere(oracle.oracle):
    """
    Spherical truncation
    """
    def __init__(self, covariance_matrix, centroid, radius, loc, cov):
        self._unbroadcasted_scale_tril = cholesky(covariance_matrix)
        self.centroid = centroid
        self.radius = radius
        self.loc, self.cov = loc, cov
        self.scale = ch.linalg.cholesky(cov)

    def __call__(self, x):
        x_rescale = x @ self.scale.T + self.loc
        diff = x_rescale - self.centroid
        dist = ch.sqrt(_batch_mahalanobis(self._unbroadcasted_scale_tril, diff))
        return (dist < self.radius).float().flatten()

    def __str__(self): 
        return 'sphere'

# right truncated normal distribution with known truncation and known variance
def test_truncated_normal_known_variance():
    M = MultivariateNormal(ch.zeros(1), ch.eye(1)) 
    samples = M.rsample([1000,])
    print(f'num total samples: {samples.size(0)}')
    # generate ground-truth data
    phi = oracle.Left_Distribution(Tensor([0.0]))
    # truncate
    indices = phi(samples).nonzero()[:,0]
    S = samples[indices]
    print(f'num truncated samples: {S.size(0)}')
    alpha = S.size(0) / samples.size(0)
    emp_loc = S.mean(0)
    print(f"emp loc: {emp_loc}")
    print(f"known variance: {ch.eye(1)}")
    
    S_std_norm = (S - emp_loc) 
    phi_std_norm = oracle.Left_Distribution((phi.left - emp_loc).flatten())
    
    # train algorithm
    args = Parameters({
                        'epochs': 10, 
                        'batch_size': 10, 
                        'trials': 1, 
                        'verbose': True,
                        'lr': 1e-1,
                        'num_samples': 100,
                    }) 
    truncated = distributions.TruncatedNormal(args,
                                              phi_std_norm, 
                                              alpha, 
                                              1,
                                              variance=ch.eye(1))
    truncated.fit(S_std_norm)
    # rescale distribution
    rescale_loc = truncated.loc_ + emp_loc
    print(f"pred loc: {rescale_loc}")
    rescale_var = truncated.variance_
    print(f"pred var: {rescale_var}")
    m = MultivariateNormal(rescale_loc, rescale_var)
        
    # check performance
    kl_truncated = kl_divergence(m, M)
    msg = f'kl divergence between estimated and true underlying distribution is greater than 1e-1. truncated kl divergence is {kl_truncated}'
    assert kl_truncated <= 1e-1, msg

# right truncated normal distribution with known truncation
def test_truncated_normal():
    M = MultivariateNormal(ch.zeros(1), ch.eye(1)) 
    samples = M.rsample([1000,])
    print(f'num total samples: {samples.size(0)}')
    # generate ground-truth data
    phi = oracle.Left_Distribution(Tensor([0.0]))
    # truncate
    indices = phi(samples).nonzero()[:,0]
    S = samples[indices]
    print(f'num truncated samples: {S.size(0)}')
    alpha = S.size(0) / samples.size(0)
    emp_loc = S.mean(0)
    print(f"emp loc: {emp_loc}")
    emp_var = S.var(0)[...,None]
    print(f"emp var: {emp_var}")
    emp_scale = ch.sqrt(emp_var) 

    S_std_norm = (S - emp_loc) / emp_scale
    phi_std_norm = oracle.Left_Distribution(((phi.left - emp_loc) / emp_scale).flatten())
    
    # train algorithm
    args = Parameters({
                        'epochs': 10, 
                        'batch_size': 10, 
                        'trials': 1, 
                        'verbose': True,
                        'lr': 1e-1,
                    }) 
    truncated = distributions.TruncatedNormal(args,
                                              phi_std_norm, 
                                              alpha, 
                                              1)
    truncated.fit(S_std_norm)
    # rescale distribution
    rescale_loc = truncated.loc_ * emp_scale + emp_loc
    print(f"pred loc: {rescale_loc}")
    rescale_var = truncated.variance_ * emp_var
    print(f"pred var: {rescale_var}")
    m = MultivariateNormal(rescale_loc, rescale_var)
        
    # check performance
    kl_truncated = kl_divergence(m, M)
    msg = f'kl divergence between estimated and true underlying distribution is greater than 1e-1. truncated kl divergence is {kl_truncated}'
    assert kl_truncated <= 1e-1, msg

# sphere truncated multivariate normal distribution (10 D) with known truncation and covariance matrix
def test_truncated_multivariate_normal_known_covariance_matrix():
    M = MultivariateNormal(ch.zeros(10), ch.eye(10)) 
    samples = M.rsample([5000,])
    # generate ground-truth data
    alpha = 0
    while alpha < .3: 
        W = Uniform(-.5, .5)
        centroid = W.sample([10,])
        phi = oracle.Sphere(M.covariance_matrix, centroid, 3.0)
        indices = phi(samples).nonzero()[:,0]
        S = samples[indices]
        alpha = S.size(0) / samples.size(0)

    print(f'alpha: {alpha}')
    print(f'num total samples: {samples.size(0)}')
    print(f'num truncated samples: {S.size(0)}')

    emp_loc = S.mean(0)
    print(f'emp loc: {emp_loc}')
    print(f'emp covariance: {M.covariance_matrix}')
    S_norm = (S - emp_loc)
    phi_norm = UnNorm_Sphere(M.covariance_matrix, phi.centroid, phi.radius, emp_loc, M.covariance_matrix)

    # train algorithm
    train_kwargs = Parameters({
                            'phi': phi_norm, 
                            'alpha': alpha,
                            'epochs': 10, 
                            'covariance_matrix': M.covariance_matrix,
                            'trials': 1,
                    }) 
    truncated = distributions.TruncatedMultivariateNormal(train_kwargs)
    truncated.fit(S_norm)
    # rescale distribution
    rescale_loc = truncated.loc_ + emp_loc
    print(f'pred loc: {rescale_loc}')
    print(f'pred covariance matrix: {M.covariance_matrix}')
    m = MultivariateNormal(rescale_loc, truncated.covariance_matrix_)
        
    # check performance
    kl_truncated = kl_divergence(m, M)
    msg = f'kl divergence between estimated and true underlying distribution is greater than 1e-1. truncated kl divergence is {kl_truncated}'
    assert kl_truncated <= 1e-1, msg

# sphere truncated multivariate normal distribution (10 D) with known truncation
def test_truncated_multivariate_normal():
    M = MultivariateNormal(ch.zeros(10), ch.eye(10)) 
    samples = M.rsample([10000,])

    phi, indices, alpha = generate_sphere_truncation(samples, M.covariance_matrix, .5)
    S = samples[indices]
        
    print(f'alpha: {alpha}')
    print(f'num total samples: {samples.size(0)}')
    print(f'num truncated samples: {S.size(0)}')
    emp_loc = S.mean(0)
    emp_cov = cov(S)
    print(f'emp loc: {emp_loc}')
    print(f'emp covariance: {emp_cov}')

    if not is_psd(emp_cov): 
        eigenvalues = ch.linalg.eigvalsh(emp_cov)
        print(f"min eigenvalue: {eigenvalues.min()}")
        print(f"max eigenvalue: {eigenvalues.max()}")
        print(f"condition number: {eigenvalues.max() / eigenvalues.max(0)[0]}")
        print(f'empirical covariance is not PSD!!')

    try:
        emp_scale = ch.linalg.cholesky(emp_cov)
    except RuntimeError:
        print("Empirical covariance not PSD, adding regularization")
        emp_cov = emp_cov + 1e-6 * ch.eye(10)
        emp_scale = ch.linalg.cholesky(emp_cov)
    
    # Standardize: solve L @ S_norm.T = (S - emp_loc).T
    S_norm = ch.linalg.solve_triangular(
        emp_scale, (S - emp_loc).T, upper=False
    ).T

    # S_norm = (S - emp_loc) @ emp_scale.inverse()
    phi_norm = UnNorm_Sphere(M.covariance_matrix, phi.centroid, phi.radius, emp_loc, emp_cov)

    # train algorithm
    train_kwargs = Parameters({
                            'phi': phi_norm, 
                            'alpha': alpha,
                            'epochs': 10, 
                            'trials': 1,
                            'lr': 1e-1,
                            'batch_size': 10
                    }) 
    truncated = distributions.TruncatedMultivariateNormal(train_kwargs)
    truncated.fit(S_norm)
    # rescale distribution
    rescale_loc = truncated.loc_ @ emp_scale + emp_loc
    rescale_cov = emp_scale @ truncated.covariance_matrix_ @ emp_scale.T
    print(f'pred loc: {rescale_loc}')
    print(f'pred covariance matrix: {rescale_cov}')
    m = MultivariateNormal(rescale_loc, rescale_cov)
        
    # check performance
    kl_truncated = kl_divergence(m, M)
    msg = f'kl divergence between estimated and true underlying distribution is greater than 1e-1. truncated kl divergence is {kl_truncated}'
    assert kl_truncated <= 1e-1, msg

def test_unknown_normal_truncation_oracle():
    M = MultivariateNormal(ch.zeros(1), ch.eye(1)) 
    samples = M.rsample([10000,])

    # Generate ground-truth data - truncate to right half
    phi = oracle.Right_Distribution(Tensor([0.0]))
    inside_mask = phi(samples).bool().squeeze()

    S = samples[inside_mask]
    trunc_samples = samples[~inside_mask]
    alpha = S.size(0) / samples.size(0)

    print(f'alpha: {alpha:.3f}')
    print(f'num inside samples: {S.size(0)}')
    print(f'num outside samples: {trunc_samples.size(0)}')

    emp_loc = S.mean(0)
    emp_cov = cov(S)

    print(f'emp loc: {emp_loc}')
    print(f'emp variance: {emp_cov}')

    # Create oracle
    max_degree = 8
    unknown_truncation_oracle = oracle.UnknownGaussian(
        emp_loc, emp_cov, S, max_degree
    )
    unknown_truncation_oracle.dist = M

    # DEBUG: Check Hermite coefficients
    print(f"\nHermite coefficients shape: {unknown_truncation_oracle.C_v.shape}")
    print(f"Hermite coefficients range: [{unknown_truncation_oracle.C_v.min():.4f}, {unknown_truncation_oracle.C_v.max():.4f}]")
    print(f"Number of multi-indices: {len(unknown_truncation_oracle.multi_indices)}")

    # DEBUG: Check what happens with inside samples
    H_inside = unknown_truncation_oracle.H_v(S)
    print(f"\nInside samples - H_v stats:")
    print(f"  H_v mean: {H_inside.mean():.4f}, std: {H_inside.std():.4f}")
    print(f"  H_v min: {H_inside.min():.4f}, max: {H_inside.max():.4f}")

    # Test predictions
    pred_inside = unknown_truncation_oracle.psi_k(S)
    pred_outside = unknown_truncation_oracle.psi_k(trunc_samples)

    print(f'\nInside set psi_k:')
    print(f'  Mean: {pred_inside.mean():.4f} ± {pred_inside.std():.4f}')
    print(f'  Range: [{pred_inside.min():.4f}, {pred_inside.max():.4f}]')
    print(f'  >0 ratio: {(pred_inside > 0).float().mean():.3f}')

    print(f'\nOutside set psi_k:')
    print(f'  Mean: {pred_outside.mean():.4f} ± {pred_outside.std():.4f}')
    print(f'  Range: [{pred_outside.min():.4f}, {pred_outside.max():.4f}]')
    print(f'  >0 ratio: {(pred_outside > 0).float().mean():.3f}')
    print(f'  <0.01 ratio: {(pred_outside < 0.01).float().mean():.3f}')
    print(f'  <0.1 ratio: {(pred_outside < 0.1).float().mean():.3f}')

    # Check separation
    separation = pred_inside.min() - pred_outside.max()
    print(f'\nSeparation: {separation:.4f}')

    # Plot histogram to understand distribution
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(pred_inside.detach().numpy(), bins=50, alpha=0.7, label='Inside')
    plt.hist(pred_outside.detach().numpy(), bins=50, alpha=0.7, label='Outside')
    plt.xlabel('psi_k(x)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Distribution of psi_k values')
    
    plt.subplot(1, 2, 2)
    plt.hist(pred_outside.detach().numpy(), bins=50, alpha=0.7)
    plt.xlabel('psi_k(x)')
    plt.ylabel('Frequency')
    plt.title('Outside samples only (zoomed)')
    
    plt.tight_layout()
    plt.show()

# right truncated 1D normal distribution with unknown truncation and known variance
def test_unknown_truncation_normal_known_variance():
    # generate ground-truth data
    M = MultivariateNormal(ch.zeros(1), ch.eye(1)) 
    samples = M.rsample([1000,])
    phi = oracle.Right_Distribution(Tensor([0.0]))
    # phi = oracle.Identity()
    indices = phi(samples).nonzero()[:,0]
    S = samples[indices]
    alpha = S.size(0) / samples.size(0)
        
    print(f'alpha: {alpha}')
    print(f'num total samples: {samples.size(0)}')
    print(f'num truncated samples: {S.size(0)}')
    emp_loc = S.mean(0)
    emp_var = S.var(0)
    S_norm = S - emp_loc

    print(f'emp loc: {emp_loc}')
    print(f'emp var: {emp_var}')
    print(f'known variance: {M.covariance_matrix}')

    # train algorithm
    train_kwargs = Parameters({
                            'alpha': alpha,
                            'epochs': 2, 
                            'trials': 1,
                            'covariance_matrix': M.covariance_matrix, 
                            'batch_size': 10,
                            'lr': 1e-1
                    }) 
    truncated = distributions.UnknownTruncationNormal(train_kwargs)
    truncated.fit(S_norm)
    # rescale distribution
    import pdb; pdb.set_trace()
    rescale_loc = truncated.loc_ + emp_loc
    print(f'pred loc: {rescale_loc}')
    print(f'pred variance: {truncated.covariance_matrix_}')
    m = MultivariateNormal(rescale_loc, truncated.covariance_matrix_)
                
    # check distribution parameter estimates
    kl_truncated = kl_divergence(m, M)
    self.assertTrue(kl_truncated <= 1e-1) 

# right truncated 1D normal distribution with unknown truncation
def test_unknown_truncation_normal():
    # generate ground-truth data
    M = MultivariateNormal(ch.zeros(1), ch.eye(1)) 
    samples = M.rsample([1000,])
    phi = oracle.Right_Distribution(Tensor([0.0]))
    indices = phi(samples).nonzero()[:,0]
    S = samples[indices]
    alpha = S.size(0) / samples.size(0)
        
    print(f'alpha: {alpha}')
    print(f'num total samples: {samples.size(0)}')
    print(f'num truncated samples: {S.size(0)}')
    emp_loc = S.mean(0)
    emp_var = S.var(0)
    S_norm = S - emp_loc 

    print(f'emp loc: {emp_loc}')
    print(f'emp variance: {emp_var}')

    # train algorithm
    train_kwargs = Parameters({
                            'alpha': alpha,
                            'epochs': 10, 
                            'trials': 1,
                            'batch_size': 10,
                            'lr': 1e-1
                    }) 
    truncated = distributions.UnknownTruncationNormal(train_kwargs)
    truncated.fit(S_norm)
    # rescale distribution
    rescale_loc = truncated.loc_ + emp_loc
    rescale_var = truncated.covariance_matrix_ * emp_var
    print(f'pred loc: {rescale_loc}')
    print(f'pred variance: {rescale_var}')
    import pdb; pdb.set_trace()
    m = MultivariateNormal(rescale_loc, rescale_var)
                
    # check distribution parameter estimates
    kl_truncated = kl_divergence(m, M)
    self.assertTrue(kl_truncated <= 1e-1) 

# 10D sphere truncated multivariate normal distribution with unknown truncation
def test_unknown_truncation_multivariate_normal():
    M = MultivariateNormal(ch.zeros(10), ch.eye(10)) 
    samples = M.rsample([5000,])
    alpha = 0.0
    while alpha < .5:
        # generate ground-truth data
        W = Uniform(-.5, .5)
        centroid = W.sample([10,])
        phi = oracle.Sphere(M.covariance_matrix, centroid, 3.5)
        indices = phi(samples).nonzero(as_tuple=True)
        S = samples[indices]
        alpha = S.size(0) / samples.size(0)

    # train algorithm
    train_kwargs = Parameters({'alpha': alpha,
                            'epochs': 25, 
                            'batch_size': 100}) 
    truncated = distributions.UnknownTruncationMultivariateNormal(train_kwargs)
    truncated.fit(S)
    # rescale distribution
    rescale_loc = truncated.loc_
    rescale_var = truncated.covariance_matrix_
    m = MultivariateNormal(rescale_loc, rescale_var)
        
    # check distribution parameter estimates
    kl_truncated = kl_divergence(m, M)
    self.assertTrue(kl_truncated <= 1e-1)  


def test_truncated_bernoulli(): 
    pass     


def plot_gradient_contours_mu():
    """Plot loss contours and gradients around the true parameters"""
    import matplotlib.pyplot as plt
    import numpy as np
    from delphi.distributions.unknown_truncated_multivariate_normal import Exp_h
    from delphi.grad import UnknownTruncationMultivariateNormalNLL

    true_mu = 1.0
    M = MultivariateNormal(ch.tensor([true_mu]), ch.eye(1))
    
    # Generate data
    samples = M.rsample([1000])
    inside_mask = samples > 0
    S = samples[inside_mask.squeeze()]
    
    # Create grid around true parameters
    mu_range = np.linspace(-1.0, 3.0, 50)  # Around true μ=1.0
    losses = np.zeros((len(mu_range),))
    gradients = np.zeros((len(mu_range),))
    
    print("=== GRADIENT CONTOUR ANALYSIS ===")
    
    for i, test_mu in enumerate(mu_range):
        u = ch.tensor([test_mu], requires_grad=True)
        B = ch.eye(1, requires_grad=False)  # Known covariance
        
        # Your loss computation
        x = samples
        pdf_vals = M.log_prob(x).exp().unsqueeze(1)
        loc_grad = S.mean(0) - x
        cov_grad = ch.zeros_like(loc_grad)
        data = ch.cat([x, pdf_vals, loc_grad, cov_grad], dim=1).float()
        
        # Perfect oracle
        class PerfectOracle:
            def psi_k(self, x):
                return (x > 0).float()
        
        phi = PerfectOracle()
        exp_h = Exp_h(S.mean(0).float(), ch.eye(1).float())
        
        # Compute loss
        params = ch.cat([u.detach().flatten(), B.detach().flatten()]).float()
        params.requires_grad = True
        loss = UnknownTruncationMultivariateNormalNLL.apply(
            params, data, phi, exp_h, 1, True
        )
        
        losses[i] = loss.item()
        
        # Compute gradient
        loss.backward()
        gradients[i] = params.grad[0] 
        u.grad = None  # Reset

    print(f'losses: {losses}')
    print(f'gradients: {gradients}')
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss contour
    ax1.plot(mu_range, losses)
    ax1.axvline(x=true_mu, color='r', linestyle='--', label=f'True μ={true_mu}')
    ax1.axvline(x=S.mean().item(), color='g', linestyle='--', label=f'Empirical mean={S.mean().item():.2f}')
    ax1.set_xlabel('μ')
    ax1.set_ylabel('Negative Log Likelihood')
    ax1.set_title('Loss Landscape')
    ax1.legend()
    ax1.grid(True)
    
    # Gradient field
    ax2.plot(mu_range, gradients)
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax2.axvline(x=true_mu, color='r', linestyle='--', label=f'True μ={true_mu}')
    ax2.set_xlabel('μ')
    ax2.set_ylabel('Gradient')
    ax2.set_title('Gradient Field')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print key insights
    print(f"\nKey Observations:")
    print(f"Minimum loss at μ = {mu_range[np.argmin(losses)]:.3f}")
    print(f"Gradient at true μ={true_mu}: {gradients[np.argmin(np.abs(mu_range - true_mu))]:.4f}")
    print(f"Gradient should be ZERO at optimum")
    
    # Check gradient signs
    left_of_opt = mu_range < true_mu
    if np.any(left_of_opt):
        avg_grad_left = np.mean(gradients[left_of_opt])
        print(f"Average gradient left of optimum: {avg_grad_left:.4f} (should be NEGATIVE)")
    
    right_of_opt = mu_range > true_mu  
    if np.any(right_of_opt):
        avg_grad_right = np.mean(gradients[right_of_opt])
        print(f"Average gradient right of optimum: {avg_grad_right:.4f} (should be POSITIVE)") 

def plot_gradient_contours_var():
    """Plot loss contours and gradients around the true parameters"""
    import matplotlib.pyplot as plt
    import numpy as np
    from delphi.distributions.unknown_truncated_multivariate_normal import Exp_h
    from delphi.grad import UnknownTruncationMultivariateNormalNLL

    true_mu = 1.0
    M = MultivariateNormal(ch.tensor([true_mu]), ch.eye(1))
    
    # Generate data
    samples = M.rsample([1000])
    inside_mask = samples > 0
    S = samples[inside_mask.squeeze()]
    
    # Create grid around true parameters
    var_range = np.linspace(.1, 3.0, 50)  # Around true var=1.0
    losses = np.zeros((len(var_range),))
    gradients = np.zeros((len(var_range),))
    
    print("=== GRADIENT CONTOUR ANALYSIS ===")
    
    for i, test_var in enumerate(var_range):
        u = ch.tensor([true_mu], requires_grad=False)
        B = ch.tensor([test_var], requires_grad=True)
        
        # Your loss computation
        x = samples
        pdf_vals = M.log_prob(S).exp().unsqueeze(1)
        loc_grad = ch.zeros_like(S) 
        # import pdb; pdb.set_trace()
        cov_grad = .5 * (ch.bmm(S.unsqueeze(2), S.unsqueeze(1)) - S.var(0) - u[...,None] @ u[None,...]).flatten(1)
        data = ch.cat([S, pdf_vals, loc_grad, cov_grad], dim=1).float()
        
        # Perfect oracle
        class PerfectOracle:
            def psi_k(self, x):
                return (x > 0).float()
        
        phi = PerfectOracle()
        exp_h = Exp_h(u, S.var(0).float())
        
        # Compute loss
        params = ch.cat([u.detach().flatten(), B.detach().flatten()]).float()
        params.requires_grad = True
        loss = UnknownTruncationMultivariateNormalNLL.apply(
            params, data, phi, exp_h, 1, False 
        )
        
        losses[i] = loss.item()
        
        # Compute gradient
        loss.backward()
        gradients[i] = params.grad[1] 
        u.grad = None  # Reset

    print(f'losses: {losses}')
    print(f'gradients: {gradients}')
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss contour
    ax1.plot(var_range, losses)
    ax1.axvline(x=true_mu, color='r', linestyle='--', label=f'True var={true_mu}')
    ax1.axvline(x=S.mean().item(), color='g', linestyle='--', label=f'Empirical Variance={S.var().item():.2f}')
    ax1.set_xlabel('/sigma')
    ax1.set_ylabel('Negative Log Likelihood')
    ax1.set_title('Loss Landscape')
    ax1.legend()
    ax1.grid(True)
    
    # Gradient field
    ax2.plot(var_range, gradients)
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax2.axvline(x=true_mu, color='r', linestyle='--', label=f'True var={1.0}')
    ax2.set_xlabel('var')
    ax2.set_ylabel('Gradient')
    ax2.set_title('Gradient Field')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print key insights
    print(f"\nKey Observations:")
    print(f"Minimum loss at var = {var_range[np.argmin(losses)]:.3f}")
    print(f"Gradient at true var={true_mu}: {gradients[np.argmin(np.abs(var_range - true_mu))]:.4f}")
    print(f"Gradient should be ZERO at optimum")
    
    # Check gradient signs
    left_of_opt = var_range < 1.0 
    if np.any(left_of_opt):
        avg_grad_left = np.mean(gradients[left_of_opt])
        print(f"Average gradient left of optimum: {avg_grad_left:.4f} (should be NEGATIVE)")
    
    right_of_opt = var_range > 1.0  
    if np.any(right_of_opt):
        avg_grad_right = np.mean(gradients[right_of_opt])
        print(f"Average gradient right of optimum: {avg_grad_right:.4f} (should be POSITIVE)") 


def debug_exponential_dependence():
    """Check how exp_h depends on u"""
    import matplotlib.pyplot as plt
    import numpy as np
    from delphi.distributions.unknown_truncated_multivariate_normal import Exp_h
    # from delphi.grad import UnknownTruncationMultivariateNormalNLL

    true_mu = 1.0
    M = MultivariateNormal(ch.tensor([true_mu]), ch.eye(1))
    samples = M.rsample([100])
    inside_mask = samples > 0
    S = samples[inside_mask.squeeze()]
    
    emp_loc = S.mean(0)
    emp_cov = cov(S)
    
    exp_h = Exp_h(emp_loc, emp_cov)
    B = ch.eye(1)
    
    print("=== EXPONENTIAL DEPENDENCE ON u ===")
    
    test_us = np.linspace(-1.0, 3.0, 20)
    exp_means = []
    
    for u_val in test_us:
        # import pdb; pdb.set_trace()
        u = ch.tensor([u_val]).float()
        exp_vals = exp_h(u, B, samples)
        exp_means.append(exp_vals.mean().item())
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(test_us, exp_means)
    plt.axvline(x=true_mu, color='r', linestyle='--', label='True μ')
    plt.xlabel('u')
    plt.ylabel('exp_h mean')
    plt.title('Exponential vs u')
    plt.legend()

    print(f'exp terms: {exp_means}')
    
    # Check if exponential has a maximum at the right place
    max_idx = np.argmax(exp_means)
    print(f"Exponential maximum at u = {test_us[max_idx]:.3f}")
    print(f"True μ = {true_mu}")
    
    plt.subplot(1, 2, 2)
    # Plot the loc_term component: uᵀ(x - μ̃_S)
    loc_terms = []
    for u_val in test_us:
        u = ch.tensor([u_val]).float()
        loc_term = (u @ (samples - emp_loc).T).mean().item()
        loc_terms.append(loc_term)

    print(f'loc terms: {loc_terms}')

        
    plt.plot(test_us, loc_terms)
    plt.axvline(x=true_mu, color='r', linestyle='--', label='True μ')
    plt.xlabel('u')
    plt.ylabel('loc_term mean')
    plt.title('loc_term vs u')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def debug_exponential_dependence_var():
    """Check how exp_h depends on B"""
    import matplotlib.pyplot as plt
    import numpy as np
    from delphi.distributions.unknown_truncated_multivariate_normal import Exp_h
    # from delphi.grad import UnknownTruncationMultivariateNormalNLL

    true_mu = 1.0
    M = MultivariateNormal(ch.tensor([true_mu]), ch.eye(1))
    samples = M.rsample([100])
    inside_mask = samples > 0
    S = samples[inside_mask.squeeze()]
    
    emp_loc = S.mean(0)
    emp_cov = cov(S)
    
    exp_h = Exp_h(emp_loc, emp_cov)
    u = ch.eye(1)
    
    print("=== EXPONENTIAL DEPENDENCE ON u ===")
    
    test_bs = np.linspace(-1.0, 3.0, 20)
    exp_vars = []
    
    for b_val in test_bs:
        # import pdb; pdb.set_trace()
        B = ch.tensor([b_val]).float()[...,None].inverse()
        exp_vals = exp_h(u, B, samples)
        exp_vars.append(exp_vals.mean().item())
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(test_bs, exp_vars)
    plt.axvline(x=1.0, color='r', linestyle='--', label='True var')
    plt.xlabel('b')
    plt.ylabel('exp_h var')
    plt.title('Exponential vs B')
    plt.legend()

    print(f'exp terms: {exp_vars}')
    
    # Check if exponential has a maximum at the right place
    max_idx = np.argmax(exp_vars)
    print(f"Exponential maximum at B = {test_bs[max_idx]:.3f}")
    print(f"True var = {true_mu}")
    
    plt.subplot(1, 2, 2)
    # Plot the loc_term component: uᵀ(x - μ̃_S)
    loc_terms = []
    for b_val in test_bs:
        b = ch.tensor([b_val]).float()
        loc_term = (b @ (samples - emp_loc).T).mean().item()
        loc_terms.append(loc_term)

    print(f'loc terms: {loc_terms}')

        
    plt.plot(test_bs, loc_terms)
    plt.axvline(x=1.0, color='r', linestyle='--', label='True var')
    plt.xlabel('B')
    plt.ylabel('cov_term var')
    plt.title('cov_term vs B')
    plt.legend()
    
    plt.tight_layout()
    plt.show()


def debug_loss_component_scales():
    """Check the relative magnitudes of each loss component"""
    # import matplotlib.pyplot as plt
    # import numpy as np
    from delphi.distributions.unknown_truncated_multivariate_normal import Exp_h
    true_mu = 1.0
    M = MultivariateNormal(ch.tensor([true_mu]), ch.eye(1))
    samples = M.rsample([100])
    inside_mask = samples > 0
    S = samples[inside_mask.squeeze()]
    
    emp_loc = S.mean(0)
    emp_cov = cov(S)
    
    exp_h = Exp_h(emp_loc, emp_cov)
    B = ch.eye(1)
    
    print("=== LOSS COMPONENT SCALES ===")
    
    test_us = [0.5, 1.0, 1.5]  # Around true mean
    
    for u_val in test_us:
        u = ch.tensor([u_val])
        
        # Compute each component
        exp_vals = exp_h(u, B, samples)
        pdf_vals = M.log_prob(samples).exp().unsqueeze(1)
        psi_vals = (samples > 0).float()
        
        print(f"\nu = {u_val:.1f}:")
        print(f"  exp_vals:   mean={exp_vals.mean().item():.6f}, std={exp_vals.std().item():.6f}")
        print(f"  pdf_vals:   mean={pdf_vals.mean().item():.6f}, std={pdf_vals.std().item():.6f}")
        print(f"  psi_vals:   mean={psi_vals.mean().item():.6f}")
        
        # Check product
        product = exp_vals * pdf_vals * psi_vals
        print(f"  product:    mean={product.mean().item():.6f}")
        
        # Check ratios
        exp_pdf_ratio = exp_vals.mean() / pdf_vals.mean()
        print(f"  exp/pdf ratio: {exp_pdf_ratio.item():.6f}")

def debug_variance_dependence():
    """Check which terms cause monotonic variance dependence"""
    true_mu = 1.0
    true_var = 1.0
    M = MultivariateNormal(ch.tensor([true_mu]), ch.tensor([[true_var]]))
    samples = M.rsample([100])
    inside_mask = samples > 0
    S = samples[inside_mask.squeeze()]
    
    emp_loc = S.mean(0)
    emp_cov = cov(S)
    
    u = ch.tensor([true_mu])  # Fixed at true mean
    import numpy as np 
    test_vars = np.linspace(0.1, 3.0, 10)
    
    print("=== VARIANCE DEPENDENCE DEBUG ===")
    
    for var_val in test_vars:
        B = ch.tensor([[var_val]]).float().inverse()  # Test different variances
        
        # Compute each term separately
        cov_term = ((samples @ B @ samples.T).diag() / 2.0)
        # import pdb; pdb.set_trace()
        trace_arg = (B - ch.eye(1)) @ (emp_cov + ch.eye(1).flatten().outer(ch.eye(1).flatten()))
        # trace_arg = (B - emp_cov) @ (emp_cov + u.outer(u))
        trace_term = ch.trace(trace_arg) / 2.0
        loc_term = ((samples - ch.eye(1)) @ ch.eye(1))
        pi_const = 0.5 * ch.log(2.0 * ch.pi * ch.ones(1))
        
        exp_val = ch.exp(cov_term - trace_term - loc_term + pi_const).mean()
        
        print(f"var={var_val:.2f}: exp={exp_val.item():.3f}") 

def debug_variance_dependence_two():
    """Check convexity/monotonicity of objective w.r.t. variance"""
    true_mu = 1.0
    true_var = 1.0
    M = MultivariateNormal(ch.tensor([true_mu]), ch.tensor([[true_var]]))
    samples = M.rsample([1000])  # More samples
    inside_mask = samples > 0
    S = samples[inside_mask.squeeze()]

    emp_loc = S.mean(0, keepdim=True)  # Shape: (1,)
    emp_cov = cov(S)  # Shape: (1, 1)
    
    print(f"emp_loc: {emp_loc.item():.3f}")
    print(f"emp_cov: {emp_cov.item():.3f}")

    u = emp_loc  # Or use true_mu, but should be consistent
    test_vars = ch.linspace(0.1, 3.0, 20)

    print("\n=== VARIANCE DEPENDENCE DEBUG ===")
    print(f"{'Variance':<10} {'Exp Value':<12} {'Objective':<12}")
    print("-" * 40)

    exp_values = []
    objectives = []
    
    for var_val in test_vars:
        B = (1.0 / var_val) * ch.eye(1)  # B = Σ^{-1}
        
        # Term 1: x^T B x / 2 (per sample)
        quadratic_term = 0.5 * (S @ B @ S.T).diag()
        
        # Term 2: -tr((B-I)(Σ_S + μ_S μ_S^T)) / 2 (constant across samples)
        trace_term = -0.5 * ch.trace(
            (B - ch.eye(1)) @ (emp_cov + emp_loc.T @ emp_loc)
        )
        
        # Term 3: -u^T(x - μ_S) (per sample)
        linear_term = -(u @ (S - emp_loc).T).squeeze()
        
        # Term 4: d/2 log(2π) (constant)
        const_term = 0.5 * ch.log(2.0 * ch.tensor(ch.pi))
        
        # Full h(u, B; x) for each sample
        h_values = quadratic_term + trace_term + linear_term + const_term
        
        # Expected value of exp(h)
        exp_h = ch.exp(h_values)
        mean_exp_h = exp_h.mean().item()
        
        exp_values.append(mean_exp_h)
        
        # For objective, we also need N(0,I;x) term
        N0_x = ch.exp(-0.5 * S.pow(2).sum(1)) / ch.sqrt(2 * ch.tensor(ch.pi))
        objective_val = (exp_h * N0_x).mean().item()
        objectives.append(objective_val)
        
        print(f"{var_val.item():<10.2f} {mean_exp_h:<12.4f} {objective_val:<12.4f}")
    
    # Check for convexity by looking at second differences
    print("\n=== CONVEXITY CHECK ===")
    exp_values = ch.tensor(exp_values)
    first_diff = exp_values[1:] - exp_values[:-1]
    second_diff = first_diff[1:] - first_diff[:-1]
    
    print(f"First differences (should change sign for non-monotonic):")
    print(f"  Range: [{first_diff.min():.4f}, {first_diff.max():.4f}]")
    print(f"  All positive (monotonic increasing): {(first_diff > 0).all()}")
    print(f"  All negative (monotonic decreasing): {(first_diff < 0).all()}")
    
    print(f"\nSecond differences (should be >0 for convex, <0 for concave):")
    print(f"  Mean: {second_diff.mean():.6f}")
    print(f"  All positive (convex): {(second_diff > 0).all()}")
    print(f"  All negative (concave): {(second_diff < 0).all()}")
    
    # Plot if matplotlib available
    try:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(test_vars.numpy(), exp_values.numpy(), 'b-o')
        ax1.set_xlabel('Variance')
        ax1.set_ylabel('E[exp(h)]')
        ax1.set_title('Expected Value of Exponential Term')
        ax1.grid(True)
        
        ax2.plot(test_vars.numpy(), objectives, 'r-o')
        ax2.set_xlabel('Variance')
        ax2.set_ylabel('Objective Value')
        ax2.set_title('Full Objective Function')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('variance_dependence.png')
        print("\nPlot saved to variance_dependence.png")
    except ImportError:
        pass

def generate_sphere_truncation(samples, covariance_matrix, target_alpha=0.5):
    """Generate spherical truncation with target retention rate."""
    # Sample centroid
    centroid = ch.randn(10) * 0.5
    
    # Binary search for radius
    low, high = 0.1, 10.0
    for _ in range(20):
        radius = (low + high) / 2
        phi = oracle.Sphere(covariance_matrix, centroid, radius)
        indices = phi(samples).nonzero()[:,0]
        alpha = len(indices) / len(samples)
        
        if alpha < target_alpha:
            low = radius
        else:
            high = radius
            
        if abs(alpha - target_alpha) < 0.05:
            return phi, indices, alpha
    
    return phi, indices, alpha  