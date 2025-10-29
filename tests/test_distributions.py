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
    samples = M.rsample([10000,])
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

def test_left_truncated_normal():
    """Test with left truncation at x=0 for N(μ, 1)"""
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    from scipy.special import erf
    
    torch.manual_seed(42)
    
    # Generate data from left-truncated N(0, 1) at x=0
    # Use rejection sampling
    true_mu = 0.0
    true_sigma = 1.0
    truncation_point = 0.0
    
    samples = []
    while len(samples) < 100:
        candidate = torch.randn(1000) * true_sigma + true_mu
        valid = candidate[candidate > truncation_point]
        samples.extend(valid.tolist())
    
    data = torch.tensor(samples[:100]).unsqueeze(1)
    
    # Left truncation oracle: phi(x) = 1 if x > 0, else 0
    left_truncation_phi = lambda x: (x > truncation_point).squeeze()
    
    print("=== LEFT TRUNCATED NORMAL TEST ===")
    print(f"True distribution: N(0, 1) truncated at x > 0")
    print(f"Empirical mean of samples: {data.mean().item():.6f}")
    print(f"Expected mean (for truncated N(0,1) at 0): ~0.798")
    print()
    
    def simple_censored_nll(z):
        if z.dim() == 1:
            z = z.unsqueeze(-1)
        zzT = -0.5 * torch.bmm(z.unsqueeze(2), z.unsqueeze(1))
        z_flat = z.reshape(z.size(0), -1)
        return torch.cat([zzT.reshape(z.size(0), -1), z_flat], dim=1)
    
    T_fixed = 1.0
    mu_values = np.linspace(-1.0, 1.5, 50)
    your_nll_values = []
    theoretical_nll_values = []
    
    print("Computing loss surface...")
    for mu_val in mu_values:
        nu_val = T_fixed * mu_val
        params = torch.tensor([T_fixed, nu_val])
        
        # Your implementation
        nll = TruncatedMultivariateNormalNLL.apply(
            params.float(), 
            torch.cat([simple_censored_nll(data), data], 1),
            left_truncation_phi, 
            1, 
            simple_censored_nll, 
            None, 
            10000  # More samples for better truncation estimate
        )
        your_nll_values.append(nll.item())
        
        # Theoretical NLL for truncated normal
        # For N(μ, σ²) truncated at a:
        # log p(x) = log(φ((x-μ)/σ)) - log(σ) - log(Φ((μ-a)/σ))
        # where φ is normal PDF, Φ is normal CDF
        
        # NLL = -log p(x) for each x, averaged over batch
        alpha = (truncation_point - mu_val) / true_sigma  # Standardized truncation point
        
        # For each data point x:
        nll_sum = 0
        for x in data.numpy().flatten():
            # PDF term: log(φ((x-μ)/σ)) - log(σ)
            pdf_term = -0.5 * np.log(2 * np.pi) - np.log(true_sigma) - 0.5 * ((x - mu_val) / true_sigma) ** 2
            
            # Normalization term: -log(Φ((-a+μ)/σ)) = -log(1 - Φ((a-μ)/σ))
            normalization = -np.log(1 - norm.cdf(alpha))
            
            nll_sum += -(pdf_term + normalization)
        
        theoretical_nll_values.append(nll_sum / len(data))
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Loss curves
    axes[0].plot(mu_values, your_nll_values, 'b-', linewidth=2, label='Your NLL')
    axes[0].plot(mu_values, theoretical_nll_values, 'r--', linewidth=2, label='Theoretical NLL')
    axes[0].axvline(x=true_mu, color='g', linestyle=':', alpha=0.7, label=f'True μ={true_mu}')
    
    your_min_idx = np.argmin(your_nll_values)
    your_min_mu = mu_values[your_min_idx]
    axes[0].axvline(x=your_min_mu, color='b', linestyle=':', alpha=0.7, 
                    label=f'Your min μ={your_min_mu:.3f}')
    
    theory_min_idx = np.argmin(theoretical_nll_values)
    theory_min_mu = mu_values[theory_min_idx]
    axes[0].axvline(x=theory_min_mu, color='r', linestyle=':', alpha=0.7, 
                    label=f'Theory min μ={theory_min_mu:.3f}')
    
    axes[0].set_xlabel('μ', fontsize=12)
    axes[0].set_ylabel('NLL', fontsize=12)
    axes[0].set_title('Left Truncated N(0,1) at x>0', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Difference
    difference = np.array(your_nll_values) - np.array(theoretical_nll_values)
    axes[1].plot(mu_values, difference, 'purple', linewidth=2)
    axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('μ', fontsize=12)
    axes[1].set_ylabel('Your NLL - Theoretical NLL', fontsize=12)
    axes[1].set_title('Difference (should be ~constant or small)', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Acceptance rate vs μ
    acceptance_rates = []
    for mu_val in mu_values:
        # Theoretical acceptance rate for N(μ, 1) truncated at 0
        alpha = (truncation_point - mu_val) / true_sigma
        acceptance = 1 - norm.cdf(alpha)
        acceptance_rates.append(acceptance)
    
    axes[2].plot(mu_values, acceptance_rates, 'green', linewidth=2)
    axes[2].set_xlabel('μ', fontsize=12)
    axes[2].set_ylabel('P(x > 0)', fontsize=12)
    axes[2].set_title('Acceptance Rate vs μ', fontsize=14)
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='50%')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('left_truncated_test.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n=== RESULTS ===")
    print(f"True μ (data generated from): {true_mu:.3f}")
    print(f"MLE should be near: {theory_min_mu:.3f}")
    print(f"Your minimum at: {your_min_mu:.3f}")
    print(f"Error: {abs(your_min_mu - theory_min_mu):.3f}")
    print()
    
    # Check difference statistics
    diff_mean = np.mean(difference)
    diff_std = np.std(difference)
    print(f"Difference mean: {diff_mean:.6f}")
    print(f"Difference std: {diff_std:.6f}")
    
    if diff_std < 0.1 and abs(your_min_mu - theory_min_mu) < 0.2:
        print("\n✓ SUCCESS! Truncation handling appears correct!")
    else:
        print("\n✗ Issues remain with truncation handling")
    
    return your_min_mu, theory_min_mu


def test_what_term2_should_be():
    """Figure out exactly what term2 should contain"""
    import torch
    from scipy.stats import norm
    import numpy as np
    
    # Test case
    x = 1.0
    mu_val = 0.2
    sigma_sq = 1.0
    truncation = 0.0
    
    T = 1.0 / sigma_sq
    nu = mu_val / sigma_sq
    
    print("=== WHAT SHOULD TERM2 BE? ===")
    print(f"x={x}, μ={mu_val}, σ²={sigma_sq}, truncation at {truncation}\n")
    
    # Full theoretical NLL for truncated normal
    alpha = (truncation - mu_val) / np.sqrt(sigma_sq)
    acceptance_prob = 1 - norm.cdf(alpha)
    
    nll_theory = (-np.log(norm.pdf(x, mu_val, np.sqrt(sigma_sq))) + 
                  np.log(acceptance_prob))
    
    print(f"Theoretical NLL: {nll_theory:.6f}")
    
    # Break it down
    pdf_part = -np.log(norm.pdf(x, mu_val, np.sqrt(sigma_sq)))
    truncation_part = np.log(acceptance_prob)
    
    print(f"  PDF part: {pdf_part:.6f}")
    print(f"  = -log(φ((x-μ)/σ)) + log(σ)")
    print(f"  = 0.5*log(2π) + 0.5*(x-μ)²/σ² + log(σ)")
    
    expanded_pdf = 0.5*np.log(2*np.pi) + 0.5*(x-mu_val)**2/sigma_sq + 0.5*np.log(sigma_sq)
    print(f"  = {expanded_pdf:.6f}")
    
    print(f"\n  Truncation part: -log(P(X > {truncation})) = {-truncation_part:.6f}")
    print(f"  So we ADD: log(P(X > {truncation})) = {truncation_part:.6f}")
    
    # Now in parameterized form
    print(f"\n  Expanding (x-μ)²/σ²:")
    print(f"  = x²/σ² - 2xμ/σ² + μ²/σ²")
    print(f"  = x²T - 2xν + ν²/T")
    
    term1 = 0.5 * x**2 * T - x * nu
    print(f"\n  term1 (0.5*x²T - xν): {term1:.6f}")
    
    # What should term2 be?
    # NLL = term1 + [0.5*ν²/T + 0.5*log(σ²) + 0.5*log(2π) - log(acceptance)]
    #     = term1 + [0.5*ν²/T - 0.5*log(T) + 0.5*log(2π) - log(acceptance)]
    
    term2_should_be = (0.5 * nu**2 / T - 
                       0.5 * np.log(T) + 
                       0.5 * np.log(2 * np.pi) - 
                       np.log(acceptance_prob))
    
    print(f"\n  term2 should be: {term2_should_be:.6f}")
    print(f"    = 0.5*ν²/T - 0.5*log(T) + 0.5*log(2π) - log(acceptance)")
    print(f"    = {0.5 * nu**2 / T:.6f} - {0.5 * np.log(T):.6f} + {0.5 * np.log(2*np.pi):.6f} - {np.log(acceptance_prob):.6f}")
    
    total = term1 + term2_should_be
    print(f"\n  Total NLL: {total:.6f}")
    print(f"  Theoretical: {nll_theory:.6f}")
    print(f"  Match: {abs(total - nll_theory) < 1e-6}")
    
    # Now what are YOU computing?
    print(f"\n=== WHAT YOU'RE COMPUTING ===")
    v_torch = torch.tensor([nu])
    T_torch = torch.tensor([[T]])
    sigma_torch = T_torch.inverse()
    
    your_term2 = (0.5 * (v_torch @ sigma_torch @ v_torch).item() - 
                  0.5 * torch.logdet(T_torch).item() + 
                  0.5 * np.log(2 * np.pi) - 
                  np.log(acceptance_prob))
    
    print(f"  Your term2: {your_term2:.6f}")
    print(f"  Should be: {term2_should_be:.6f}")
    print(f"  Match: {abs(your_term2 - term2_should_be) < 1e-6}")

def test_correct_truncated_nll():
    """Correctly compute truncated NLL"""
    import numpy as np
    from scipy.stats import norm
    
    x = 1.0
    mu_val = 0.2
    sigma_sq = 1.0
    truncation = 0.0
    
    print("=== CORRECT TRUNCATED NLL CALCULATION ===")
    print(f"x={x}, μ={mu_val}, σ²={sigma_sq}, truncation at x > {truncation}\n")
    
    # The truncated normal PDF is:
    # p(x | x > a) = φ((x-μ)/σ) / [σ · Φ((∞-μ)/σ - (a-μ)/σ)]
    #              = φ((x-μ)/σ) / [σ · (1 - Φ((a-μ)/σ))]
    #              = φ((x-μ)/σ) / [σ · P(X > a)]
    
    # So NLL = -log p(x | x > a) 
    #        = -log φ((x-μ)/σ) + log(σ) + log(P(X > a))
    
    # But wait, P(X > a) is less than 1, so log(P(X > a)) is NEGATIVE
    # This means truncation DECREASES the NLL? That doesn't make sense...
    
    # Let me reconsider. The truncated density is:
    # p(x | x ∈ S) = p(x) / P(X ∈ S)  for x ∈ S
    
    # So: log p(x | x ∈ S) = log p(x) - log P(X ∈ S)
    # And: NLL = -log p(x | x ∈ S) = -log p(x) + log P(X ∈ S)
    
    # But P(X ∈ S) < 1, so log P(X ∈ S) < 0
    # So NLL_truncated = NLL_untruncated + log P(X ∈ S) < NLL_untruncated
    
    # Hmm, that means truncation reduces NLL? Let me verify with scipy...
    
    from scipy.stats import truncnorm
    
    # For left truncation at a, scipy uses: (a - mu) / sigma as lower bound
    a_standard = (truncation - mu_val) / np.sqrt(sigma_sq)
    b_standard = np.inf  # no upper truncation
    
    truncated_dist = truncnorm(a_standard, b_standard, loc=mu_val, scale=np.sqrt(sigma_sq))
    
    nll_truncated = -truncated_dist.logpdf(x)
    nll_untruncated = -norm.logpdf(x, mu_val, np.sqrt(sigma_sq))
    
    print(f"Scipy truncated NLL: {nll_truncated:.6f}")
    print(f"Scipy untruncated NLL: {nll_untruncated:.6f}")
    print(f"Difference: {nll_truncated - nll_untruncated:.6f}")
    
    # Manual calculation
    phi_x = norm.pdf(x, mu_val, np.sqrt(sigma_sq))
    Phi_acceptance = 1 - norm.cdf(a_standard)  # P(X > a) for standardized
    
    print(f"\nManual calculation:")
    print(f"  φ(x): {phi_x:.6f}")
    print(f"  P(X ∈ S): {Phi_acceptance:.6f}")
    print(f"  p_truncated(x) = φ(x) / P(X ∈ S) = {phi_x / Phi_acceptance:.6f}")
    print(f"  NLL = -log(p_truncated) = {-np.log(phi_x / Phi_acceptance):.6f}")
    print(f"      = -log(φ(x)) + log(P(X ∈ S))")
    print(f"      = {-np.log(phi_x):.6f} + {np.log(Phi_acceptance):.6f}")
    print(f"      = {-np.log(phi_x) + np.log(Phi_acceptance):.6f}")
    
    # Now in your parameterization
    T = 1.0 / sigma_sq
    nu = mu_val / sigma_sq
    
    print(f"\nIn your parameterization (T={T}, ν={nu}):")
    term1 = 0.5 * x**2 * T - x * nu
    print(f"  term1 = {term1:.6f}")
    
    # The untruncated NLL is:
    # 0.5*log(2π) + 0.5*log(σ²) + 0.5*(x-μ)²/σ²
    # = 0.5*log(2π) - 0.5*log(T) + 0.5*x²T - x*ν + 0.5*ν²/T
    # = [0.5*x²T - x*ν] + [0.5*ν²/T - 0.5*log(T) + 0.5*log(2π)]
    
    # The truncated NLL adds: -log(P(X ∈ S))
    # Since P(X ∈ S) < 1, -log(P(X ∈ S)) > 0
    
    term2_correct = (0.5 * nu**2 / T - 
                     0.5 * np.log(T) + 
                     0.5 * np.log(2 * np.pi) - 
                     np.log(Phi_acceptance))  # MINUS log of acceptance!
    
    print(f"  term2 = {term2_correct:.6f}")
    print(f"  total = {term1 + term2_correct:.6f}")
    print(f"  scipy = {nll_truncated:.6f}")
    print(f"  match = {abs(term1 + term2_correct - nll_truncated) < 1e-5}")


def test_final_breakdown():
    """Final correct breakdown"""
    import numpy as np
    from scipy.stats import norm
    
    x = 1.0
    mu_val = 0.2
    sigma_sq = 1.0
    truncation = 0.0
    T = 1.0
    nu = 0.2
    
    print("=== FINAL CORRECT BREAKDOWN ===\n")
    
    # The Gaussian PDF is:
    # φ(x) = (2πσ²)^(-1/2) exp(-0.5(x-μ)²/σ²)
    # -log φ(x) = 0.5*log(2πσ²) + 0.5*(x-μ)²/σ²
    
    log_2pi_sigma = 0.5 * np.log(2 * np.pi * sigma_sq)
    quadratic = 0.5 * (x - mu_val)**2 / sigma_sq
    
    print(f"-log φ(x) = {log_2pi_sigma:.6f} + {quadratic:.6f} = {log_2pi_sigma + quadratic:.6f}")
    
    # Expanding the quadratic:
    # 0.5*(x-μ)²/σ² = 0.5*x²/σ² - x*μ/σ² + 0.5*μ²/σ²
    #               = 0.5*x²*T - x*ν + 0.5*ν²/T
    
    term_half_x2T = 0.5 * x**2 * T
    term_xnu = x * nu
    term_half_nu2T = 0.5 * nu**2 / T
    
    print(f"\nExpanding quadratic:")
    print(f"  0.5*x²*T = {term_half_x2T:.6f}")
    print(f"  -x*ν = {-term_xnu:.6f}")
    print(f"  0.5*ν²/T = {term_half_nu2T:.6f}")
    print(f"  Sum = {term_half_x2T - term_xnu + term_half_nu2T:.6f}")
    
    # So -log φ(x) = 0.5*log(2π) + 0.5*log(σ²) + 0.5*x²*T - x*ν + 0.5*ν²/T
    #              = 0.5*log(2π) - 0.5*log(T) + 0.5*x²*T - x*ν + 0.5*ν²/T
    
    untruncated_nll = (0.5 * np.log(2 * np.pi) - 
                       0.5 * np.log(T) + 
                       0.5 * x**2 * T - 
                       x * nu + 
                       0.5 * nu**2 / T)
    
    print(f"\nUntruncated NLL = {untruncated_nll:.6f}")
    print(f"Should match: {norm.logpdf(x, mu_val, np.sqrt(sigma_sq)):.6f}")
    print(f"Actually: {-norm.logpdf(x, mu_val, np.sqrt(sigma_sq)):.6f}")
    
    # For truncated, add -log(P(X ∈ S))
    acceptance = 1 - norm.cdf((truncation - mu_val) / np.sqrt(sigma_sq))
    truncation_correction = -np.log(acceptance)
    
    print(f"\nTruncation correction: -log(P(X>0)) = -log({acceptance:.6f}) = {truncation_correction:.6f}")
    
    truncated_nll = untruncated_nll + truncation_correction
    print(f"\nTruncated NLL = {truncated_nll:.6f}")
    print(f"Scipy says: {-norm(mu_val, np.sqrt(sigma_sq)).logpdf(x) + truncation_correction:.6f}")
    
    # Now split into term1 and term2
    print(f"\n=== SPLITTING INTO TERM1 AND TERM2 ===")
    term1 = 0.5 * x**2 * T - x * nu
    print(f"term1 (0.5*x²*T - x*ν) = {term1:.6f}")
    
    term2 = (0.5 * nu**2 / T - 
             0.5 * np.log(T) + 
             0.5 * np.log(2 * np.pi) + 
             truncation_correction)
    print(f"term2 (rest) = {term2:.6f}")
    print(f"  = 0.5*ν²/T - 0.5*log(T) + 0.5*log(2π) - log(acceptance)")
    print(f"  = {0.5 * nu**2 / T:.6f} - {0.5 * np.log(T):.6f} + {0.5 * np.log(2*np.pi):.6f} + {truncation_correction:.6f}")
    
    print(f"\nterm1 + term2 = {term1 + term2:.6f}")
    
    from scipy.stats import truncnorm
    a_std = (truncation - mu_val) / np.sqrt(sigma_sq)
    scipy_answer = -truncnorm(truncation, np.inf, loc=mu_val, scale=np.sqrt(sigma_sq)).logpdf(x)
    print(f"Scipy truncated NLL: {scipy_answer:.6f}")
    print(f"Match: {abs(term1 + term2 - scipy_answer) < 1e-5}")

        
# right truncated normal distribution with known truncation
def test_truncated_normal():
    M = MultivariateNormal(ch.zeros(1), ch.eye(1)) 
    samples = M.rsample([10000,])
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
                        'epochs': 1, 
                        'batch_size': 10, 
                        'trials': 1, 
                        'verbose': True,
                        'lr': 1e-2,
                        'max_update_norm': 10.0
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