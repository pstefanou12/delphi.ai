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
                        'epochs': 3, 
                        'batch_size': 10, 
                        'trials': 1, 
                        'verbose': True,
                        'lr': 1e-1,
                        'optimizer': 'sgd',
                        'max_update_norm': 10.0,
                        'damping': 1e-5,
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

def test_hessian_computation():
    import torch
    from delphi.distributions.truncated_multivariate_normal import TruncatedMultivariateNormalHessian
    """Test if Hessian matches the paper's Fisher information"""
    torch.manual_seed(42)
    
    dims = 1
    phi = oracle.Left_Distribution(torch.tensor([0.0]))
    
    def fixed_censored_sample_nll(z):
        if z.dim() == 1:
            z = z.unsqueeze(-1)
        batch_size, d = z.shape
        zzT = -0.5 * torch.bmm(z.unsqueeze(2), z.unsqueeze(1))
        zzT_flat = zzT.reshape(batch_size, -1)
        z_flat = z.reshape(batch_size, -1)
        return torch.cat([zzT_flat, z_flat], dim=1)
    
    hessian_calc = TruncatedMultivariateNormalHessian(phi, dims, fixed_censored_sample_nll, num_samples=1000)
    
    # Test parameters
    test_params = torch.tensor([1.0, 0.5])  # T, ν
    hessian_calc.set_params(test_params)
    
    hessian = hessian_calc()
    
    print("Hessian analysis:")
    print(f"Hessian shape: {hessian.shape}")
    print(f"Hessian: {hessian}")
    
    # For dims=1, the score vector is [(-½z²), z]
    # So the Fisher information = Cov[[(-½z²), z]]
    
    # Check if it's reasonable
    eigenvalues = torch.linalg.eigvals(hessian).real
    print(f"Eigenvalues: {eigenvalues}")
    
    # Should be positive definite
    assert torch.all(eigenvalues > 0), "Hessian not positive definite!"

def test_debug_hessian_quality():
    import torch
    """Check if your Hessian is actually helping"""
    torch.manual_seed(42)
    
    # Your setup
    M = MultivariateNormal(torch.zeros(1), torch.eye(1))
    samples = M.rsample([1000])
    phi = oracle.Left_Distribution(torch.tensor([0.0]))
    indices = phi(samples).nonzero()[:,0]
    S = samples[indices]
    alpha = S.size(0) / samples.size(0)
    
    args = Parameters({'epochs': 0, 'batch_size': 10, 'verbose': True, 'trials': 1})
    truncated = distributions.TruncatedNormal(args, phi, alpha, 1)
    truncated.fit(S) 
    print("=== HESSIAN QUALITY ANALYSIS ===")
    
    # Get one batch
    
    # Compute loss and gradients
    params = list(truncated.parameters())[0]
    params.requires_grad = True
    loss = truncated.criterion(params, truncated.train_loader_.dataset.data[:10], *truncated.criterion_params)
    loss.backward()
    
    # Get gradients
    grads = torch.cat([p.grad.flatten() for p in truncated.parameters()])
    print(f"Gradient norm: {torch.norm(grads).item():.6f}")
    print(f"Gradients: {grads}")
    
    # Get Hessian
    custom_hessian_fn = truncated.optimizer.param_groups[0].get('custom_hessian_fn')
    if custom_hessian_fn:
        hessian = custom_hessian_fn()
        print(f"Hessian: {hessian}")
        
        # Check Newton direction quality
        try:
            # Newton direction: -H⁻¹g
            newton_direction = -torch.linalg.solve(hessian, grads)
            print(f"Newton direction: {newton_direction}")
            
            # Gradient direction: -g  
            grad_direction = -grads
            print(f"Gradient direction: {grad_direction}")
            
            # Check if Newton direction points downhill
            directional_derivative = grads @ newton_direction
            print(f"Directional derivative: {directional_derivative.item():.6f}")
            
            if directional_derivative < 0:
                print("✓ Newton direction is downhill")
            else:
                print("❌ Newton direction is UPHILL - Hessian is wrong!")
                
        except Exception as e:
            print(f"❌ Could not compute Newton direction: {e}")

def test_check_hessian_quality():
    import torch
    """Verify Hessian is actually helpful"""
    torch.manual_seed(42)

    M = MultivariateNormal(torch.zeros(1), torch.eye(1))
    samples = M.rsample([1000])
    phi = oracle.Left_Distribution(torch.tensor([0.0]))
    indices = phi(samples).nonzero()[:,0]
    S = samples[indices]
    alpha = S.size(0) / samples.size(0)
    
    args = Parameters({'epochs': 0, 'batch_size': 10, 'verbose': True, 'trials': 1})
    
    # Your setup
    truncated = distributions.TruncatedNormal(args, phi, alpha, 1, variance=None)
    truncated.fit(S)
    
    print("=== HESSIAN QUALITY CHECK ===")
    
    # Test at multiple parameter points
    test_points = [
        torch.tensor([1.0, 0.0]),  # T=1, ν=0
        torch.tensor([1.0, 0.5]),  # T=1, ν=0.5  
        torch.tensor([2.0, 0.0]),  # T=2, ν=0
    ]
    
    for point in test_points:
        print(f"\nTesting at T={point[0]}, ν={point[1]}:")
        
        # Set parameters
        with torch.no_grad():
            params = list(truncated.parameters())[0]
            params[0].data.fill_(point[0])
            params[1].data.fill_(point[1])
        
        params.requires_grad = True 
        # Compute Hessian
        batch = S[:10]
        loss = truncated.criterion(params, truncated.train_loader_.dataset.data[:10], *truncated.criterion_params)
        loss.backward()
        
        hessian_fn = truncated.optimizer.param_groups[0]['custom_hessian_fn']
        hessian = hessian_fn()
        
        grads = torch.cat([p.grad.flatten() for p in truncated.parameters()])
        
        print(f"  Gradients: {grads}")
        print(f"  Hessian: {hessian}")
        
        # Check if Hessian points downhill
        try:
            newton_dir = -torch.linalg.solve(hessian + 1e-3*torch.eye(2), grads)
            directional_deriv = grads @ newton_dir
            print(f"  Newton direction: {newton_dir}")
            print(f"  Directional derivative: {directional_deriv:.6f}")
            
            if directional_deriv < -1e-8:
                print("  ✓ Good: Newton direction is downhill")
            else:
                print("  ❌ Bad: Newton direction is not downhill!")
        except Exception as e:
            print(f"  ❌ Failed: {e}")


def test_interpret_parameters():
    import torch
    """Interpret what your optimized parameters actually mean"""
    print("=== PARAMETER INTERPRETATION ===")
    
    # Your results
    results = [
        {'name': 'Default Newton', 'T': 2.1271, 'ν': 1.9336},
        {'name': 'Conservative Newton', 'T': 2.8801, 'ν': 1.7090},
        {'name': 'Very Conservative Newton', 'T': 2.8088, 'ν': 2.1844},
    ]

    # Your empirical data stats (you'll need to compute these)
    M = MultivariateNormal(torch.zeros(1), torch.eye(1))
    samples = M.rsample([1000])
    phi = oracle.Left_Distribution(torch.tensor([0.0]))
    indices = phi(samples).nonzero()[:,0]
    S = samples[indices]
    
    emp_mean = S.mean().item()
    emp_var = S.var().item()

    
    # Empirical statistics from your data
    emp_mean = S.mean().item()
    emp_var = S.var().item()
    
    print(f"Empirical statistics from data:")
    print(f"  Mean (μ): {emp_mean:.3f}")
    print(f"  Variance (Σ): {emp_var:.3f}")
    print(f"  For truncated N(0,1) with x>0, we expect: μ ≈ 0.798, Σ ≈ 0.363")
    print()
    
    for result in results:
        T = result['T']
        ν = result['ν']
        
        # Recover actual distribution parameters
        Σ = 1.0 / T  # variance
        μ = ν / T    # mean
        
        print(f"{result['name']}:")
        print(f"  Raw parameters: T={T:.3f} (Σ⁻¹), ν={ν:.3f} (Σ⁻¹μ)")
        print(f"  Recovered: μ={μ:.3f}, Σ={Σ:.3f}")
        
        # Check if reasonable
        expected_μ = 0.798  # For standard normal truncated at x>0
        expected_Σ = 0.363  # For standard normal truncated at x>0
        
        μ_error = abs(μ - expected_μ)
        Σ_error = abs(Σ - expected_Σ)
        
        print(f"  Expected: μ={expected_μ:.3f}, Σ={expected_Σ:.3f}")
        print(f"  Errors: Δμ={μ_error:.3f}, ΔΣ={Σ_error:.3f}")
        
        if μ_error < 0.2 and Σ_error < 0.2:
            print("  ✓ Reasonable estimates!")
        else:
            print("  ⚠️ Estimates seem off")
        print()

def test_extended_optimization():
    import torch 
    import matplotlib.pyplot as plt
    """Test with more iterations to see if it converges properly"""
    torch.manual_seed(42)
    
    M = MultivariateNormal(torch.zeros(1), torch.eye(1))
    samples = M.rsample([1000])
    phi = oracle.Left_Distribution(torch.tensor([0.0]))
    indices = phi(samples).nonzero()[:,0]
    S = samples[indices]
    alpha = S.size(0) / samples.size(0)
    
    args = Parameters({
        'epochs': 0, 
        'batch_size': 10,
        'verbose': False,
        'trials': 1,
        'lr': 0.1,  # Best from previous test
        'damping': 0.1,
        'max_update_norm': 0.5
    })
        
    truncated = distributions.TruncatedNormal(args, phi, alpha, 1)
    truncated.fit(S)
    
    print("=== EXTENDED OPTIMIZATION ===")
    
    losses = []
    parameters_history = []
    
    for step in range(20):
        # Your backward pass
        params = list(truncated.parameters())[0]
        params.requires_grad = True
        
        loss = truncated.criterion(
            params,
            truncated.train_loader_.dataset.data[:10],
            *truncated.criterion_params
        )
        # custom_hessian_fn = truncated.optimizer.param_groups[0].get('custom_hessian_fn')
        truncated.hessian.set_params(params)
        
        truncated.optimizer.zero_grad()
        loss.backward()
        truncated.optimizer.step()
        
        losses.append(loss.item())
        
        # Store parameters for analysis
        with torch.no_grad():
            current_params = [p.data.clone() for p in truncated.parameters()]
            parameters_history.append(current_params)
        
        if step % 5 == 0:
            T = current_params[0][0].item()
            ν = current_params[0][1].item()
            μ = ν / T
            Σ = 1.0 / T
            
            print(f"Step {step}: loss={loss.item():.6f}, μ={μ:.3f}, Σ={Σ:.3f}")
    
    # Final interpretation
    final_params = parameters_history[-1][0]
    T_final = final_params[0].item()
    ν_final = final_params[1].item()
    μ_final = ν_final / T_final
    Σ_final = 1.0 / T_final
    
    print(f"\nFinal results:")
    print(f"  μ: {μ_final:.3f} (expected: ~0.798)")
    print(f"  Σ: {Σ_final:.3f} (expected: ~0.363)")
    print(f"  Loss improvement: {(losses[0] - losses[-1]) / losses[0] * 100:+.1f}%")
    
    # Plot convergence
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss Convergence')
    
    plt.subplot(1, 2, 2)
    μ_history = [p[0][1].item() / p[0][0].item() for p in parameters_history]
    Σ_history = [1.0 / p[0][0].item() for p in parameters_history]
    plt.plot(μ_history, label='μ')
    plt.plot(Σ_history, label='Σ')
    plt.axhline(y=0.798, color='r', linestyle='--', alpha=0.5, label='Expected μ')
    plt.axhline(y=0.363, color='g', linestyle='--', alpha=0.5, label='Expected Σ')
    plt.xlabel('Iteration')
    plt.ylabel('Parameter Value')
    plt.legend()
    plt.title('Parameter Convergence')
    
    plt.tight_layout()
    plt.show()

def test_loss_function_quality():
    import torch
    """Test if the loss function makes sense"""
    torch.manual_seed(42)
    
    M = MultivariateNormal(torch.zeros(1), torch.eye(1))
    samples = M.rsample([1000])
    phi = oracle.Left_Distribution(torch.tensor([0.0]))
    indices = phi(samples).nonzero()[:,0]
    S = samples[indices]
    alpha = S.size(0) / samples.size(0)
    
    args = Parameters({'epochs': 0, 'batch_size': 10, 'verbose': False, 'trials': 1})
    truncated = distributions.TruncatedNormal(args, phi, alpha, 1)
    truncated.fit(S)
    
    print("=== LOSS FUNCTION QUALITY ===")
    
    # Test loss at different parameter values
    test_points = [
        (0.8, 0.36),  # Near truth
        (0.5, 0.5),   # Far from truth
        (1.0, 0.2),   # Far from truth
    ]
    
    for μ_test, Σ_test in test_points:
        T_test = 1.0 / Σ_test
        ν_test = T_test * μ_test
        
        # Set parameters
        with torch.no_grad():
            params = list(truncated.parameters())[0]
            params[0] = T_test
            params[1] = ν_test
        
        # Compute loss
        params_tensor = list(truncated.parameters())[0]
        params_tensor.requires_grad = True
        
        loss = truncated.criterion(
            params_tensor,
            truncated.train_loader_.dataset.data[:50],
            *truncated.criterion_params
        )
        
        print(f"μ={μ_test:.2f}, Σ={Σ_test:.2f} → loss={loss.item():.6f}")

def test_debug_likelihood_computation():
    import torch
    """Figure out what's wrong with the likelihood"""
    torch.manual_seed(42)
    
    M = MultivariateNormal(torch.zeros(1), torch.eye(1))
    samples = M.rsample([1000])
    phi = oracle.Left_Distribution(torch.tensor([0.0]))
    indices = phi(samples).nonzero()[:,0]
    S = samples[indices]
    alpha = S.size(0) / samples.size(0)
    
    args = Parameters({'epochs': 0, 'batch_size': 50, 'verbose': False, 'trials': 1})
    truncated = distributions.TruncatedNormal(args, phi, alpha, 1)
    truncated.fit(S)
    
    print("=== LIKELIHOOD COMPUTATION DEBUG ===")
    
    # Test the same points but examine intermediate values
    test_points = [
        (0.8, 0.36, "Near truth"),
        (0.5, 0.5, "Wrong but low loss"), 
        (1.0, 0.2, "Very wrong"),
    ]
    
    for μ_test, Σ_test, desc in test_points:
        T_test = 1.0 / Σ_test
        ν_test = T_test * μ_test
        
        print(f"\n--- {desc}: μ={μ_test:.2f}, Σ={Σ_test:.2f} ---")
        
        # Set parameters
        with torch.no_grad():
            params = list(truncated.parameters())[0]
            params[0] = T_test
            params[1] = ν_test
        
        # Get the data that will be passed to criterion
        data = truncated.train_loader_.dataset.data[:10]  # Small batch for debugging
        print(f"Data sample: {data[:5].flatten()}")
        
        # Manually compute what should happen in your criterion
        params_tensor = torch.tensor([T_test, ν_test], requires_grad=True)
        
        # Your criterion call - but let's see what happens inside
        loss = truncated.criterion(
            params_tensor,
            data,
            *truncated.criterion_params
        )
        
        print(f"Final loss: {loss.item():.6f}")
        
        # Let's also check the gradients at this point
        truncated.optimizer.zero_grad()
        loss.backward()
        grads = params_tensor.grad
        print(f"Gradients: dL/dT={grads[0]:.4f}, dL/dν={grads[1]:.4f}")

def test_criterion_directly():
    from delphi.grad import TruncatedMultivariateNormalNLL
    import torch

    def simple_censored_nll(z):
        # Mock this for now
        if z.dim() == 1:
            z = z.unsqueeze(-1)
        zzT = -0.5 * torch.bmm(z.unsqueeze(2), z.unsqueeze(1))
        z_flat = z.reshape(z.size(0), -1)
        return torch.cat([zzT.reshape(z.size(0), -1), z_flat], dim=1)

    """Test the criterion function in isolation"""
    torch.manual_seed(42)
    
    # Generate some simple test data
    test_data = torch.tensor([[0.5], [1.0], [1.5], [2.0]])
    test_data = torch.cat([simple_censored_nll(test_data), test_data], 1)
    
    # Your criterion parameters
    phi = oracle.Left_Distribution(torch.tensor([0.0]))
    dims = 1
    
    print("=== CRITERION DIRECT TEST ===")
    
    # Test different parameters
    test_params = [
        torch.tensor([2.755, 2.199]),  # True: T=1/0.363, ν=T*0.798
        torch.tensor([2.000, 1.000]),  # Wrong
        torch.tensor([3.000, 2.400]),  # Wrong
    ]
    
    for params in test_params:
        print(f"\nTesting params: T={params[0]:.3f}, ν={params[1]:.3f}")
        # Call your criterion directly
        loss = TruncatedMultivariateNormalNLL.apply(
            params, test_data, phi, dims, simple_censored_nll, None, 100
        )
        
        print(f"Loss: {loss.item():.6f}")
    
    # The loss should be lowest near the true parameters!

def test_newton_optimization_with_backward():
    import torch
    """Test Newton optimization using your exact backward pass setup"""
    torch.manual_seed(42)
    
    # Your exact setup
    M = MultivariateNormal(torch.zeros(1), torch.eye(1))
    samples = M.rsample([1000])
    phi = oracle.Left_Distribution(torch.tensor([0.0]))
    indices = phi(samples).nonzero()[:,0]
    S = samples[indices]
    alpha = S.size(0) / samples.size(0)
    
    print("=== NEWTON OPTIMIZATION DIAGNOSIS ===")
    
    # Test different optimizer configurations
    configs = [
        {'lr': 1.0, 'damping': 0.1, 'max_update_norm': 1.0, 'name': 'Default Newton'},
        {'lr': 0.1, 'damping': 0.1, 'max_update_norm': 0.5, 'name': 'Conservative Newton'},
        {'lr': 0.01, 'damping': 0.5, 'max_update_norm': 0.1, 'name': 'Very Conservative Newton'},
    ]
    
    for config in configs:
        print(f"\n--- Testing: {config['name']} ---")
        print(f"Config: lr={config['lr']}, damping={config['damping']}, max_update_norm={config['max_update_norm']}")
        
        args = Parameters({
            'epochs': 0, 
            'batch_size': 10, 
            'verbose': False, 
            'trials': 1,
            'lr': config['lr'],
            'damping': config['damping'],
            'max_update_norm': config['max_update_norm']
        })
        
        truncated = distributions.TruncatedNormal(args, phi, alpha, 1)
        truncated.fit(S)

        
        # Store optimization trajectory
        losses = []
        parameters_history = []
        gradient_norms = []
        update_norms = []
        
        # Manual optimization loop to track everything
        for step in range(5):
            # Get current parameters
            current_params = [p.data.clone() for p in truncated.parameters()]
            parameters_history.append(current_params)

            # Your exact backward pass
            params = list(truncated.parameters())[0]
            params.requires_grad = True
            
            # Compute loss and gradients
            loss = truncated.criterion(
                params, 
                truncated.train_loader_.dataset.data[:10], 
                *truncated.criterion_params
            )
            
            losses.append(loss.item())
            custom_hessian_fn = truncated.optimizer.param_groups[0].get('custom_hessian_fn')
            truncated.hessian.set_params(params)
            # Compute gradients
            truncated.optimizer.zero_grad()
            loss.backward()
            # Record gradient norms
            grads = [p.grad for p in truncated.parameters() if p.grad is not None]
            if grads:
                grad_norm = torch.norm(torch.cat([g.flatten() for g in grads])).item()
                gradient_norms.append(grad_norm)
            else:
                gradient_norms.append(0.0)
            
            # Get Hessian for analysis
            custom_hessian_fn = truncated.optimizer.param_groups[0].get('custom_hessian_fn')
            truncated.hessian.set_params(params)
            if custom_hessian_fn:
                hessian = custom_hessian_fn()
                print(f"Step {step}: loss={loss.item():.6f}, grad_norm={grad_norm:.6f}")
                print(f"  Hessian: {hessian}")
                
                # Analyze Newton step
                if grads:
                    grads_flat = torch.cat([g.flatten() for g in grads])
                    try:
                        raw_newton_step = -torch.linalg.solve(
                            hessian + config['damping'] * torch.eye(hessian.shape[0]), 
                            grads_flat
                        )
                        scaled_newton_step = config['lr'] * raw_newton_step
                        print(f"  Raw Newton step: {raw_newton_step}, norm: {torch.norm(raw_newton_step):.3f}")
                        print(f"  Scaled Newton step: {scaled_newton_step}, norm: {torch.norm(scaled_newton_step):.3f}")
                    except Exception as e:
                        print(f"  Newton step failed: {e}")
            
            # Take optimization step
            truncated.optimizer.step()
            
            # Record update size
            if step > 0:
                update = torch.cat([
                    (p2 - p1).flatten() 
                    for p1, p2 in zip(parameters_history[-2], parameters_history[-1])
                ])
                update_norms.append(torch.norm(update).item())
        
        # Analyze results
        if len(losses) > 1:
            improvement = (losses[0] - losses[-1]) / losses[0] * 100
            print(f"  Result: {improvement:+.1f}% improvement")
            
            if improvement > 0:
                print("  ✓ Making progress")
            else:
                print("  ❌ Not improving or diverging")
        
        print(f"  Final loss: {losses[-1]:.6f}")
        print(f"  Final parameters: {[p for p in truncated.parameters()]}")

def test_plot_loss_surface():
    import torch
    import matplotlib.pyplot as plt
    from delphi.grad import TruncatedMultivariateNormalNLL
    """Plot the loss surface to visualize what's wrong"""
    torch.manual_seed(42)
    
    # Generate test data
    M = MultivariateNormal(torch.zeros(1), torch.eye(1))
    samples = M.rsample([1000])
    phi = oracle.Left_Distribution(torch.tensor([0.0]))
    indices = phi(samples).nonzero()[:,0]
    S = samples[indices]
    
    # Your criterion parameters
    dims = 1
    
    def simple_censored_nll(z):
        if z.dim() == 1:
            z = z.unsqueeze(-1)
        zzT = -0.5 * torch.bmm(z.unsqueeze(2), z.unsqueeze(1))
        z_flat = z.reshape(z.size(0), -1)
        return torch.cat([zzT.reshape(z.size(0), -1), z_flat], dim=1)
    
    print("=== LOSS SURFACE PLOT ===")
    
    # Create parameter grid
    T_values = torch.linspace(0.5, 4.0, 30)
    ν_values = torch.linspace(0.5, 3.0, 30)
    
    loss_grid = torch.zeros(len(T_values), len(ν_values))
    
    # True values for reference
    true_T = 1.0 / 0.363  # ≈2.755
    true_ν = true_T * 0.798  # ≈2.199

    S_grad = simple_censored_nll(S)
    data = ch.cat([S_grad, S], dim=1)
    print(f'data shape: {data.size()}')
    
    print(f"True parameters: T={true_T:.3f}, ν={true_ν:.3f}")
    
    # Compute loss at each grid point
    for i, T_val in enumerate(T_values):
        for j, ν_val in enumerate(ν_values):
            params = torch.tensor([T_val, ν_val])
            
            try:
                loss = TruncatedMultivariateNormalNLL.apply(
                    params, data[:20], phi, dims, simple_censored_nll, None, 50
                )
                loss_grid[i, j] = loss.item()
            except Exception as e:
                loss_grid[i, j] = float('nan')
                if i == 0 and j == 0:
                    print(f"Warning: {e}")
    
    # Create the plot
    plt.figure(figsize=(12, 10))
    
    # 2D contour plot
    plt.subplot(2, 2, 1)
    import pdb; pdb.set_trace()
    X, Y = torch.meshgrid(T_values, ν_values, indexing='ij')
    contour = plt.contourf(X, Y, loss_grid, levels=20, cmap='viridis')
    plt.colorbar(contour, label='Loss')
    plt.plot(true_T, true_ν, 'r*', markersize=15, label='True parameters')
    plt.xlabel('T (Σ⁻¹)')
    plt.ylabel('ν (Σ⁻¹μ)')
    plt.title('Loss Surface')
    plt.legend()
    
    # 3D surface plot
    plt.subplot(2, 2, 2, projection='3d')
    surf = plt.gca().plot_surface(X.numpy(), Y.numpy(), loss_grid.numpy(), 
                                 cmap='viridis', alpha=0.8)
    plt.gca().scatter([true_T], [true_ν], [loss_grid.min()], 
                     color='red', s=100, label='True params')
    plt.xlabel('T')
    plt.ylabel('ν')
    plt.title('3D Loss Surface')
    
    # Slice through ν = true_ν
    plt.subplot(2, 2, 3)
    ν_idx = torch.argmin(torch.abs(ν_values - true_ν))
    plt.plot(T_values, loss_grid[:, ν_idx])
    plt.axvline(x=true_T, color='r', linestyle='--', label=f'True T={true_T:.3f}')
    plt.xlabel('T')
    plt.ylabel('Loss')
    plt.title(f'Slice at ν={ν_values[ν_idx]:.3f}')
    plt.legend()
    plt.grid(True)
    
    # Slice through T = true_T
    plt.subplot(2, 2, 4)
    T_idx = torch.argmin(torch.abs(T_values - true_T))
    plt.plot(ν_values, loss_grid[T_idx, :])
    plt.axvline(x=true_ν, color='r', linestyle='--', label=f'True ν={true_ν:.3f}')
    plt.xlabel('ν')
    plt.ylabel('Loss')
    plt.title(f'Slice at T={T_values[T_idx]:.3f}')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print some diagnostics
    min_loss_idx = torch.argmin(loss_grid)
    min_i, min_j = min_loss_idx // loss_grid.shape[1], min_loss_idx % loss_grid.shape[1]
    min_T = T_values[min_i].item()
    min_ν = ν_values[min_j].item()
    min_loss = loss_grid[min_i, min_j].item()
    
    print(f"\nMinimum loss: {min_loss:.6f} at T={min_T:.3f}, ν={min_ν:.3f}")
    print(f"True parameters loss: {loss_grid[T_idx, ν_idx]:.6f}")
    print(f"Loss at true params - min loss: {loss_grid[T_idx, ν_idx] - min_loss:.6f}")
    
    # Recover μ and Σ from optimal parameters
    optimal_μ = min_ν / min_T
    optimal_Σ = 1.0 / min_T
    print(f"Optimal parameters correspond to: μ={optimal_μ:.3f}, Σ={optimal_Σ:.3f}")

def test_debug_likelihood_sign():
    import torch
    from delphi.grad import TruncatedMultivariateNormalNLL
    """Find where the sign error occurs in the likelihood computation"""
    torch.manual_seed(42)
    
    M = MultivariateNormal(torch.zeros(1), torch.eye(1))
    samples = M.rsample([1000])
    phi = oracle.Left_Distribution(torch.tensor([0.0]))
    indices = phi(samples).nonzero()[:,0]
    S = samples[indices]
    
    dims = 1
    
    print("=== LIKELIHOOD SIGN DEBUG ===")
    
    # Test a simple case that gives negative loss
    T_test, ν_test = 1.0, 1.0  # This gives negative loss in your grid
    
    params = torch.tensor([T_test, ν_test], requires_grad=True)
    
    # Let's manually compute what should happen step by step
    print(f"Testing at T={T_test}, ν={ν_test}")
    
    # Your forward pass - let's break it down
    v = params[dims**2:]
    T = params[:dims**2].reshape(dims, dims)
    data = S[:10]  # Small batch
    S_data = data[:, :dims]
    
    print(f"Data mean: {S_data.mean().item():.3f}")
    print(f"T: {T.item():.3f}, ν: {ν_test:.3f}")
    
    # Compute sigma and mu
    sigma = T.inverse()
    mu = (sigma @ v).flatten()
    
    print(f"Recovered: μ={mu.item():.3f}, Σ={sigma.item():.3f}")
    
    # Sample from distribution
    M_dist = MultivariateNormal(mu, sigma)
    s = M_dist.sample([10 * S_data.size(0)])  # num_samples * batch_size
    
    # Apply truncation
    filtered = phi(s).nonzero(as_tuple=True)
    z = s[filtered][:S_data.size(0)]
    if z.dim() == 1: 
        z = z[..., None]
    
    print(f"Samples in truncation: {z.size(0)}/{s.size(0)}")
    
    # Compute the two terms of the likelihood
    # Term 1: 0.5 * S^T T S - S^T ν
    term1 = 0.5 * torch.bmm((S_data@T).view(S_data.size(0), 1, S_data.size(1)), 
                           S_data.view(S_data.size(0), S_data.size(1), 1)).squeeze(-1) - S_data@v[None,...].T
    
    print(f"Term 1 (data fit): mean={term1.mean().item():.6f}")
    
    # Term 2: -0.5 * z^T T z + z^T ν  (normalizing constant)
    if z.size(0) > 0:
        term2 = -0.5 * torch.bmm((z@T).view(z.size(0), 1, z.size(1)), 
                                z.view(z.size(0), z.size(1), 1)).squeeze(-1) + z@v[None,...].T
        print(f"Term 2 (normalization): mean={term2.mean().item():.6f}")
        
        # Total negative log-likelihood
        nll = term1 + term2
        print(f"Negative log-likelihood: mean={nll.mean().item():.6f}")
        
        # Check if any components are negative
        print(f"Term1 range: [{term1.min().item():.6f}, {term1.max().item():.6f}]")
        print(f"Term2 range: [{term2.min().item():.6f}, {term2.max().item():.6f}]")
        print(f"NLL range: [{nll.min().item():.6f}, {nll.max().item():.6f}]")
        
        # Negative log-likelihood should be POSITIVE
        if torch.any(nll < 0):
            print("❌ NEGATIVE VALUES IN NLL!")
            negative_mask = nll < 0
            print(f"  {negative_mask.sum().item()}/{nll.numel()} values are negative")
            print(f"  Most negative: {nll.min().item():.6f}")
    else:
        print("❌ No samples in truncation region!")
    
    # Now compute via your function for comparison
    def simple_censored_nll(z):
        if z.dim() == 1:
            z = z.unsqueeze(-1)
        zzT = -0.5 * torch.bmm(z.unsqueeze(2), z.unsqueeze(1))
        z_flat = z.reshape(z.size(0), -1)
        return torch.cat([zzT.reshape(z.size(0), -1), z_flat], dim=1)
    
    loss = TruncatedMultivariateNormalNLL.apply(
        params, ch.cat([simple_censored_nll(data), data], dim=1), phi, dims, simple_censored_nll, None, 10
    )
    
    print(f"Your function loss: {loss.item():.6f}")

def test_analyze_optimal_region():
    import torch
    from delphi.grad import TruncatedMultivariateNormalNLL
    """Find where the true optimum should be vs where it actually is"""
    torch.manual_seed(42)
    
    M = MultivariateNormal(torch.zeros(1), torch.eye(1))
    samples = M.rsample([10000])
    phi = oracle.Left_Distribution(torch.tensor([0.0]))
    indices = phi(samples).nonzero()[:,0]
    S = samples[indices]
    
    dims = 1
    
    print("=== OPTIMAL REGION ANALYSIS ===")
    
    # True parameters for truncated N(0,1) with x>0
    true_Σ = 0.363  # Theoretical variance for truncated standard normal
    true_μ = 0.798  # Theoretical mean for truncated standard normal
    true_T = 1.0 / true_Σ  # ≈2.755
    true_ν = true_T * true_μ  # ≈2.199
    
    print(f"True parameters: μ={true_μ:.3f}, Σ={true_Σ:.3f}")
    print(f"Corresponding to: T={true_T:.3f}, ν={true_ν:.3f}")
    
    # We need to recompute the loss grid or use your existing one
    # Let's compute a smaller grid for speed
    T_values = torch.linspace(1.0, 4.0, 20)  # Smaller grid around expected optimum
    ν_values = torch.linspace(1.0, 3.0, 20)
    
    loss_grid = torch.zeros(len(T_values), len(ν_values))
    
    def simple_censored_nll(z):
        if z.dim() == 1:
            z = z.unsqueeze(-1)
        zzT = -0.5 * torch.bmm(z.unsqueeze(2), z.unsqueeze(1))
        z_flat = z.reshape(z.size(0), -1)
        return torch.cat([zzT.reshape(z.size(0), -1), z_flat], dim=1)
    
    print("Computing loss grid...")
    for i, T_val in enumerate(T_values):
        for j, ν_val in enumerate(ν_values):
            params = torch.tensor([T_val, ν_val])
            
            try:
                loss = TruncatedMultivariateNormalNLL.apply(
                    params, ch.cat([simple_censored_nll(S[:20]), S[:20]], dim=1), phi, dims, simple_censored_nll, None, 100
                )
                loss_grid[i, j] = loss.item()
            except Exception as e:
                loss_grid[i, j] = float('nan')
    
    # Find the actual minimum
    min_loss_idx = torch.argmin(loss_grid)
    min_i, min_j = min_loss_idx // loss_grid.shape[1], min_loss_idx % loss_grid.shape[1]
    
    min_T = T_values[min_i].item()
    min_ν = ν_values[min_j].item()
    min_loss = loss_grid[min_i, min_j].item()
    
    min_μ = min_ν / min_T
    min_Σ = 1.0 / min_T
    
    print(f"\nActual minimum in loss grid:")
    print(f"  At T={min_T:.3f}, ν={min_ν:.3f}")
    print(f"  Corresponding to: μ={min_μ:.3f}, Σ={min_Σ:.3f}")
    print(f"  Loss value: {min_loss:.6f}")
    
    print(f"\nComparison:")
    print(f"  True: μ={true_μ:.3f}, Σ={true_Σ:.3f}")
    print(f"  Found: μ={min_μ:.3f}, Σ={min_Σ:.3f}")
    print(f"  Errors: Δμ={abs(min_μ-true_μ):.3f}, ΔΣ={abs(min_Σ-true_Σ):.3f}")
    
    # Check the loss at true parameters
    true_T_idx = torch.argmin(torch.abs(T_values - true_T))
    true_ν_idx = torch.argmin(torch.abs(ν_values - true_ν))
    true_params_loss = loss_grid[true_T_idx, true_ν_idx].item()
    
    print(f"\nLoss at true parameters: {true_params_loss:.6f}")
    print(f"Loss difference (true - min): {true_params_loss - min_loss:.6f}")
    
    # The key question: Is the loss lower at wrong parameters than at true parameters?
    if min_loss < true_params_loss:
        print("❌ PROBLEM: Loss is LOWER at wrong parameters than at true parameters!")
        print("   This means your likelihood function is incorrect.")
        
        # How much better is the wrong optimum?
        improvement_ratio = (true_params_loss - min_loss) / abs(min_loss)
        print(f"   Wrong parameters give {improvement_ratio*100:.1f}% 'better' loss")
    else:
        print("✓ Loss is properly lower at true parameters")
    import matplotlib.pyplot as plt
    # Plot to visualize
    plt.figure(figsize=(10, 8))
    X, Y = torch.meshgrid(T_values, ν_values, indexing='ij')
    
    plt.contourf(X.numpy(), Y.numpy(), loss_grid.numpy(), levels=20, cmap='viridis')
    plt.colorbar(label='Loss')
    plt.plot(true_T, true_ν, 'r*', markersize=15, label='True parameters')
    plt.plot(min_T, min_ν, 'wx', markersize=12, markeredgewidth=2, label='Found minimum')
    plt.xlabel('T (Σ⁻¹)')
    plt.ylabel('ν (Σ⁻¹μ)')
    plt.title('Loss Surface with True vs Found Optimum')
    plt.legend()
    plt.show()

def test_analyze_mean_estimation_only():
    import torch 
    from delphi.grad import TruncatedMultivariateNormalNLL
    import matplotlib.pyplot as plt
    import numpy as np
    """Test loss when only estimating mean (covariance known/fixed)"""
    torch.manual_seed(42)
    
    M = MultivariateNormal(torch.zeros(1), torch.eye(1))
    samples = M.rsample([1000])
    phi = oracle.Left_Distribution(torch.tensor([0.0]))
    indices = phi(samples).nonzero()[:,0]
    S = samples[indices]
    
    print("=== MEAN ESTIMATION ONLY (KNOWN COVARIANCE) ===")
    print(f"Data: {S.size(0)} truncated samples")
    print(f"Empirical mean: {S.mean().item():.3f}")
    print(f"Theoretical mean for N(0,1) truncated at 0: 0.798")
    
    # Fix covariance at the true value
    true_Σ = 0.363
    fixed_T = 1.0 / true_Σ  # ≈2.755 (fixed)
    
    # Test different mean values
    μ_values = torch.linspace(0.2, 1.4, 50)
    losses = []
    
    def simple_censored_nll(z):
        if z.dim() == 1:
            z = z.unsqueeze(-1)
        zzT = -0.5 * torch.bmm(z.unsqueeze(2), z.unsqueeze(1))
        z_flat = z.reshape(z.size(0), -1)
        return torch.cat([zzT.reshape(z.size(0), -1), z_flat], dim=1)
    
    print("Computing losses for different mean values...")
    
    for μ_val in μ_values:
        # For fixed T, compute ν = T * μ
        ν_val = fixed_T * μ_val
        params = torch.tensor([fixed_T, ν_val])
        
        try:
            loss = TruncatedMultivariateNormalNLL.apply(
                params, ch.cat([simple_censored_nll(S[:20]), S[:20]], 1), phi, 1, simple_censored_nll, None, 100 
            )
            losses.append(loss.item())
        except Exception as e:
            losses.append(float('nan'))
    
    # Find optimal mean
    valid_losses = [l for l in losses if not torch.isnan(torch.tensor(l))]
    if valid_losses:
        min_idx = np.argmin(valid_losses)
        optimal_μ = μ_values[min_idx].item()
        min_loss = valid_losses[min_idx]
        
        print(f"\nOptimal mean: μ={optimal_μ:.3f}")
        print(f"Minimum loss: {min_loss:.6f}")
        print(f"Theoretical optimal: μ=0.798")
        print(f"Error: Δμ={abs(optimal_μ-0.798):.3f}")
        
        # Check loss at theoretical optimum
        theoretical_ν = fixed_T * 0.798
        theoretical_params = torch.tensor([fixed_T, theoretical_ν])
        theoretical_loss = TruncatedMultivariateNormalNLL.apply(
                theoretical_params, ch.cat([simple_censored_nll(S[:20]), S[:20]], 1), phi, 1, simple_censored_nll, None, 100 
            ).item()
        
        print(f"Loss at theoretical μ=0.798: {theoretical_loss:.6f}")
        print(f"Loss difference: {theoretical_loss - min_loss:.6f}")
        
        if min_loss < theoretical_loss:
            print("❌ Minimum is NOT at theoretical optimum!")
        else:
            print("✓ Minimum is at theoretical optimum!")
    
    # Plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(μ_values.numpy(), losses)
    plt.axvline(x=0.798, color='r', linestyle='--', label='Theoretical optimum (0.798)')
    if valid_losses:
        plt.axvline(x=optimal_μ, color='g', linestyle='--', label=f'Found optimum ({optimal_μ:.3f})')
    plt.xlabel('Mean (μ)')
    plt.ylabel('Loss')
    plt.title('Loss vs Mean (Fixed Covariance)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    # Zoom in around the optimum
    if valid_losses:
        zoom_mask = (μ_values >= max(0.5, optimal_μ-0.3)) & (μ_values <= min(1.1, optimal_μ+0.3))
        zoom_μ = μ_values[zoom_mask]
        zoom_losses = [losses[i] for i in range(len(μ_values)) if zoom_mask[i]]
        
        plt.plot(zoom_μ.numpy(), zoom_losses)
        plt.axvline(x=0.798, color='r', linestyle='--', label='Theoretical (0.798)')
        plt.axvline(x=optimal_μ, color='g', linestyle='--', label=f'Found ({optimal_μ:.3f})')
        plt.xlabel('Mean (μ)')
        plt.ylabel('Loss')
        plt.title('Zoomed View Around Optimum')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return optimal_μ if valid_losses else None

def test_debug_loss_function_directly():
    import torch
    from delphi.grad import TruncatedMultivariateNormalNLL
    """Manually compute the loss to find where it goes wrong"""
    torch.manual_seed(42)
    
    M = MultivariateNormal(torch.zeros(1), torch.eye(1))
    samples = M.rsample([10000])
    phi = oracle.Left_Distribution(torch.tensor([0.0]))
    indices = phi(samples).nonzero()[:,0]
    S = samples[indices]
    
    print("=== MANUAL LOSS COMPUTATION ===")
    
    # Fixed known covariance
    true_Σ = 0.363
    fixed_T = 1.0 / true_Σ
    
    # Test the optimal vs theoretical means
    test_means = [0.763, 0.798]  # Your optimum vs theoretical optimum
    
    for μ_val in test_means:
        print(f"\n--- Testing μ={μ_val:.3f} ---")
        ν_val = fixed_T * μ_val
        
        # Manual computation following paper equation (3.5)
        # ℓ(ν, T; x) = ½xᵀTx - xᵀν + log(∫ exp(-½zᵀTz + zᵀν) dz)
        
        data = S[:10]
        T_tensor = torch.tensor([[fixed_T]])
        ν_tensor = torch.tensor([ν_val])
        
        # Term 1: ½xᵀTx - xᵀν
        term1 = 0.5 * (data @ T_tensor @ data.T).diag() - data @ ν_tensor
        print(f"Term 1 (½xᵀTx - xᵀν): mean = {term1.mean().item():.6f}")
        
        # Term 2: log(∫ exp(-½zᵀTz + zᵀν) dz)
        # We need to approximate this integral via sampling
        
        # Recover μ and Σ from T and ν
        Σ = 1.0 / fixed_T
        μ = ν_val / fixed_T  # Should equal μ_val
        
        # Sample from truncated distribution
        M_dist = MultivariateNormal(torch.tensor([μ]), torch.tensor([[Σ]]))
        num_samples = 1000
        s = M_dist.sample([num_samples])
        
        # Apply truncation
        filtered = phi(s).nonzero(as_tuple=True)
        z = s[filtered][...,None]
        
        print(f"Samples in truncation: {z.size(0)}/{num_samples}")
        
        if z.size(0) > 0:
            # Compute the exponent: -½zᵀTz + zᵀν
            exponent = -0.5 * (z @ T_tensor @ z.T).diag() + z @ ν_tensor
            
            # Compute the integral: mean(exp(exponent))
            integral = exponent.exp().mean()
            
            # Term 2: log(integral)
            term2 = integral.log()
            
            print(f"Term 2 (log(∫ exp(...) dz)): {term2.item():.6f}")
            print(f"  Integral: {integral.item():.6f}")
            print(f"  Max exponent: {exponent.max().item():.3f}")
            print(f"  Min exponent: {exponent.min().item():.3f}")
            
            # Total log-likelihood
            log_likelihood = term1.mean() + term2
            negative_log_likelihood = log_likelihood
            
            print(f"Log-likelihood: {log_likelihood.item():.6f}")
            print(f"Negative log-likelihood: {negative_log_likelihood.item():.6f}")
            
            # Compare with your function
            params = torch.tensor([fixed_T, ν_val])
            
            def simple_censored_nll(z):
                if z.dim() == 1:
                    z = z.unsqueeze(-1)
                zzT = -0.5 * torch.bmm(z.unsqueeze(2), z.unsqueeze(1))
                z_flat = z.reshape(z.size(0), -1)
                return torch.cat([zzT.reshape(z.size(0), -1), z_flat], dim=1)
            
            your_loss = TruncatedMultivariateNormalNLL.apply(
                params, ch.cat([simple_censored_nll(data), data], 1), phi, 1, simple_censored_nll, None, 10000 
            ).item()
            
            print(f"Your function loss: {your_loss:.6f}")
            print(f"Difference: {your_loss - negative_log_likelihood.item():.6f}")
        else:
            print("❌ No samples in truncation region!")

def test_optimization_with_fixed_likelihood():
    import torch
    from delphi.grad import TruncatedMultivariateNormalNLL
    import numpy as np
    import matplotlib.pyplot as plt
    """Test if Newton optimization now finds the correct optimum"""
    torch.manual_seed(42)
    
    M = MultivariateNormal(torch.zeros(1), torch.eye(1))
    samples = M.rsample([1000])
    phi = oracle.Left_Distribution(torch.tensor([0.0]))
    indices = phi(samples).nonzero()[:,0]
    S = samples[indices]
    
    print("=== OPTIMIZATION TEST WITH FIXED LIKELIHOOD ===")
    
    # Test mean estimation with known covariance
    true_Σ = 0.363
    fixed_T = 1.0 / true_Σ
    
    μ_values = torch.linspace(0.5, 1.1, 50)
    losses = []
    
    def simple_censored_nll(z):
        if z.dim() == 1:
            z = z.unsqueeze(-1)
        zzT = -0.5 * torch.bmm(z.unsqueeze(2), z.unsqueeze(1))
        z_flat = z.reshape(z.size(0), -1)
        return torch.cat([zzT.reshape(z.size(0), -1), z_flat], dim=1)
    
    for μ_val in μ_values:
        ν_val = fixed_T * μ_val
        params = torch.tensor([fixed_T, ν_val])
        
        loss = TruncatedMultivariateNormalNLL.apply(
            params, ch.cat([simple_censored_nll(S[:20]), S[:20]], dim=1), phi, 1, simple_censored_nll, None, 1000
        ).item()
        losses.append(loss)
    
    # Find optimum
    min_idx = np.argmin(losses)
    optimal_μ = μ_values[min_idx].item()
    min_loss = losses[min_idx]
    
    print(f"Optimal mean: μ={optimal_μ:.3f}")
    print(f"Theoretical optimum: μ=0.798")
    print(f"Error: Δμ={abs(optimal_μ-0.798):.3f}")
    
    # Check loss at theoretical optimum
    theoretical_idx = torch.argmin(torch.abs(μ_values - 0.798))
    theoretical_loss = losses[theoretical_idx]
    
    print(f"Loss at theoretical μ=0.798: {theoretical_loss:.6f}")
    print(f"Loss at found optimum: {min_loss:.6f}")
    
    if abs(optimal_μ - 0.798) < 0.02:  # Allow some tolerance
        print("✓ SUCCESS! Optimization finds near-correct optimum!")
    else: 
        print("❌ Still not converging to correct optimum")
        
    # Plot the loss surface
    plt.figure(figsize=(10, 6))
    plt.plot(μ_values.numpy(), losses)
    plt.axvline(x=0.798, color='r', linestyle='--', label='Theoretical optimum (0.798)')
    plt.axvline(x=optimal_μ, color='g', linestyle='--', label=f'Found optimum ({optimal_μ:.3f})')
    plt.xlabel('Mean (μ)')
    plt.ylabel('Negative Log-Likelihood')
    plt.title('Loss Surface with Fixed Likelihood')
    plt.legend()
    plt.grid(True)
    plt.show()

def test_check_integral_approximation():
    import torch
    """Verify the Monte Carlo integral approximation"""
    torch.manual_seed(42)
    
    M = MultivariateNormal(torch.zeros(1), torch.eye(1))
    samples = M.rsample([1000])
    phi = oracle.Left_Distribution(torch.tensor([0.0]))
    indices = phi(samples).nonzero()[:,0]
    S = samples[indices]
    
    print("=== INTEGRAL APPROXIMATION CHECK ===")
    
    true_Σ = 0.363
    fixed_T = 1.0 / true_Σ
    
    test_means = [0.798, 1.100]  # Theoretical vs found optimum
    
    for μ_val in test_means:
        ν_val = fixed_T * μ_val
        Σ = 1.0 / fixed_T
        
        print(f"\n--- μ={μ_val:.3f}, Σ={Σ:.3f} ---")
        
        # Sample from the distribution
        M_dist = MultivariateNormal(torch.tensor([μ_val]), torch.tensor([[Σ]]))
        num_samples = 10000  # Large sample for good approximation
        s = M_dist.sample([num_samples])
        
        # Apply truncation
        filtered = phi(s).nonzero(as_tuple=True)
        z = s[filtered][...,None]
        
        print(f"Samples in truncation: {z.size(0)}/{num_samples} ({z.size(0)/num_samples*100:.1f}%)")
        
        if z.size(0) > 0:
            # The integral we're approximating: ∫ exp(-½zᵀTz + zᵀν) dz over truncation region
            T_tensor = torch.tensor([[fixed_T]])
            ν_tensor = torch.tensor([ν_val])
            
            exponent = -0.5 * (z @ T_tensor @ z.T).diag() + z @ ν_tensor
            integral = exponent.exp().mean()
            
            print(f"Integral approximation: {integral.item():.6f}")
            print(f"log(integral): {integral.log().item():.6f}")
            print(f"Mean exponent: {exponent.mean().item():.6f}")
            print(f"Max exponent: {exponent.max().item():.6f}")
            print(f"Min exponent: {exponent.min().item():.6f}")


def test_fixed_proposal_integral():
    import torch
    """Compute the integral using fixed proposal distribution"""
    torch.manual_seed(42)
    
    M = MultivariateNormal(torch.zeros(1), torch.eye(1))
    samples = M.rsample([1000])
    phi = oracle.Left_Distribution(torch.tensor([0.0]))
    indices = phi(samples).nonzero()[:,0]
    S = samples[indices]
    
    print("=== FIXED PROPOSAL INTEGRAL ===")
    
    # Use a fixed proposal distribution (e.g., untruncated standard normal)
    proposal = MultivariateNormal(torch.zeros(1), torch.eye(1))
    num_samples = 10000
    s_proposal = proposal.sample([num_samples])
    
    # Only keep samples in truncation region
    filtered = phi(s_proposal).nonzero(as_tuple=True)
    z_fixed = s_proposal[filtered][...,None]
    
    print(f"Fixed proposal samples in truncation: {z_fixed.size(0)}/{num_samples}")
    
    test_means = [0.798, 1.100]
    true_Σ = 0.363
    fixed_T = 1.0 / true_Σ
    
    for μ_val in test_means:
        ν_val = fixed_T * μ_val
        Σ = 1.0 / fixed_T
        
        print(f"\n--- μ={μ_val:.3f} ---")
        
        # Target distribution: N(μ, Σ) truncated to phi
        # We want: ∫_S exp(-½zᵀTz + zᵀν) dz
        
        T_tensor = torch.tensor([[fixed_T]])
        ν_tensor = torch.tensor([ν_val])
        
        # Compute the integrand for all fixed samples
        exponent = -0.5 * (z_fixed @ T_tensor @ z_fixed.T).diag() + z_fixed @ ν_tensor
        integrand = exponent.exp()
        
        # The integral is the mean of the integrand (Monte Carlo)
        integral = integrand.mean()
        
        print(f"Integral: {integral.item():.6f}")
        print(f"log(integral): {integral.log().item():.6f}")

def test_theoretical_integral_check():
    import torch
    import math
    """Check what the integral should be theoretically"""
    torch.manual_seed(42)
    
    print("=== THEORETICAL INTEGRAL CHECK ===")
    
    # For a standard normal truncated to x > 0, the normalizing constant
    # should be P(x > 0) for N(0,1) = 0.5
    
    # But we're computing ∫ exp(-½zᵀTz + zᵀν) dz, which for T=I, ν=0 becomes:
    # ∫ exp(-½z²) dz from 0 to ∞ = √(π/2) ≈ 1.253
    
    true_Σ = 0.363
    fixed_T = 1.0 / true_Σ
    
    test_means = [0.798, 1.100]
    
    for μ_val in test_means:
        ν_val = fixed_T * μ_val
        Σ = 1.0 / fixed_T
        
        print(f"\n--- μ={μ_val:.3f}, Σ={Σ:.3f} ---")
        
        # For the untruncated N(μ, Σ), the integral over all R should be:
        # ∫ exp(-½(z-μ)²/Σ) dz = √(2πΣ)
        theoretical_untuncated = math.sqrt(2 * math.pi * Σ)
        print(f"Theoretical untruncated integral: {theoretical_untuncated:.6f}")
        
        # For truncated to x > 0, it should be this times P(x > 0)
        # P(x > 0) for N(μ, Σ) = 1 - Φ(-μ/√Σ)
        from scipy.stats import norm
        p_positive = 1 - norm.cdf(-μ_val / math.sqrt(Σ))
        theoretical_truncated = theoretical_untuncated * p_positive
        print(f"P(x > 0): {p_positive:.6f}")
        print(f"Theoretical truncated integral: {theoretical_truncated:.6f}")
        
        # Now compute what our integral should be in the paper's parameterization
        # We have: ∫ exp(-½zᵀTz + zᵀν) dz
        # For T = Σ⁻¹, ν = Σ⁻¹μ, this becomes:
        # ∫ exp(-½(z-μ)ᵀΣ⁻¹(z-μ) + ½μᵀΣ⁻¹μ) dz
        # = exp(½μᵀΣ⁻¹μ) ∫ exp(-½(z-μ)ᵀΣ⁻¹(z-μ)) dz
        # = exp(½μᵀΣ⁻¹μ) × normalizing_constant
        
        extra_factor = math.exp(0.5 * μ_val * fixed_T * μ_val)
        theoretical_paper = theoretical_truncated * extra_factor
        print(f"Extra factor exp(½μᵀTμ): {extra_factor:.6f}")
        print(f"Theoretical paper integral: {theoretical_paper:.6f}")


def test_fixed_likelihood_with_determinant():
    import torch
    """Likelihood with the missing determinant term"""
    torch.manual_seed(42)
    
    M = MultivariateNormal(torch.zeros(1), torch.eye(1))
    samples = M.rsample([1000])
    phi = oracle.Left_Distribution(torch.tensor([0.0]))
    indices = phi(samples).nonzero()[:,0]
    S = samples[indices]
    
    print("=== LIKELIHOOD WITH DETERMINANT TERM ===")
    
    true_Σ = 0.363
    fixed_T = 1.0 / true_Σ
    
    test_means = [0.798, 1.100]
    
    for μ_val in test_means:
        ν_val = fixed_T * μ_val
        
        data = S[:10]
        T_tensor = torch.tensor([[fixed_T]])
        ν_tensor = torch.tensor([ν_val])
        
        # Term 1: ½xᵀTx - xᵀν (from paper)
        term1 = 0.5 * (data @ T_tensor @ data.T).diag() - data @ ν_tensor
        
        # Term 2: log(∫ exp(-½zᵀTz + zᵀν) dz) (from paper)
        Σ = 1.0 / fixed_T
        μ = ν_val / fixed_T
        M_dist = MultivariateNormal(torch.tensor([μ]), torch.tensor([[Σ]]))
        
        num_samples = 1000
        s = M_dist.sample([num_samples])
        filtered = phi(s).nonzero(as_tuple=True)
        z = s[filtered][...,None]
        
        if z.size(0) > 0:
            exponent = -0.5 * (z @ T_tensor @ z.T).diag() + z @ ν_tensor
            integral = exponent.exp().mean()
            term2 = integral.log()
        else:
            term2 = torch.tensor(0.0)
        
        # MISSING TERM: ½log|T|
        det_term = 0.5 * torch.log(torch.tensor(fixed_T))  # For 1D, |T| = T
        
        # Total log-likelihood
        log_likelihood = term1.mean() + term2 + det_term
        negative_log_likelihood = log_likelihood
        
        print(f"μ={μ_val:.3f}:")
        print(f"  Term1: {term1.mean().item():.6f}")
        print(f"  Term2: {term2.item():.6f}") 
        print(f"  Det term: {det_term.item():.6f}")
        print(f"  Log-likelihood: {log_likelihood.item():.6f}")
        print(f"  Negative log-likelihood: {negative_log_likelihood.item():.6f}")


def test_final_optimization():
    import torch
    from delphi.grad import TruncatedMultivariateNormalNLL
    import numpy as np
    import matplotlib.pyplot as plt
    """Test optimization with the fully fixed likelihood"""
    torch.manual_seed(42)
    
    M = MultivariateNormal(torch.zeros(1), torch.eye(1))
    samples = M.rsample([1000])
    phi = oracle.Left_Distribution(torch.tensor([0.0]))
    indices = phi(samples).nonzero()[:,0]
    S = samples[indices]
    
    print("=== FINAL OPTIMIZATION TEST ===")
    
    true_Σ = 0.363
    fixed_T = 1.0 / true_Σ
    
    μ_values = torch.linspace(0.5, 1.1, 50)
    nll_values = []
    
    def simple_censored_nll(z):
        if z.dim() == 1:
            z = z.unsqueeze(-1)
        zzT = -0.5 * torch.bmm(z.unsqueeze(2), z.unsqueeze(1))
        z_flat = z.reshape(z.size(0), -1)
        return torch.cat([zzT.reshape(z.size(0), -1), z_flat], dim=1)
    
    for μ_val in μ_values:
        ν_val = fixed_T * μ_val
        params = torch.tensor([fixed_T, ν_val])
        
        nll = TruncatedMultivariateNormalNLL.apply(
            params, ch.cat([simple_censored_nll(S[:20]), S[:20]], 1), phi, 1, simple_censored_nll, None, 100
        ).item()
        nll_values.append(nll)
    
    # Find optimum (minimum NLL)
    min_idx = np.argmin(nll_values)
    optimal_μ = μ_values[min_idx].item()
    min_nll = nll_values[min_idx]
    
    print(f"Optimal mean: μ={optimal_μ:.3f}")
    print(f"Theoretical optimum: μ=0.798")
    print(f"Error: Δμ={abs(optimal_μ-0.798):.3f}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(μ_values.numpy(), nll_values)
    plt.axvline(x=0.798, color='r', linestyle='--', label='Theoretical optimum (0.798)')
    plt.axvline(x=optimal_μ, color='g', linestyle='--', label=f'Found optimum ({optimal_μ:.3f})')
    plt.xlabel('Mean (μ)')
    plt.ylabel('Negative Log-Likelihood')
    plt.title('Final Loss Surface')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    if abs(optimal_μ - 0.798) < 0.02:
        print("✓ SUCCESS! Final fix works!")
        return True
    else:
        print("❌ Still needs adjustment")
        return False

import torch 
import matplotlib.pyplot as plt 
import numpy as np 
from delphi.grad import TruncatedMultivariateNormalNLL

def test_verify_mle_for_truncated_normal():
    """Verify what the MLE should be for truncated normal"""
    torch.manual_seed(42)
    
    print("=== MLE FOR TRUNCATED NORMAL VERIFICATION ===")
    
    # Generate data from N(0,1) truncated to x > 0
    true_dist = torch.distributions.Normal(0, 1)
    samples = true_dist.sample((10000,))
    truncated_samples = samples[samples > 0]
    
    print(f"True μ: 0.0")
    print(f"Empirical mean of truncated data: {truncated_samples.mean().item():.6f}")
    
    # Now let's compute log-likelihood for different μ values
    # For N(μ,1) truncated to x > 0
    μ_values = torch.linspace(-0.5, 0.5, 50)
    log_likelihoods = []
    
    for μ_val in μ_values:
        # Log-likelihood for one sample x from N(μ,1) truncated to x > 0 is:
        # log p(x) = log(φ((x-μ)/1)) - log(Φ(∞) - Φ(-μ)) but since truncated at 0:
        # = log(φ(x-μ)) - log(1 - Φ(-μ))
        # = log(φ(x-μ)) - log(Φ(μ))  [since 1 - Φ(-μ) = Φ(μ)]
        
        log_phi = true_dist.log_prob(truncated_samples - μ_val)  # log(φ(x-μ))
        log_Phi_mu = torch.log(torch.distributions.Normal(0, 1).cdf(torch.tensor(μ_val)))  # log(Φ(μ))
        
        log_likelihood = (log_phi - log_Phi_mu).mean()
        log_likelihoods.append(log_likelihood.item())
    
    # Find MLE
    max_idx = np.argmax(log_likelihoods)
    mle_μ = μ_values[max_idx].item()
    max_ll = log_likelihoods[max_idx]
    
    print(f"MLE for μ: {mle_μ:.6f}")
    print(f"Maximum log-likelihood: {max_ll:.6f}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(μ_values.numpy(), log_likelihoods)
    plt.axvline(x=0.0, color='r', linestyle='--', label='True μ (0.0)')
    plt.axvline(x=mle_μ, color='g', linestyle='--', label=f'MLE ({mle_μ:.3f})')
    plt.xlabel('μ')
    plt.ylabel('Log-Likelihood')
    plt.title('Log-Likelihood vs μ for Truncated Normal')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return mle_μ


def test_your_implementation_correctly():
    """Test your implementation with the correct understanding"""
    torch.manual_seed(42)
    
    # Generate data from N(0,1) truncated to x > 0
    M = MultivariateNormal(torch.zeros(1), torch.eye(1))
    samples = M.rsample([1000])
    phi = oracle.Left_Distribution(torch.tensor([0.0]))
    indices = phi(samples).nonzero()[:,0]
    S = samples[indices]
    
    print("=== TEST YOUR IMPLEMENTATION ===")
    print(f"Data from: N(0,1) truncated to x > 0")
    print(f"Empirical mean: {S.mean().item():.6f}")
    print(f"Expected MLE for μ: ≈0.0")
    
    # We're estimating N(μ,1) truncated to x > 0
    # So variance is known = 1, so T = 1/1 = 1 (fixed)
    fixed_T = 1.0  # Because variance = 1
    
    μ_values = torch.linspace(-0.5, 0.5, 50)
    nll_values = []
    
    def simple_censored_nll(z):
        if z.dim() == 1:
            z = z.unsqueeze(-1)
        zzT = -0.5 * torch.bmm(z.unsqueeze(2), z.unsqueeze(1))
        z_flat = z.reshape(z.size(0), -1)
        return torch.cat([zzT.reshape(z.size(0), -1), z_flat], dim=1)
    
    for μ_val in μ_values:
        ν_val = fixed_T * μ_val  # ν = Tμ
        params = torch.tensor([fixed_T, ν_val])
        
        nll = TruncatedMultivariateNormalNLL.apply(
            params, ch.cat([simple_censored_nll(S[:20]), S[:20]], 1), phi, 1, simple_censored_nll, None, 100
        ).item()
        nll_values.append(nll)
    
    # Find optimum (minimum NLL)
    min_idx = np.argmin(nll_values)
    optimal_μ = μ_values[min_idx].item()
    min_nll = nll_values[min_idx]
    
    print(f"Your optimal μ: {optimal_μ:.6f}")
    print(f"Error from true μ=0: {abs(optimal_μ):.3f}")
    
    # Check if near 0
    if abs(optimal_μ) < 0.1:
        print(f"✓ SUCCESS! Your implementation finds optimum near μ=0!")
        
        # Plot to see the shape
        plt.figure(figsize=(10, 6))
        plt.plot(μ_values.numpy(), nll_values)
        plt.axvline(x=0.0, color='r', linestyle='--', label='True μ (0.0)')
        plt.axvline(x=optimal_μ, color='g', linestyle='--', label=f'Your optimum ({optimal_μ:.3f})')
        plt.xlabel('μ')
        plt.ylabel('Your Negative Log-Likelihood')
        plt.title('Your Implementation: NLL vs μ')
        plt.legend()
        plt.grid(True)
        plt.show()
        return True
    else:
        print(f"❌ Your optimum at μ={optimal_μ:.3f}, not near 0.0")
        
        # Plot to debug
        plt.figure(figsize=(10, 6))
        plt.plot(μ_values.numpy(), nll_values)
        plt.axvline(x=0.0, color='r', linestyle='--', label='True μ (0.0)')
        plt.axvline(x=optimal_μ, color='g', linestyle='--', label=f'Your optimum ({optimal_μ:.3f})')
        plt.xlabel('μ')
        plt.ylabel('Your Negative Log-Likelihood')
        plt.title('Your Implementation: NLL vs μ (WRONG)')
        plt.legend()
        plt.grid(True)
        plt.show()
        return False
    

def correct_truncated_normal_nll(μ, data, truncation_point=0.0):
    """
    Correct negative log-likelihood for N(μ,1) truncated to x > truncation_point
    """
    from scipy.stats import norm
    import numpy as np
    
    # Convert to numpy for scipy
    μ_np = μ.item() if hasattr(μ, 'item') else μ
    data_np = data.numpy() if hasattr(data, 'numpy') else data
    
    # For N(μ,1) truncated to x > a, the PDF is:
    # f(x) = φ(x-μ) / (1 - Φ(a-μ)) for x > a
    # where φ is standard normal PDF, Φ is standard normal CDF
    
    a = truncation_point
    log_phi = norm.logpdf(data_np, loc=μ_np, scale=1.0)  # log(φ(x-μ))
    log_denom = np.log(1 - norm.cdf(a - μ_np))  # log(1 - Φ(a-μ))
    
    # Log-likelihood per sample
    log_likelihood = log_phi - log_denom
    
    # Negative log-likelihood (what we minimize)
    nll = -np.mean(log_likelihood)
    
    return nll

def test_compare_with_correct_implementation():
    """Compare your implementation with a known correct one"""
    torch.manual_seed(42)
    
    # Generate data from N(0,1) truncated to x > 0
    M = MultivariateNormal(torch.zeros(1), torch.eye(1))
    samples = M.rsample([1000])
    phi = oracle.Left_Distribution(torch.tensor([0.0]))
    indices = phi(samples).nonzero()[:,0]
    S = samples[indices]
    
    print("=== COMPARISON WITH CORRECT IMPLEMENTATION ===")
    
    fixed_T = 1.0
    
    test_μ = [-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
    
    def simple_censored_nll(z):
        if z.dim() == 1:
            z = z.unsqueeze(-1)
        zzT = -0.5 * torch.bmm(z.unsqueeze(2), z.unsqueeze(1))
        z_flat = z.reshape(z.size(0), -1)
        return torch.cat([zzT.reshape(z.size(0), -1), z_flat], dim=1)
    
    print("μ\tYour NLL\tCorrect NLL\tDifference")
    print("-" * 50)
    
    for μ_val in test_μ:
        ν_val = fixed_T * μ_val
        params = torch.tensor([fixed_T, ν_val])
        
        # Your implementation
        your_nll = TruncatedMultivariateNormalNLL.apply(
            params, ch.cat([simple_censored_nll(S[:20]), S[:20]], 1), phi, 1, simple_censored_nll, None, 100
        ).item()
        
        # Correct implementation
        correct_nll = correct_truncated_normal_nll(μ_val, S[:20])
        
        print(f"{μ_val:.2f}\t{your_nll:.6f}\t{correct_nll:.6f}\t{your_nll - correct_nll:.6f}")


def test_analyze_likelihood_shape():
    """Analyze the shape of your likelihood surface"""
    torch.manual_seed(42)
    
    # Generate data from N(0,1) truncated to x > 0
    M = MultivariateNormal(torch.zeros(1), torch.eye(1))
    samples = M.rsample([1000])
    phi = oracle.Left_Distribution(torch.tensor([0.0]))
    indices = phi(samples).nonzero()[:,0]
    S = samples[indices]
    
    print("=== LIKELIHOOD SHAPE ANALYSIS ===")
    
    fixed_T = 1.0
    
    μ_values = torch.linspace(-1.0, 2.0, 100)  # Wider range
    nll_values = []
    
    def simple_censored_nll(z):
        if z.dim() == 1:
            z = z.unsqueeze(-1)
        zzT = -0.5 * torch.bmm(z.unsqueeze(2), z.unsqueeze(1))
        z_flat = z.reshape(z.size(0), -1)
        return torch.cat([zzT.reshape(z.size(0), -1), z_flat], dim=1)
    
    for μ_val in μ_values:
        ν_val = fixed_T * μ_val
        params = torch.tensor([fixed_T, ν_val])
        
        nll = TruncatedMultivariateNormalNLL.apply(
            params, ch.cat([simple_censored_nll(S[:20]), S[:20]], 1), phi, 1, simple_censored_nll, None, 100
        ).item() 
        nll_values.append(nll)
    
    # Check if monotonic
    differences = np.diff(nll_values)
    always_increasing = np.all(differences >= 0)
    always_decreasing = np.all(differences <= 0)
    monotonic = always_increasing or always_decreasing
    
    print(f"Monotonic: {monotonic}")
    if always_increasing:
        print("  Always increasing")
    elif always_decreasing:
        print("  Always decreasing")
    else:
        print("  Not monotonic - has local minima/maxima")
    
    # Find all local minima
    from scipy.signal import argrelextrema
    local_minima = argrelextrema(np.array(nll_values), np.less)[0]
    local_maxima = argrelextrema(np.array(nll_values), np.greater)[0]
    
    print(f"Local minima at μ: {[μ_values[i].item() for i in local_minima]}")
    print(f"Local maxima at μ: {[μ_values[i].item() for i in local_maxima]}")
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(μ_values.numpy(), nll_values)
    plt.axvline(x=0.0, color='r', linestyle='--', label='True μ (0.0)')
    plt.xlabel('μ')
    plt.ylabel('Your NLL')
    plt.title('Likelihood Surface Shape Analysis')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return monotonic, local_minima

def test_debug_numerical_issues():
    """Check for numerical instability in the integral"""
    torch.manual_seed(42)
    
    M = MultivariateNormal(torch.zeros(1), torch.eye(1))
    samples = M.rsample([1000])
    phi = oracle.Left_Distribution(torch.tensor([0.0]))
    indices = phi(samples).nonzero()[:,0]
    S = samples[indices]
    
    print("=== NUMERICAL INSTABILITY DEBUG ===")
    
    fixed_T = 1.0
    
    μ_values = [0.0, 0.5, 1.0, 1.5]
    
    for μ_val in μ_values:
        ν_val = fixed_T * μ_val
        
        data = S[:10]
        T_tensor = torch.tensor([[fixed_T]])
        ν_tensor = torch.tensor([ν_val])
        
        Σ = 1.0 / fixed_T
        μ_model = ν_val / fixed_T
        M_dist = MultivariateNormal(torch.tensor([μ_model]), torch.tensor([[Σ]]))
        
        # Test with different numbers of samples
        sample_sizes = [10, 100, 1000, 10000]
        
        print(f"\nμ={μ_val:.1f}:")
        for num_samples in sample_sizes:
            s = M_dist.sample([num_samples])
            filtered = phi(s).nonzero(as_tuple=True)
            z = s[filtered][...,None]
            
            if z.size(0) > 0:
                exponent = -0.5 * (z @ T_tensor @ z.T).diag() + z @ ν_tensor
                
                # Check for numerical issues
                max_exp = exponent.max().item()
                min_exp = exponent.min().item()
                integral = exponent.exp().mean()
                term2 = integral.log().item()
                
                print(f"  Samples: {num_samples:5d} -> "
                      f"z_count: {z.size(0):3d}, "
                      f"exp_range: [{min_exp:7.3f}, {max_exp:7.3f}], "
                      f"integral: {integral:.6f}, "
                      f"term2: {term2:.6f}")
            else:
                print(f"  Samples: {num_samples:5d} -> NO SAMPLES IN TRUNCATION!")

def test_no_truncation():
    """Test with identity oracle (no truncation) where we know the exact solution"""
    torch.manual_seed(42)
    
    # Generate data from N(0,1) - no truncation
    true_dist = torch.distributions.Normal(0.0, 1.0)
    data = true_dist.sample((100,)).unsqueeze(1)  # Shape: [100, 1]
    
    # Identity oracle - no truncation
    identity_phi = lambda x: torch.ones_like(x, dtype=torch.bool)
    
    print("=== TEST WITH NO TRUNCATION ===")
    print(f"Data from: N(0,1)")
    print(f"Empirical mean: {data.mean().item():.6f}")
    print(f"Expected MLE for μ: 0.0")
    
    fixed_T = 1.0  # variance = 1
    
    μ_values = torch.linspace(-2.0, 2.0, 50)
    nll_values = []
    
    def simple_censored_nll(z):
        if z.dim() == 1:
            z = z.unsqueeze(-1)
        zzT = -0.5 * torch.bmm(z.unsqueeze(2), z.unsqueeze(1))
        z_flat = z.reshape(z.size(0), -1)
        return torch.cat([zzT.reshape(z.size(0), -1), z_flat], dim=1)
    
    for μ_val in μ_values:
        ν_val = fixed_T * μ_val
        params = torch.tensor([fixed_T, ν_val])
        
        nll = TruncatedMultivariateNormalNLL.apply(
            params, ch.cat([data, simple_censored_nll(data)], 1), identity_phi, 1, simple_censored_nll, None, 100
        ).item()  
        nll_values.append(nll)
    
    # Find optimum
    min_idx = np.argmin(nll_values)
    optimal_μ = μ_values[min_idx].item()
    min_nll = nll_values[min_idx]
    
    print(f"Your optimal μ: {optimal_μ:.6f}")
    print(f"Error from true μ=0: {abs(optimal_μ):.3f}")
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(μ_values.numpy(), nll_values, 'b-', label='Your NLL')
    
    # Plot theoretical NLL for comparison
    theoretical_nll = []
    for μ_val in μ_values:
        # Theoretical NLL for N(μ,1): ½log(2π) + ½(data - μ)²
        nll = 0.5 * np.log(2 * np.pi) + 0.5 * ((data.numpy() - μ_val.item())**2).mean()
        theoretical_nll.append(nll)
    
    plt.plot(μ_values.numpy(), theoretical_nll, 'r--', label='Theoretical NLL')
    
    plt.axvline(x=0.0, color='r', linestyle='--', alpha=0.5, label='True μ (0.0)')
    plt.axvline(x=optimal_μ, color='g', linestyle='--', label=f'Your optimum ({optimal_μ:.3f})')
    plt.xlabel('μ')
    plt.ylabel('Negative Log-Likelihood')
    plt.title('No Truncation Test: Your NLL vs Theoretical NLL')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Check if shapes match
    your_min = min(nll_values)
    theoretical_min = min(theoretical_nll)
    print(f"Your min NLL: {your_min:.6f}")
    print(f"Theoretical min NLL: {theoretical_min:.6f}")
    
    if abs(optimal_μ) < 0.1 and abs(your_min - theoretical_min) < 0.5:
        print("✓ SUCCESS! Your implementation works for no truncation case!")
        return True
    else:
        print("❌ Your implementation fails even for no truncation case!")
        return False
    


def test_individual_terms():
    """Test each term of the NLL separately"""
    torch.manual_seed(42)
    
    # Simple case: one data point x=0, μ=0, T=1 (σ²=1)
    x = torch.tensor([[0.0]])
    T_val = 1.0
    mu_val = 0.0
    nu_val = T_val * mu_val  # = 0
    
    print("=== INDIVIDUAL TERM TEST ===")
    print(f"Data point: x={x.item()}")
    print(f"Parameters: μ={mu_val}, T={T_val}, ν={nu_val}")
    print()
    
    # Expected theoretical NLL for N(0,1) at x=0:
    theoretical_nll = 0.5 * np.log(2 * np.pi)
    print(f"Theoretical NLL at x=0, μ=0, σ²=1: {theoretical_nll:.6f}")
    print()
    
    # Now let's compute what YOUR code computes for each term:
    T = torch.tensor([[T_val]])
    v = torch.tensor([nu_val])
    S = x
    
    # Term 1: 1/2 x^T T x - x^T ν
    term1_matrix = torch.bmm((S @ T).view(1, 1, -1), S.view(1, -1, 1)).squeeze()
    term1 = (0.5 * term1_matrix - (S @ v).squeeze()).item()
    print(f"Your term1 (1/2 x^T T x - x^T ν): {term1:.6f}")
    print(f"  Expected for x=0: 0.0")
    print()
    
    # Sample for term2
    mu = torch.tensor([mu_val])
    sigma = torch.tensor([[1.0/T_val]])
    M = torch.distributions.MultivariateNormal(mu, sigma)
    z = M.sample([1000])
    
    print(f"z shape: {z.shape}")
    
    # Term 2: log of normalization constant (using YOUR code's formula)
    z_times_T = z @ T  # [1000, 1]
    print(f"z @ T shape: {z_times_T.shape}")
    
    bmm_result = torch.bmm(z_times_T.view(z.size(0), 1, -1), 
                           z.view(z.size(0), -1, 1))
    print(f"bmm result shape: {bmm_result.shape}")
    
    squeezed = bmm_result.squeeze(-1)
    print(f"after squeeze(-1) shape: {squeezed.shape}")
    
    z_times_v = z @ v
    print(f"z @ v shape: {z_times_v.shape}")
    
    norm_const = -0.5 * squeezed + z_times_v[...,None]
    print(f"norm_const shape: {norm_const.shape}")
    
    # Need to squeeze properly
    norm_const = norm_const.squeeze()
    print(f"norm_const after squeeze shape: {norm_const.shape}")
    
    term2 = torch.logsumexp(norm_const, dim=0) - np.log(z.size(0))
    print(f"term2 shape: {term2.shape}")
    print(f"Your term2 (log normalization): {term2.item():.6f}")
    print(f"  This should be: log(√(2π)) ≈ 0.9189")
    print()
    
    # Term 3: det term
    det_term = 0.5 * torch.logdet(T).item()
    print(f"Your det_term (0.5 * log|T|): {det_term:.6f}")
    print(f"  For T=1: 0.0")
    print()
    
    total_your = term1 + term2.item() + det_term
    print(f"Your total: {total_your:.6f}")
    print(f"Theoretical: {theoretical_nll:.6f}")
    print(f"Difference: {abs(total_your - theoretical_nll):.6f}")


def test_what_is_term2_computing():
    """Figure out what term2 is actually computing"""
    torch.manual_seed(42)
    
    print("\n=== WHAT IS TERM2? ===")
    
    # For μ=0, σ²=1, sample z ~ N(0,1)
    z = torch.randn(10000, 1)
    T = torch.tensor([[1.0]])
    v = torch.tensor([0.0])


    z_times_T = z @ T  # [1000, 1]
    print(f"z @ T shape: {z_times_T.shape}")
    
    bmm_result = torch.bmm(z_times_T.view(z.size(0), 1, -1), 
                           z.view(z.size(0), -1, 1))
    print(f"bmm result shape: {bmm_result.shape}")
    
    squeezed = bmm_result.squeeze(-1)
    print(f"after squeeze(-1) shape: {squeezed.shape}")
    
    z_times_v = z @ v
    print(f"z @ v shape: {z_times_v.shape}")
    
    norm_const = -0.5 * squeezed + z_times_v[...,None]
    print(f"norm_const shape: {norm_const.shape}")
    
    # Need to squeeze properly
    norm_const = norm_const.squeeze()
    print(f"norm_const after squeeze shape: {norm_const.shape}")

    term2 = torch.logsumexp(norm_const, dim=0) - np.log(z.size(0))
    
    # Your computation (matching your code exactly)
    # norm_const = -0.5 * torch.bmm((z @ T).view(z.size(0), 1, -1), 
    #                                z.view(z.size(0), -1, 1)).squeeze(-1) + (z @ v)[...,None]
    # term2 = (torch.logsumexp(norm_const, dim=0) - np.log(z.size(0))).item()
    
    print(f"Your term2: {term2:.6f}")
    print(f"This computes: log(mean(exp(-0.5*z²))) for z ~ N(0,1)")
    print()
    
    # What should it be?
    correct_value = 0.5 * np.log(2*np.pi)
    print(f"Should be: log(√(2π)) = {correct_value:.6f}")
    print(f"Difference: {abs(term2 - correct_value):.6f}")
    print()
    
    # Analytical computation of what you're getting
    # E[exp(-0.5*Z²)] where Z~N(0,1) = 1/√3 (by MGF of normal)
    analytical_wrong = -0.5 * np.log(2)  # This is what you'd get
    print(f"If computing E[exp(-0.5*Z²)] for Z~N(0,1): {analytical_wrong:.6f}")
    print(f"Your term2 should be close to this: {term2:.6f}")


def test_term2_comparison():
    """Compare OLD vs NEW term2 computation"""
    torch.manual_seed(42)
    
    T = torch.tensor([[1.0]])
    v = torch.tensor([0.0])
    sigma = T.inverse()
    mu = (sigma @ v).flatten()
    dims = 1
    
    # Sample
    M = torch.distributions.MultivariateNormal(mu, sigma)
    s = M.sample([1000])
    z = s  # No truncation
    
    print("=== COMPARING OLD vs NEW term2 ===\n")
    
    # OLD METHOD (your original code)
    norm_const = -0.5 * torch.bmm((z @ T).view(z.size(0), 1, -1), 
                                   z.view(z.size(0), -1, 1)).squeeze() + (z @ v).squeeze()
    term2_old = torch.log(torch.exp(norm_const).mean(dim=0))
    print(f"OLD term2 (Monte Carlo): {term2_old.item():.6f}")
    
    # NEW METHOD (analytical formula)
    acceptance_rate = z.size(0) / s.size(0)  # = 1.0 for no truncation
    log_full_integral = (0.5 * dims * torch.log(torch.tensor(2 * torch.pi)) + 
                         0.5 * torch.logdet(sigma) + 
                         0.5 * (v @ sigma @ v))
    term2_new = torch.log(torch.tensor(acceptance_rate)) + log_full_integral
    print(f"NEW term2 (analytical):  {term2_new.item():.6f}")
    
    print(f"\nExpected: {0.5 * np.log(2 * np.pi):.6f}")
    print(f"Difference OLD: {abs(term2_old.item() - 0.5 * np.log(2 * np.pi)):.6f}")
    print(f"Difference NEW: {abs(term2_new.item() - 0.5 * np.log(2 * np.pi)):.6f}")

def test_term_by_term_verification():
    """Verify each term against known correct formula"""
    torch.manual_seed(42)
    
    # Test case: x=1.0, μ=0.5, σ²=1.0
    x = 1.0
    mu_true = 0.5
    sigma_sq = 1.0
    
    T_val = 1.0 / sigma_sq  # T = 1
    nu_val = T_val * mu_true  # ν = 0.5
    
    print("=== TERM BY TERM VERIFICATION ===")
    print(f"Test case: x={x}, μ={mu_true}, σ²={sigma_sq}")
    print(f"Parameters: T={T_val}, ν={nu_val}\n")
    
    # THEORETICAL NLL
    theory_nll = 0.5 * np.log(2 * np.pi * sigma_sq) + 0.5 * (x - mu_true)**2 / sigma_sq
    print(f"Theoretical NLL: {theory_nll:.6f}")
    print(f"  = 1/2 log(2π) + 1/2(x-μ)²/σ²")
    print(f"  = {0.5 * np.log(2 * np.pi):.6f} + {0.5 * (x - mu_true)**2:.6f}")
    print()
    
    # EXPANDED FORM: 1/2 x² - xμ + 1/2 μ² + 1/2 log(2π)
    print("Expanded form:")
    term_a = 0.5 * x**2
    term_b = -x * mu_true  
    term_c = 0.5 * mu_true**2
    term_d = 0.5 * np.log(2 * np.pi)
    print(f"  1/2 x²T:        {term_a:.6f}")
    print(f"  -x·ν:           {term_b:.6f}")
    print(f"  1/2 ν²/T:       {term_c:.6f}")
    print(f"  -1/2 log|T|:    {0:.6f} (T=1)")
    print(f"  d/2 log(2π):    {term_d:.6f}")
    print(f"  Total:          {term_a + term_b + term_c + term_d:.6f}")
    print()
    
    # YOUR CODE should compute:
    T = torch.tensor([[T_val]])
    v = torch.tensor([nu_val])
    sigma = T.inverse()
    
    your_term1 = 0.5 * x**2 * T_val - x * nu_val
    your_term2_should_be = 0.5 * (v @ sigma @ v).item() - 0.5 * torch.logdet(T).item() + 0.5 * np.log(2 * np.pi)
    
    print("What YOUR code should compute:")
    print(f"  term1 (1/2 x²T - x·ν):     {your_term1:.6f}")
    print(f"  term2 (rest):              {your_term2_should_be:.6f}")
    print(f"  Total:                     {your_term1 + your_term2_should_be:.6f}")
    print()
    
    print(f"Match? {abs(theory_nll - (your_term1 + your_term2_should_be)) < 1e-6}")


def test_mu_sweep_simple():
    """Simplest possible test - sweep μ for fixed data"""
    torch.manual_seed(42)
    
    # Single data point
    x = 1.0
    T_val = 1.0
    
    print("\n=== SIMPLE μ SWEEP TEST ===")
    print(f"Single data point: x={x}")
    print(f"Fixed: T=1 (variance=1)")
    print()
    
    mu_vals = torch.linspace(-2, 2, 21)
    
    print("μ value | Your NLL | Theoretical NLL | Difference")
    print("-" * 60)
    
    for mu in mu_vals:
        # Theoretical: -log p(x|μ,σ²=1) = 0.5*log(2π) + 0.5*(x-μ)²
        theory = 0.5 * np.log(2 * np.pi) + 0.5 * (x - mu.item())**2
        
        # Your implementation
        nu = T_val * mu
        params = torch.tensor([T_val, nu.item()])
        
        def simple_censored_nll(z):
            if z.dim() == 0:
                z = z.unsqueeze(0).unsqueeze(0)
            if z.dim() == 1:
                z = z.unsqueeze(-1)
            zzT = -0.5 * torch.bmm(z.unsqueeze(2), z.unsqueeze(1))
            z_flat = z.reshape(z.size(0), -1)
            return torch.cat([zzT.reshape(z.size(0), -1), z_flat], dim=1)
        
        data_point = torch.tensor([[x]])
        identity_phi = lambda z: torch.ones(z.size(0), dtype=torch.bool)
        
        your_nll = TruncatedMultivariateNormalNLL.apply(
            params, 
            torch.cat([simple_censored_nll(data_point), data_point], 1),
            identity_phi, 
            1, 
            simple_censored_nll, 
            None, 
            1000
        ).item()
        
        print(f"{mu.item():7.3f} | {your_nll:11.6f} | {theory:15.6f} | {abs(your_nll - theory):10.6f}")
    
    print()
    print("The 'Your NLL' column should have minimum at μ=1.0")
    print("The shape should match 'Theoretical NLL' (parabola with min at μ=1.0)")


def test_monte_carlo_vs_analytical():
    """Compare Monte Carlo normalization vs analytical"""
    import torch as ch
    from torch.distributions import MultivariateNormal
    import numpy as np
    
    torch.manual_seed(42)
    
    T = ch.tensor([[1.0]])
    v = ch.tensor([0.5])
    sigma = T.inverse()
    mu = (sigma @ v).flatten()
    dims = 1
    
    print("=== MONTE CARLO vs ANALYTICAL ===")
    print(f"Parameters: T={T.item():.1f}, ν={v.item():.1f}, μ={mu.item():.1f}, σ²={sigma.item():.1f}\n")
    
    # Analytical value
    analytical = (0.5 * dims * ch.log(ch.tensor(2.0 * ch.pi)) + 
                  0.5 * ch.logdet(sigma) + 
                  0.5 * (v @ sigma @ v))
    print(f"Analytical log(∫ exp(...) dz): {analytical.item():.6f}")
    print(f"  = 0.5*log(2π) + 0.5*log|Σ| + 0.5*ν^T Σ ν")
    print(f"  = {0.5 * np.log(2 * np.pi):.6f} + 0 + {0.5 * (v @ sigma @ v).item():.6f}\n")
    
    # Monte Carlo approximation (your method)
    M = MultivariateNormal(mu, sigma)
    
    for num_samples in [100, 1000, 10000, 100000]:
        z = M.sample([num_samples])
        
        # Your computation
        norm_const = (-0.5 * ch.bmm((z @ T).view(z.size(0), 1, -1), 
                                     z.view(z.size(0), -1, 1)).squeeze() + 
                      (z @ v).squeeze())
        
        mc_estimate = ch.logsumexp(norm_const, dim=0) - ch.log(ch.tensor(float(num_samples)))
        
        error = abs(mc_estimate.item() - analytical.item())
        print(f"MC with {num_samples:6d} samples: {mc_estimate.item():.6f}  (error: {error:.6f})")
    
    print(f"\nThe issue: You're computing log(E[f(z)/q(z)]) where:")
    print(f"  - z ~ q(z) = N(μ, Σ)")
    print(f"  - f(z) = exp(-0.5 z^T T z + z^T ν)")
    print(f"  - You want: ∫ f(z) dz")
    
    print(f"\nBut this is importance sampling WITHOUT the correction factor!")
    print(f"The samples z already come from N(μ,Σ), which has density:")
    print(f"  q(z) ∝ exp(-0.5(z-μ)^T T(z-μ)) = exp(-0.5 z^T T z + z^T ν - 0.5 ν^T Σ ν)")
    
    print(f"\nSo f(z)/q(z) = exp(0.5 ν^T Σ ν) = exp({0.5 * (v @ sigma @ v).item():.6f}) = {ch.exp(0.5 * (v @ sigma @ v)).item():.6f}")
    print(f"This is a CONSTANT! The ratio doesn't depend on z.")
    
    print(f"\nWhat you should compute:")
    print(f"  ∫ f(z) dz = E_q[f(z)/q(z)] * Z_q")
    print(f"           = {ch.exp(0.5 * (v @ sigma @ v)).item():.6f} * {ch.sqrt(2 * ch.pi * sigma).item():.6f}")
    print(f"           = {(ch.exp(0.5 * (v @ sigma @ v)) * ch.sqrt(2 * ch.pi * sigma)).item():.6f}")
    print(f"  log(∫ f(z) dz) = {analytical.item():.6f}")


def test_loss_surface_diagnostic():
    """Plot loss surface and gradients to diagnose the issue"""
    torch.manual_seed(42)
    
    # Simple case: single data point
    x_data = torch.tensor([[1.0]])
    T_fixed = 1.0
    
    # Identity oracle - no truncation
    identity_phi = lambda x: torch.ones(x.size(0), dtype=torch.bool)
    
    def simple_censored_nll(z):
        if z.dim() == 1:
            z = z.unsqueeze(-1)
        zzT = -0.5 * torch.bmm(z.unsqueeze(2), z.unsqueeze(1))
        z_flat = z.reshape(z.size(0), -1)
        return torch.cat([zzT.reshape(z.size(0), -1), z_flat], dim=1)
    
    # Compute loss for different μ values
    mu_values = np.linspace(-2, 3, 100)
    your_nll_values = []
    theoretical_nll_values = []
    
    print("Computing loss surface...")
    for mu_val in mu_values:
        nu_val = T_fixed * mu_val
        params = torch.tensor([T_fixed, nu_val], requires_grad=True)
        
        # Your implementation
        data_formatted = torch.cat([simple_censored_nll(x_data), x_data], 1)
        nll = TruncatedMultivariateNormalNLL.apply(
            params.float(), data_formatted, identity_phi, 1, simple_censored_nll, None, 1000
        )
        your_nll_values.append(nll.item())
        
        # Theoretical NLL for N(μ, 1)
        theory = 0.5 * np.log(2 * np.pi) + 0.5 * (1.0 - mu_val)**2
        theoretical_nll_values.append(theory)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Loss curves
    axes[0].plot(mu_values, your_nll_values, 'b-', linewidth=2, label='Your NLL')
    axes[0].plot(mu_values, theoretical_nll_values, 'r--', linewidth=2, label='Theoretical NLL')
    axes[0].axvline(x=1.0, color='g', linestyle=':', alpha=0.7, label='True μ=1.0')
    
    your_min_idx = np.argmin(your_nll_values)
    your_min_mu = mu_values[your_min_idx]
    axes[0].axvline(x=your_min_mu, color='b', linestyle=':', alpha=0.7, 
                    label=f'Your min μ={your_min_mu:.3f}')
    
    axes[0].set_xlabel('μ', fontsize=12)
    axes[0].set_ylabel('NLL', fontsize=12)
    axes[0].set_title('Loss Surface Comparison', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Difference
    difference = np.array(your_nll_values) - np.array(theoretical_nll_values)
    axes[1].plot(mu_values, difference, 'purple', linewidth=2)
    axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('μ', fontsize=12)
    axes[1].set_ylabel('Your NLL - Theoretical NLL', fontsize=12)
    axes[1].set_title('Difference (should be constant offset)', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('loss_diagnostic.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n=== DIAGNOSTIC RESULTS ===")
    print(f"True minimum: μ = 1.0")
    print(f"Your minimum: μ = {your_min_mu:.6f}")
    print(f"Error: {abs(your_min_mu - 1.0):.6f}")
    print()
    
    # Check if difference is constant (meaning shape is correct, just offset)
    diff_std = np.std(difference)
    print(f"Std of difference: {diff_std:.6f}")
    if diff_std < 0.01:
        print("✓ Shape is CORRECT - just a constant offset issue")
        print(f"  Offset: {np.mean(difference):.6f}")
        print("  This means you're missing a constant term")
    else:
        print("✗ Shape is WRONG - the curvature doesn't match")
        print("  This means the gradient computation is incorrect")
    
    # Compute numerical gradient at a few points
    print("\n=== GRADIENT CHECK ===")
    test_mus = [0.0, 0.5, 1.0, 1.5]
    for test_mu in test_mus:
        # Finite difference
        h = 0.001
        mu_vals_fd = [test_mu - h, test_mu + h]
        nll_vals_fd = []
        
        for mu_fd in mu_vals_fd:
            nu_fd = T_fixed * mu_fd
            params_fd = torch.tensor([T_fixed, nu_fd])
            data_formatted = torch.cat([simple_censored_nll(x_data), x_data], 1)
            nll_fd = TruncatedMultivariateNormalNLL.apply(
                params_fd, data_formatted, identity_phi, 1, simple_censored_nll, None, 1000
            )
            nll_vals_fd.append(nll_fd.item())
        
        numerical_grad = (nll_vals_fd[1] - nll_vals_fd[0]) / (2 * h)
        theoretical_grad = -(1.0 - test_mu)  # d/dμ [0.5(x-μ)²] = -(x-μ)
        
        print(f"At μ={test_mu:.1f}: numerical={numerical_grad:.6f}, theoretical={theoretical_grad:.6f}, "
              f"diff={abs(numerical_grad - theoretical_grad):.6f}")

def test_importance_sampling_corrected():
    """Show the correct importance sampling approach"""
    import torch as ch
    from torch.distributions import MultivariateNormal
    
    torch.manual_seed(42)
    
    T = ch.tensor([[1.0]])
    v = ch.tensor([0.5])
    sigma = T.inverse()
    mu = (sigma @ v).flatten()
    dims = 1
    
    print("=== CORRECTED IMPORTANCE SAMPLING ===\n")
    
    # Sample from proposal q(z) = N(μ, Σ)
    M = MultivariateNormal(mu, sigma)
    z = M.sample([10000])
    
    # Target function: f(z) = exp(-0.5 z^T T z + z^T ν)
    f_z = ch.exp(-0.5 * ch.bmm((z @ T).view(z.size(0), 1, -1), 
                               z.view(z.size(0), -1, 1)).squeeze() + 
                 (z @ v).squeeze())
    
    # Proposal density: q(z) = (2π)^(-d/2) |Σ|^(-1/2) exp(-0.5 (z-μ)^T T (z-μ))
    # But we don't need to compute q(z) explicitly!
    
    # The ratio f(z)/q(z) simplifies to:
    # f(z)/q(z) = exp(-0.5 z^T T z + z^T ν) / [(2π)^(-d/2) |Σ|^(-1/2) exp(-0.5 (z-μ)^T T (z-μ))]
    #           = (2π)^(d/2) |Σ|^(1/2) * exp(-0.5 z^T T z + z^T ν) / exp(-0.5 z^T T z + z^T T μ - 0.5 μ^T T μ)
    #           = (2π)^(d/2) |Σ|^(1/2) * exp(z^T T μ - z^T ν - 0.5 μ^T T μ)
    #           = (2π)^(d/2) |Σ|^(1/2) * exp(0 - 0.5 ν^T Σ ν)  [since Tμ = ν]
    #           = (2π)^(d/2) |Σ|^(1/2) * exp(-0.5 ν^T Σ ν)
    
    # This is a CONSTANT (doesn't depend on z)!
    ratio_constant = ((2 * ch.pi) ** (dims/2) * 
                      ch.sqrt(ch.det(sigma)) * 
                      ch.exp(-0.5 * (v @ sigma @ v)))
    
    print(f"The ratio f(z)/q(z) = {ratio_constant.item():.6f} (constant!)\n")
    
    # So: ∫ f(z) dz = E[f(z)/q(z)] * Z_q
    #               = ratio_constant * ∫ q(z) dz  
    #               = ratio_constant * 1  [since q is a normalized density]
    #               = ratio_constant
    
    # But wait, that's not right either...
    # Let me recalculate properly
    
    print("Actually, let's be more careful:\n")
    print("We want: log(∫ f(z) dz)")
    print("We have: z ~ q(z) = N(μ, Σ)")
    print()
    
    # METHOD 1: Direct Monte Carlo (what you're doing - WRONG)
    log_f_z = (-0.5 * ch.bmm((z @ T).view(z.size(0), 1, -1), 
                             z.view(z.size(0), -1, 1)).squeeze() + 
               (z @ v).squeeze())
    method1 = ch.logsumexp(log_f_z, dim=0) - ch.log(ch.tensor(float(z.size(0))))
    
    # METHOD 2: Importance sampling (CORRECT)
    # ∫ f(z) dz = ∫ [f(z)/q(z)] * q(z) dz * Z_q
    # where Z_q = (2π)^(d/2) |Σ|^(1/2)
    
    # But there's a simpler way: recognize that when sampling from N(μ,Σ),
    # the samples have implicit density, so:
    # E[f(z)] where z~q ≈ (1/n) Σ f(z_i)
    # But ∫ f(z) dz = ∫ [f(z)/q(z)] q(z) dz 
    
    # Actually, the cleanest approach for truncated case:
    # ∫_S f(z) dz ≈ P(z ∈ S) * (2π)^(d/2) |Σ|^(1/2) * exp(0.5 ν^T Σ ν)
    
    acceptance_rate = 1.0  # No truncation in this test
    method2 = (ch.log(ch.tensor(acceptance_rate)) +
               0.5 * dims * ch.log(ch.tensor(2.0 * ch.pi)) +
               0.5 * ch.logdet(sigma) +
               0.5 * (v @ sigma @ v))
    
    # Analytical answer
    analytical = (0.5 * dims * ch.log(ch.tensor(2.0 * ch.pi)) + 
                  0.5 * ch.logdet(sigma) + 
                  0.5 * (v @ sigma @ v))
    
    print(f"Method 1 (your MC):        {method1.item():.6f}")
    print(f"Method 2 (corrected):      {method2.item():.6f}")
    print(f"Analytical:                {analytical.item():.6f}")
    print()
    print(f"Error method 1: {abs(method1.item() - analytical.item()):.6f}")
    print(f"Error method 2: {abs(method2.item() - analytical.item()):.6f}")

# ```

## The Key Insight

# When you sample from `N(μ, Σ)`, the ratio `f(z)/q(z)` is actually **constant** (doesn't depend on the samples)! This is because:
# ```
# f(z) = exp(-1/2 z^T T z + z^T ν)
# q(z) ∝ exp(-1/2 (z-μ)^T T (z-μ)) = exp(-1/2 z^T T z + z^T Tμ - 1/2 μ^T Tμ)``


def test_manual_computation():
    """Manually compute NLL and compare"""
    
    x = 1.0
    mu = 0.5
    sigma_sq = 1.0
    
    T = 1.0 / sigma_sq  # = 1.0
    nu = mu / sigma_sq  # = 0.5
    sigma = 1.0 / T     # = 1.0
    
    print("=== MANUAL COMPUTATION ===")
    print(f"x={x}, μ={mu}, σ²={sigma_sq}")
    print(f"T={T}, ν={nu}, Σ={sigma}\n")
    
    # Theoretical NLL
    theory = 0.5 * np.log(2 * np.pi * sigma_sq) + 0.5 * (x - mu)**2 / sigma_sq
    print(f"Theoretical NLL: {theory:.6f}\n")
    
    # Your formula breakdown
    print("Your formula terms:")
    term1 = 0.5 * x**2 * T - x * nu
    print(f"  term1 (1/2 x²T - xν):        {term1:.6f}")
    
    term2a = 0.5 * nu**2 / T
    print(f"  term2a (1/2 ν²/T):           {term2a:.6f}")
    
    term2b = -0.5 * np.log(T)
    print(f"  term2b (-1/2 log T):         {term2b:.6f}")
    
    term2c = 0.5 * np.log(2 * np.pi)
    print(f"  term2c (1/2 log 2π):         {term2c:.6f}")
    
    acceptance = 1.0  # no truncation
    term2d = np.log(acceptance)
    print(f"  term2d (log acceptance):     {term2d:.6f}")
    
    total = term1 + term2a + term2b + term2c + term2d
    print(f"\n  Total: {total:.6f}")
    print(f"  Match? {abs(total - theory) < 1e-6}")
    
    # Wait, let me double-check the term2a calculation
    print(f"\n  Verification: ν²/T = {nu}² / {T} = {nu**2 / T:.6f}")
    print(f"  Also: ν^T Σ ν = {nu} * {sigma} * {nu} = {nu * sigma * nu:.6f}")


def test_debug_actual_code():
    """Debug what the actual code computes"""
    import torch as ch
    from torch.distributions import MultivariateNormal
    
    # Set up test case
    x_data = torch.tensor([[1.0]])
    T_fixed = 1.0
    mu_test = 0.5
    nu_test = T_fixed * mu_test
    
    identity_phi = lambda x: torch.ones(x.size(0), dtype=torch.bool)
    
    def simple_censored_nll(z):
        if z.dim() == 1:
            z = z.unsqueeze(-1)
        zzT = -0.5 * torch.bmm(z.unsqueeze(2), z.unsqueeze(1))
        z_flat = z.reshape(z.size(0), -1)
        return torch.cat([zzT.reshape(z.size(0), -1), z_flat], dim=1)
    
    # Manually run through forward pass
    params = torch.tensor([T_fixed, nu_test])
    data = torch.cat([simple_censored_nll(x_data), x_data], 1)
    
    print("=== DEBUGGING ACTUAL CODE ===")
    print(f"Input: x={x_data.item():.1f}, T={T_fixed:.1f}, ν={nu_test:.1f}")
    print(f"Expected NLL: {1.043939:.6f}\n")
    
    # Extract values like the forward function does
    dims = 1
    v = params[dims**2:]
    T = params[:dims**2].reshape(dims, dims)
    S = data[:, dims**2+dims:]
    import pdb; pdb.set_trace()
    
    print(f"Extracted v: {v}")
    print(f"Extracted T: {T}")
    print(f"Extracted S: {S}\n")
    
    sigma = T.inverse()
    mu = (sigma @ v).flatten()
    
    print(f"Computed σ: {sigma}")
    print(f"Computed μ: {mu}\n")
    
    # Sample
    torch.manual_seed(42)
    M = MultivariateNormal(mu, sigma)
    s = M.sample([1000])
    filtered = identity_phi(s).nonzero(as_tuple=True)
    z = s[filtered]
    
    print(f"Sampled {s.size(0)} points, {z.size(0)} passed filter\n")
    
    # Compute term1
    term1_matrix = ch.bmm((S @ T).view(S.size(0), 1, -1), 
                          S.view(S.size(0), -1, 1)).squeeze(-1)
    term1_full = 0.5 * term1_matrix - (S @ v)
    
    print(f"term1 computation:")
    print(f"  S @ T = {(S @ T).item():.6f}")
    print(f"  (S @ T) @ S.T = {term1_matrix.item():.6f}")
    print(f"  0.5 * that = {(0.5 * term1_matrix).item():.6f}")
    print(f"  S @ v = {(S @ v).item():.6f}")
    print(f"  term1 = {term1_full.item():.6f}")
    print(f"  Expected: {0.0:.6f}\n")
    
    # Compute term2
    acceptance_rate = float(z.size(0)) / float(s.size(0))
    
    v_sigma_v = (v @ sigma @ v).item()
    log_det_sigma = ch.logdet(sigma).item()
    log_det_T = ch.logdet(T).item()
    
    print(f"term2 computation:")
    print(f"  v @ sigma @ v = {v_sigma_v:.6f} (expected: 0.25)")
    print(f"  0.5 * v @ sigma @ v = {0.5 * v_sigma_v:.6f} (expected: 0.125)")
    print(f"  log|Σ| = {log_det_sigma:.6f} (expected: 0.0)")
    print(f"  log|T| = {log_det_T:.6f} (expected: 0.0)")
    print(f"  -0.5 * log|T| = {-0.5 * log_det_T:.6f} (expected: 0.0)")
    print(f"  0.5 * log(2π) = {0.5 * ch.log(ch.tensor(2.0 * ch.pi)).item():.6f} (expected: 0.918939)")
    print(f"  acceptance_rate = {acceptance_rate:.6f} (expected: 1.0)")
    print(f"  log(acceptance) = {ch.log(ch.tensor(acceptance_rate)).item():.6f} (expected: 0.0)")
    
    term2 = (0.5 * (v @ sigma @ v) - 
             0.5 * ch.logdet(T) + 
             0.5 * dims * ch.log(ch.tensor(2.0 * ch.pi)) +
             ch.log(ch.tensor(acceptance_rate)))
    
    print(f"  term2 total = {term2.item():.6f} (expected: 1.043939)\n")
    
    total = term1_full + term2
    print(f"TOTAL NLL: {total.item():.6f}")
    print(f"Expected:  {1.043939:.6f}")
    print(f"Difference: {abs(total.item() - 1.043939):.6f}")


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