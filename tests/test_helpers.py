"""
Test suite for helper functions.
"""
import torch as ch 
import torch.nn as nn
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from delphi.optimizers import NewtonOptimizer


def test_gradient_tracking():
    """Test to identify where gradients are being lost"""
    print("=== Debugging Gradient Tracking ===")
    
    # Simple quadratic model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.x = nn.Parameter(ch.tensor([2.0]))
            
        def forward(self):
            return self.x ** 2
    
    model = SimpleModel()
    optimizer = NewtonOptimizer(model.parameters())
    
    # Step 1: Forward pass and backward
    print("1. Computing loss and gradients...")
    loss = model()
    print(f"   loss.requires_grad: {loss.requires_grad}")
    print(f"   loss.grad_fn: {loss.grad_fn}")
    
    loss.backward()
    print(f"   model.x.grad: {model.x.grad}")
    print(f"   model.x.grad.requires_grad: {model.x.grad.requires_grad if model.x.grad is not None else 'None'}")
    
    # Step 2: Check gradient flattening
    print("\n2. Flattening gradients...")
    params = [p for p in model.parameters() if p.grad is not None]
    grads_flat = ch.cat([p.grad.flatten() for p in params])
    print(f"   grads_flat.requires_grad: {grads_flat.requires_grad}")
    print(f"   grads_flat.grad_fn: {grads_flat.grad_fn}")
    
    # Step 3: Check Hessian computation
    print("\n3. Computing Hessian...")
    try:
        hessian = optimizer._compute_diagonal_hessian(params, grads_flat)
        print(f"   Hessian computation successful")
        print(f"   hessian.requires_grad: {hessian.requires_grad}")
    except Exception as e:
        print(f"   Hessian computation failed: {e}")
    
    print("=== End Debug ===")

def test_newton_step_corrected():
    """Corrected debug to show what's really happening"""
    import torch
    print("=== Corrected Newton Step Debug ===")
    
    A = torch.tensor([[3.0, 1.0], [1.0, 2.0]])
    b = torch.tensor([1.0, -1.0])
    x_current = torch.tensor([5.0, -5.0])
    
    print(f"Current position: {x_current}")
    
    # Compute gradient at current position
    g = A @ x_current + b  # ∇f(x) = Ax + b
    print(f"Gradient at current position: {g}")
    
    # Newton update: Δx = -A⁻¹g
    A_inv = torch.linalg.inv(A)
    delta_x = -A_inv @ g
    print(f"Newton update Δx = -A⁻¹g: {delta_x}")
    
    # New position after update
    x_new = x_current + delta_x
    print(f"New position x + Δx: {x_new}")
    
    # True optimum from solving ∇f(x) = 0 → Ax + b = 0 → x = -A⁻¹b
    x_optimal = -torch.linalg.solve(A, b)
    print(f"True optimum -A⁻¹b: {x_optimal}")
    
    print(f"x_new equals x_optimal: {torch.allclose(x_new, x_optimal)}")
    
    # What the "Expected update" SHOULD have been:
    # If someone thought "update" meant "move to this absolute position"
    wrong_expected_update = x_optimal  # This is WRONG!
    print(f"WRONG 'expected update' (absolute position): {wrong_expected_update}")
    
    print(f"Correct update Δx (relative): {delta_x}")
    print(f"Update moves us from {x_current} to {x_new}")

def test_minimal_manual():
    """Minimal manual test to isolate the issue"""
    print("=== Minimal Manual Test ===")
    
    # Simple 1D case: f(x) = 0.5*2*x² + 1*x = x² + x
    # f'(x) = 2x + 1, f''(x) = 2
    # Minimum at: 2x + 1 = 0 → x = -0.5
    
    x = ch.tensor([3.0], requires_grad=True)  # Start at x=3
    
    # Compute loss and gradients manually
    loss = 0.5 * 2 * x**2 + 1 * x  # = x² + x
    loss.backward(create_graph=True)
    
    print(f"x: {x.item()}")
    print(f"f(x): {loss.item()}")  # Should be 3² + 3 = 12
    print(f"f'(x): {x.grad.item()}")  # Should be 2*3 + 1 = 7
    print(f"f''(x): 2.0")
    
    # Newton update: Δx = -f''(x)⁻¹ * f'(x) = - (1/2) * 7 = -3.5
    # So x_new = 3 - 3.5 = -0.5 (exactly the optimum!)
    newton_update = - (1/2) * x.grad.item()
    print(f"Newton update Δx: {newton_update}")
    print(f"x_new should be: {x.item() + newton_update} (should be -0.5)")
    
    # Now test with our optimizer on the same problem
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.x = nn.Parameter(ch.tensor([3.0]))
            
        def forward(self):
            return 0.5 * 2 * self.x**2 + 1 * self.x
    
    model = SimpleModel()
    optimizer = NewtonOptimizer(model.parameters(), damping=1e-6)
    
    def closure():
        optimizer.zero_grad()
        loss = model()
        loss.backward(create_graph=True)
        return loss
    
    print(f"\nBefore optimizer step:")
    print(f"model.x: {model.x.item()}")
    
    loss_val = optimizer.step(closure)
    
    print(f"After optimizer step:")
    print(f"model.x: {model.x.item()} (should be -0.5)")
    print(f"Loss: {loss_val.item()} (should be 0.25)")

def test_quadratic_convergence():
    """Debug why quadratic isn't converging correctly"""
    import torch
    print("=== Debugging Quadratic Convergence ===")
    
    A = torch.tensor([[3.0, 1.0], [1.0, 2.0]])
    b = torch.tensor([1.0, -1.0])
    
    class QuadraticModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.x = nn.Parameter(torch.tensor([5.0, -5.0]))
            
        def forward(self):
            return 0.5 * self.x @ A @ self.x + b @ self.x
    
    model = QuadraticModel()
    
    # Manual computation
    print("Manual computation:")
    x = torch.tensor([5.0, -5.0], requires_grad=True)
    loss = 0.5 * x @ A @ x + b @ x
    print(f"Initial loss: {loss.item()}")
    
    # Compute gradient
    grad = torch.autograd.grad(loss, x)[0]
    print(f"Gradient: {grad}")
    
    # Compute Hessian manually
    print(f"Hessian (should be A):\n{A}")
    
    # Newton update: Δx = -A⁻¹g
    A_inv = torch.linalg.inv(A)
    delta_x_manual = -A_inv @ grad
    print(f"Manual Newton update: {delta_x_manual}")
    print(f"Expected new position: {x + delta_x_manual}")
    print(f"True optimum: {-torch.linalg.solve(A, b)}")
    
    # Now test the optimizer
    print("\nOptimizer computation:")
    optimizer = NewtonOptimizer(model.parameters(), damping=1e-6)
    
    def closure():
        optimizer.zero_grad()
        loss = model()
        loss.backward(create_graph=True)
        return loss
    
    # Check what the optimizer computes
    initial_loss = closure()
    print(f"Model gradient: {model.x.grad}")
    
    # Let's manually compute what the optimizer should do
    params = [p for p in model.parameters() if p.grad is not None]
    grads_flat = torch.cat([p.grad.flatten() for p in params])
    
    # Compute Hessian
    hessian = torch.zeros(2, 2)
    for i in range(2):
        grad2 = torch.autograd.grad(grads_flat[i], params, retain_graph=True)
        grad2_flat = torch.cat([g.flatten() for g in grad2])
        hessian[i, :] = grad2_flat
    
    print(f"Optimizer computed Hessian:\n{hessian}")
    
    # Compute update
    update = torch.linalg.solve(hessian + 1e-6 * torch.eye(2), -grads_flat)
    print(f"Optimizer computed update: {update}")
    print(f"Expected new position: {model.x.data + update}")
    
    # Apply the update manually and see what happens
    with torch.no_grad():
        model.x.data += update
    
    print(f"Actual new position: {model.x.data}")
    print(f"New loss: {model().item()}")


def test_quadratic_function():
    """Test on simple quadratic function where Newton should converge in 1 step"""
    print("=== Test 1: Quadratic Function ===")
        
    # Minimize f(x) = 0.5 * x^T A x + b^T x
    A = ch.tensor([[3.0, 1.0], [1.0, 2.0]])  # Positive definite
    b = ch.tensor([1.0, -1.0])
        
    class QuadraticModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.x = nn.Parameter(ch.tensor([5.0, -5.0]))  # Start far from optimum
                
        def forward(self):
            return 0.5 * self.x @ A @ self.x + b @ self.x
        
    model = QuadraticModel()
    optimizer = NewtonOptimizer(
        model.parameters(), 
        damping=1e-6,
        max_update_norm=10.0, 
        hessian_approx='full'
    )
        
    initial_loss = model().item()
    print(f"Initial: x = {model.x.data}, loss = {initial_loss:.6f}")
        
    # One Newton step should find the exact solution for quadratic functions
    for step in range(1):
        def closure():
            optimizer.zero_grad()
            loss = model()
            loss.backward(create_graph=True)
            return loss
        loss = optimizer.step(closure)
        print(f"Step {step}: x = {model.x.data}, loss = {loss:.6f}")
        
    # Verify we found the true minimum: x* = -A^{-1}b
    x_optimal = -ch.linalg.solve(A, b)
    final_error = ch.norm(model.x.data - x_optimal)
        
    print(f"True optimum: {x_optimal}")
    print(f"Final error: {final_error:.6f}")
    assert final_error < 1e-5, f"Failed to find optimum, error: {final_error}"
    print("✓ Quadratic test passed!\n")

def test_debug_hessian_computation():
    """Debug exactly what Hessian is being computed"""
    import torch
    print("=== Debugging Hessian Computation ===")
    
    A = torch.tensor([[3.0, 1.0], [1.0, 2.0]])
    b = torch.tensor([1.0, -1.0])
    
    class QuadraticModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.x = nn.Parameter(torch.tensor([5.0, -5.0]))
            
        def forward(self):
            return 0.5 * self.x @ A @ self.x + b @ self.x
    
    model = QuadraticModel()
    
    # Manual computation
    print("Manual computation:")
    x_manual = torch.tensor([5.0, -5.0], requires_grad=True)
    loss_manual = 0.5 * x_manual @ A @ x_manual + b @ x_manual
    loss_manual.backward(create_graph=True)
    grad_manual = x_manual.grad.clone()
    print(f"Gradient: {grad_manual}")
    print(f"Expected Hessian (A):\n{A}")
    
    # Compute Hessian manually using autograd
    print("\nManual Hessian via autograd:")
    hessian_manual = torch.zeros(2, 2)
    for i in range(2):
        grad2 = torch.autograd.grad(grad_manual[i], x_manual, retain_graph=True)[0]
        hessian_manual[i, :] = grad2
        print(f"Row {i}: {grad2}")
    print(f"Computed Hessian:\n{hessian_manual}")
    print(f"Matches expected: {torch.allclose(hessian_manual, A, atol=1e-6)}")
    
    # Newton update with manual Hessian
    update_manual = -torch.linalg.solve(hessian_manual, grad_manual)
    print(f"Manual Newton update: {update_manual}")
    print(f"Manual new position: {x_manual + update_manual}")
    
    # Now test what the optimizer computes
    print("\nOptimizer computation:")
    
    # Let's temporarily modify the optimizer to print the Hessian it computes
    original_compute_hessian = NewtonOptimizer._compute_hessian_auto
    
    def debug_compute_hessian(self, params, grads_flat, hessian_approx):
        print("OPTIMIZER HESSIAN COMPUTATION:")
        hessian = original_compute_hessian(self, params, grads_flat, hessian_approx)
        print(f"Optimizer computed Hessian:\n{hessian}")
        print(f"Gradients: {grads_flat}")
        return hessian
    
    NewtonOptimizer._compute_hessian_auto = debug_compute_hessian
    
    optimizer = NewtonOptimizer(model.parameters(), damping=1e-6, max_update_norm=10.0)
    
    def closure():
        optimizer.zero_grad()
        loss = model()
        loss.backward(create_graph=True)
        return loss
    
    initial_loss = closure()
    print(f"Model gradient: {model.x.grad}")
    
    final_loss = optimizer.step(closure)
    print(f"Optimizer result: {model.x.data}")
    
    # Restore original method
    NewtonOptimizer._compute_hessian_auto = original_compute_hessian

def test_verify_quadratic_math():
    """Verify the expected Newton update for the quadratic test"""
    import torch
    print("=== Verifying Quadratic Math ===")
    
    A = torch.tensor([[3.0, 1.0], [1.0, 2.0]])
    b = torch.tensor([1.0, -1.0])
    x = torch.tensor([5.0, -5.0])
    
    # Gradient: ∇f(x) = Ax + b
    grad = A @ x + b
    print(f"Gradient at x={x}: {grad}")
    
    # Newton update: Δx = -A⁻¹(Ax + b) = -x - A⁻¹b
    A_inv = torch.linalg.inv(A)
    update = -A_inv @ grad
    print(f"Newton update: {update}")
    print(f"Update norm: {torch.norm(update):.6f}")
    
    # New position
    x_new = x + update
    print(f"New x: {x_new}")
    print(f"Should be: {-torch.linalg.solve(A, b)}")
    
    # Check if they match
    x_optimal = -torch.linalg.solve(A, b)
    error = torch.norm(x_new - x_optimal)
    print(f"Error: {error:.6f}")

def test_rosenbrock_function():
    """Test on Rosenbrock function - classic optimization benchmark"""
    print("=== Test 3: Rosenbrock Function ===")
        
    # Rosenbrock: f(x,y) = (1-x)^2 + 100*(y-x^2)^2
    # Minimum at (1,1)
    class RosenbrockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.x = nn.Parameter(ch.tensor([-1.0, 2.0]))  # Start away from optimum
                
        def forward(self):
            x, y = self.x[0], self.x[1]
            return (1 - x)**2 + 100 * (y - x**2)**2
        
    model = RosenbrockModel()
    optimizer = NewtonOptimizer(
        model.parameters(), 
        damping=1e-4,
        max_update_norm=10.0,
        hessian_approx='full'
    )
        
    initial_loss = model().item()
    print(f"Initial: x = {model.x.data}, loss = {initial_loss:.6f}")
        
    losses = []
    for step in range(20):
        def closure():
            optimizer.zero_grad()
            loss = model()
            loss.backward(create_graph=True)
            return loss
            
        loss = optimizer.step(closure)
        losses.append(loss)
            
        if step % 5 == 0:
            print(f"Step {step}: x = {model.x.data}, loss = {loss:.6f}")
        
    # Check if we're close to optimum (1,1)
    final_pos = model.x.data
    error = ch.norm(final_pos - ch.tensor([1.0, 1.0]))
    final_loss = model().item()
        
    print(f"Final position: {final_pos}, error: {error:.6f}, loss: {final_loss:.6f}")
    assert error < 0.1, f"Failed to find Rosenbrock minimum, error: {error}"
    assert final_loss < 0.1, f"Final loss too high: {final_loss}"
    print("✓ Rosenbrock test passed!\n")
    
def test_custom_hessian():
    """Test with custom Hessian function"""
    import torch
    print("=== Test 4: Custom Hessian ===")
    
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.tensor([2.0, 3.0]))  # FIX: torch instead of ch
            
        def forward(self):
            # f(w) = 0.5 * w^T diag([2,3]) w = w₁² + 1.5w₂²
            return 0.5 * (2*self.w[0]**2 + 3*self.w[1]**2)
    
    def custom_hessian_fn():
        # Analytical Hessian for f(w) = 0.5 * w^T diag([2,3]) w
        # H = [[2, 0], [0, 3]]
        return torch.diag(torch.tensor([2.0, 3.0]))  # FIX: torch instead of ch
    
    model = SimpleModel()
    
    # FIX: Disable update clipping for this test
    optimizer = NewtonOptimizer(
        model.parameters(), 
        custom_hessian_fn=custom_hessian_fn,  # FIX: Pass function directly, no lambda
        damping=1e-6,
        max_update_norm=10.0  # FIX: Larger value to avoid clipping
    )
    
    initial_loss = model().item()
    print(f"Initial: w = {model.w.data}, loss = {initial_loss:.6f}")
    
    # One step with exact Hessian should find optimum at (0,0)
    def closure():
        optimizer.zero_grad()
        loss = model()
        loss.backward(create_graph=True)  # FIX: Added create_graph=True
        return loss
    
    loss_after = optimizer.step(closure)
    
    final_pos = model.w.data
    error = torch.norm(final_pos)  # FIX: torch instead of ch
    print(f"Final: w = {final_pos}, loss = {loss_after:.6f}, error: {error:.6f}")
    
    assert error < 1e-5, f"Custom Hessian failed, error: {error}"
    print("✓ Custom Hessian test passed!\n")

def test_custom_hessian_final():
    """Final working custom Hessian test"""
    import torch
    print("=== Test 4: Custom Hessian (Fixed) ===")
    
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.tensor([2.0, 3.0]))
            
        def forward(self):
            return 0.5 * (2*self.w[0]**2 + 3*self.w[1]**2)
    
    def custom_hessian_fn():
        # Analytical Hessian
        return torch.diag(torch.tensor([2.0, 3.0]))
    
    model = SimpleModel()
    
    # KEY: Disable update clipping for this test
    optimizer = NewtonOptimizer(
        model.parameters(), 
        custom_hessian_fn=custom_hessian_fn,
        damping=1e-6,
        max_update_norm=100.0  # Large value to avoid clipping
    )
    
    initial_loss = model().item()
    print(f"Initial: w = {model.w.data}, loss = {initial_loss:.6f}")
    
    def closure():
        optimizer.zero_grad()
        loss = model()
        loss.backward(create_graph=True)
        return loss
    
    loss_after = optimizer.step(closure)
    
    final_w = model.w.data
    final_error = torch.norm(final_w)
    
    print(f"Final: w = {final_w}, loss = {loss_after:.6f}, error: {final_error:.6f}")
    
    # With correct custom Hessian and no clipping, we should reach near [0,0]
    assert final_error < 1e-5, f"Failed to reach optimum, error: {final_error}"
    print("✓ Custom Hessian test passed!\n")


def test_custom_hessian_detailed():
    """Test custom Hessian with detailed debugging in the optimizer"""
    import torch
    print("=== Test: Custom Hessian Detailed ===")
    
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.tensor([2.0, 3.0]))
            
        def forward(self):
            return 0.5 * (2*self.w[0]**2 + 3*self.w[1]**2)
    
    custom_hessian_calls = [0]  # Counter to track calls
    
    def custom_hessian_fn():
        custom_hessian_calls[0] += 1
        print(f"CUSTOM HESSIAN CALLED! (call #{custom_hessian_calls[0]})")
        hessian = torch.diag(torch.tensor([2.0, 3.0]))
        print(f"Returning Hessian:\n{hessian}")
        return hessian
    
    model = SimpleModel()
    
    # Add debugging to the optimizer step method temporarily
    original_step = NewtonOptimizer.step
    
    def debug_step(self, closure=None, *hessian_args, **hessian_kwargs):
        print("OPTIMIZER STEP CALLED")
        for group in self.param_groups:
            print(f"Group custom_hessian_fn: {group.get('custom_hessian_fn', 'NOT FOUND')}")
        return original_step(self, closure, *hessian_args, **hessian_kwargs)
    
    NewtonOptimizer.step = debug_step
    
    optimizer = NewtonOptimizer(
        model.parameters(),
        custom_hessian_fn=custom_hessian_fn,
        damping=1e-6,
        max_update_norm=10.0
    )
    
    def closure():
        optimizer.zero_grad()
        loss = model()
        loss.backward(create_graph=True)
        return loss
    
    print("Before step:")
    print(f"Initial w: {model.w.data}")
    initial_loss = closure()
    print(f"Initial loss: {initial_loss.item()}")
    print(f"Gradient: {model.w.grad}")
    
    final_loss = optimizer.step(closure)
    
    print(f"After step:")
    print(f"Final w: {model.w.data}") 
    print(f"Final loss: {final_loss.item()}")
    print(f"Custom Hessian was called {custom_hessian_calls[0]} times")
    
    # Restore original method
    NewtonOptimizer.step = original_step
    
    error = torch.norm(model.w.data)
    if error < 1e-5:
        print("✅ Custom Hessian test passed!")
    else:
        print(f"❌ Custom Hessian test failed, error: {error:.6f}")

def test_parameter_groups():
    """Test different settings for parameter groups"""
    print("=== Test 5: Parameter Groups ===")
        
    class MultiLayerModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(2, 3)
            self.layer2 = nn.Linear(3, 1)
                
        def forward(self, x):
            return self.layer2(ch.relu(self.layer1(x)))
        
    model = MultiLayerModel()
        
    # Different damping for different layers
    optimizer = NewtonOptimizer([
        {'params': model.layer1.parameters(), 'damping': 1e-2},
        {'params': model.layer2.parameters(), 'damping': 1e-4}
    ], damping=1e-3)  # Global default
        
    # Generate some dummy data
    x = ch.randn(10, 2)
    y = ch.randn(10, 1)
    criterion = nn.MSELoss()
        
    # Test that optimization runs without errors
    try:
        def closure():
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward(create_graph=True)
            return loss
            
        initial_loss = closure()
        final_loss = optimizer.step(closure)
            
        print(f"Initial loss: {initial_loss:.6f}")
        print(f"Final loss: {final_loss:.6f}")
        print("✓ Parameter groups test passed!\n")
            
    except Exception as e:
        assert False, f"Parameter groups test failed: {e}"

def test_parameter_groups_debug():
    """Debug parameter groups with detailed output"""
    import torch
    print("=== Debug: Parameter Groups ===")
    
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(2, 2)  # Simpler model
            self.layer2 = nn.Linear(2, 1)
            
        def forward(self, x):
            return self.layer2(torch.relu(self.layer1(x)))
    
    model = SimpleModel()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    layer1_params = sum(p.numel() for p in model.layer1.parameters()) 
    layer2_params = sum(p.numel() for p in model.layer2.parameters())
    
    print(f"Total parameters: {total_params}")
    print(f"Layer 1 parameters: {layer1_params} (weights: 2x2=4, bias: 2)")
    print(f"Layer 2 parameters: {layer2_params} (weights: 2x1=2, bias: 1)")
    
    optimizer = NewtonOptimizer([
        {'params': model.layer1.parameters(), 'damping': 1e-2},
        {'params': model.layer2.parameters(), 'damping': 1e-4}
    ], damping=1e-3)
    
    # Simple data
    x = torch.randn(5, 2)
    y = torch.randn(5, 1)
    criterion = nn.MSELoss()
    
    def closure():
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward(create_graph=True)
        return loss
    
    print("\nBefore optimization:")
    initial_loss = closure()
    print(f"Initial loss: {initial_loss.item():.6f}")
    
    # Check gradients
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = torch.norm(param.grad).item()
            print(f"  {name}: grad norm = {grad_norm:.6f}")
    
    # Take one step with detailed debugging
    print("\nTaking optimization step...")
    final_loss = optimizer.step(closure)
    
    print(f"Final loss: {final_loss.item():.6f}")
    print(f"Loss change: {final_loss.item() - initial_loss.item():.6f}")
    
    # Check if loss decreased
    if final_loss < initial_loss:
        print("✅ Loss decreased - optimizer working!")
    else:
        print("❌ Loss increased - something's wrong!")
    
def test_convergence_plots():
    """Generate convergence plots for visualization"""
    print("=== Test 6: Convergence Analysis ===")
        
    # Simple quadratic
    class Quadratic1D(nn.Module):
        def __init__(self):
            super().__init__()
            self.x = nn.Parameter(ch.tensor([5.0]))
                
        def forward(self):
            return 0.5 * (self.x - 2.0)**2
        
    model = Quadratic1D()
    optimizer = NewtonOptimizer(model.parameters(), damping=1e-6)
        
    newton_losses = []
    positions = []
        
    for step in range(5):
        def closure():
            optimizer.zero_grad()
            loss = model()
            loss.backward()
            return loss
            
        loss = optimizer.step(closure)
        newton_losses.append(loss.detach())
        positions.append(model.x.item())
        
    # Compare with gradient descent
    model_gd = Quadratic1D()
    optimizer_gd = ch.optim.SGD(model_gd.parameters(), lr=0.1)
        
    gd_losses = []
    for step in range(50):  # More steps needed for GD
        optimizer_gd.zero_grad()
        loss = model_gd()
        loss.backward()
        optimizer_gd.step()
        gd_losses.append(loss.item())
        
    plt.figure(figsize=(12, 4))
        
    plt.subplot(1, 2, 1)
    plt.plot(newton_losses, 'o-', label='Newton', linewidth=2)
    plt.plot(gd_losses[:10], 's-', label='GD (first 10 steps)', alpha=0.7)
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Loss (log scale)')
    plt.title('Convergence Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
        
    plt.subplot(1, 2, 2)
    x_vals = np.linspace(-1, 6, 100)
    y_vals = 0.5 * (x_vals - 2)**2
    plt.plot(x_vals, y_vals, 'k-', label='Objective function', alpha=0.5)
    plt.plot(positions, [0.5*(p-2)**2 for p in positions], 'ro-', 
            label='Newton optimization path', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Optimization Path')
    plt.legend()
    plt.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig('newton_convergence_test.png', dpi=150, bbox_inches='tight')
    plt.show()
        
    print("✓ Convergence analysis completed!")
    print("  Check 'newton_convergence_test.png' for visualization\n")


def test_debug_custom_hessian():
    import torch
    """Debug version of the failing test"""
    print("=== DEBUG: Custom Hessian Test ===")
    
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.tensor([2.0, 3.0]))
            
        def forward(self):
            return 0.5 * (2*self.w[0]**2 + 3*self.w[1]**2)
    
    def custom_hessian_fn():
        print("✓ Custom Hessian function called!")
        hessian = torch.diag(torch.tensor([2.0, 3.0]))
        print(f"Custom Hessian: {hessian}")
        return hessian
    
    model = SimpleModel()
    
    # Add debugging to the optimizer
    original_step = NewtonOptimizer.step
    
    def debug_step(self, closure, *hessian_args, **hessian_kwargs):
        print("=== OPTIMIZER STEP DEBUG ===")
        
        for group in self.param_groups:
            print(f"Group keys: {group.keys()}")
            custom_hessian_fn = group.get('custom_hessian_fn')
            print(f"Custom Hessian function present: {custom_hessian_fn is not None}")
            
            params = [p for p in group['params'] if p.grad is not None]
            print(f"Number of parameters with gradients: {len(params)}")
            
            if params:
                grads_flat = torch.cat([p.grad.flatten() for p in params])
                print(f"Gradients shape: {grads_flat.shape}")
                print(f"Gradients: {grads_flat}")
                
                if custom_hessian_fn is not None:
                    print("Calling custom Hessian function...")
                    hessian = custom_hessian_fn(*hessian_args, **hessian_kwargs)
                    print(f"Hessian shape: {hessian.shape}")
                    print(f"Hessian: {hessian}")
                    
                    # Check if shapes match
                    if hessian.shape[0] != grads_flat.shape[0]:
                        print(f"❌ SHAPE MISMATCH: Hessian {hessian.shape} vs gradients {grads_flat.shape}")
                    
                    # Compute expected update
                    expected_update = torch.linalg.solve(hessian, -grads_flat)
                    print(f"Expected update: {expected_update}")
        
        return original_step(self, closure, *hessian_args, **hessian_kwargs)
    
    NewtonOptimizer.step = debug_step
    
    optimizer = NewtonOptimizer(
        model.parameters(), 
        custom_hessian_fn=custom_hessian_fn,
        damping=1e-6,
        max_update_norm=10.0
    )
    
    print(f"Initial w: {model.w.data}")
    print(f"Initial loss: {model().item()}")
    
    def closure():
        optimizer.zero_grad()
        loss = model()
        loss.backward(create_graph=True)
        print(f"Gradients after backward: w.grad = {model.w.grad}")
        return loss
    
    loss_after = optimizer.step(closure)
    print(f"Final w: {model.w.data}")
    print(f"Final loss: {loss_after.item()}")
    
    # Restore original method
    NewtonOptimizer.step = original_step

def test_rigorous_newton_vs_sgd_comparison():
    import torch
    """Rigorous comparison on a problem where Newton should excel"""
    torch.manual_seed(42)
    
    print("=== RIGOROUS NEWTON vs SGD COMPARISON ===")
    
    # Simple 2D quadratic where we know Newton should converge in 1 step
    class QuadraticModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.tensor([3.0, 2.0]))  # Start away from optimum
            
        def forward(self):
            # f(w) = 0.5 * w^T H w, where H = [[4, 1], [1, 3]]
            H = torch.tensor([[4.0, 1.0], [1.0, 3.0]])
            return 0.5 * (self.w @ H @ self.w)
    
    # Analytical Hessian
    def exact_hessian():
        return torch.tensor([[4.0, 1.0], [1.0, 3.0]])
    
    # Test Newton
    model_newton = QuadraticModel()
    initial_w_newton = model_newton.w.data.clone()
    
    optimizer_newton = NewtonOptimizer(
        model_newton.parameters(),
        custom_hessian_fn=exact_hessian,
        damping=1e-6,
        max_update_norm=10.0,
        lr=1.0
    )
    
    def newton_closure():
        optimizer_newton.zero_grad()
        loss = model_newton()
        loss.backward(create_graph=True)
        return loss
    
    newton_loss_before = model_newton().item()
    optimizer_newton.step(newton_closure)
    newton_loss_after = model_newton().item()
    newton_update = model_newton.w.data - initial_w_newton
    
    # Test SGD
    model_sgd = QuadraticModel()
    initial_w_sgd = model_sgd.w.data.clone()
    
    optimizer_sgd = torch.optim.SGD(model_sgd.parameters(), lr=0.01)
    
    def sgd_closure():
        optimizer_sgd.zero_grad()
        loss = model_sgd()
        loss.backward()
        return loss
    
    sgd_loss_before = model_sgd().item()
    sgd_closure()  # Compute gradients
    optimizer_sgd.step()
    sgd_loss_after = model_sgd().item()
    sgd_update = model_sgd.w.data - initial_w_sgd
    
    print("RESULTS:")
    print(f"Newton - Loss: {newton_loss_before:.6f} -> {newton_loss_after:.6f}")
    print(f"Newton - Update: {newton_update}")
    print(f"SGD    - Loss: {sgd_loss_before:.6f} -> {sgd_loss_after:.6f}") 
    print(f"SGD    - Update: {sgd_update}")
    
    # Newton should reach near-zero loss in one step for quadratic
    if newton_loss_after < 1e-6:
        print("✓ Newton works perfectly on quadratic!")
    else:
        print(f"❌ Newton failed on simple quadratic! Final loss: {newton_loss_after}")
    
    return newton_loss_after < sgd_loss_after
