"""
Newton Optimizer for newton raphson SGD optimization.
"""

import torch
from torch.optim import Optimizer
from typing import Callable, Optional, List, Union


class NewtonOptimizer(Optimizer):
    def __init__(self, params, lr: Union[float, torch.Tensor]=1e-3, custom_hessian_fn: Optional[Callable] = None,
                 damping: float = 1e-3, hessian_approx: str = 'diagonal',
                 max_update_norm: float = 1.0):
        defaults = dict(lr=lr, damping=damping, hessian_approx=hessian_approx,
                       custom_hessian_fn=custom_hessian_fn, max_update_norm=max_update_norm)
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None, *hessian_args, **hessian_kwargs):
        """
        Performs a single optimization step with safeguards for neural networks
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = [p for p in group['params'] if p.grad is not None]
            if not params:
                continue
                
            damping = group.get('damping', self.defaults['damping'])
            hessian_approx = group.get('hessian_approx', self.defaults['hessian_approx'])
            custom_hessian_fn = group.get('custom_hessian_fn', self.defaults['custom_hessian_fn'])
            max_update_norm = group.get('max_update_norm', self.defaults['max_update_norm'])
            
            grads_flat = torch.cat([p.grad.flatten() for p in params])
            
            # Compute Hessian
            if custom_hessian_fn is not None:
                hessian = custom_hessian_fn(*hessian_args, **hessian_kwargs)
            else:
                hessian = self._compute_hessian_auto(params, grads_flat, hessian_approx)
            
            # Newton update with safeguards
            with torch.no_grad():
                update = self._compute_newton_update(grads_flat, hessian, damping)
                lr = group.get('lr', self.defaults['lr'])
                update = lr * update  # Scale Newton step by learning rate
                
                # Clip update to prevent explosion
                update_norm = torch.norm(update)
                if update_norm > max_update_norm:
                    print(f"Clipping update norm: {update_norm:.6f} -> {max_update_norm:.6f}")
                    update = update * (max_update_norm / update_norm)
                
                self._apply_update(update, params)
            
        return loss

    def _compute_hessian_auto(self, params: List[torch.Tensor], grads_flat: torch.Tensor, 
                            hessian_approx: str) -> torch.Tensor:
        """Compute Hessian with neural network safeguards"""
        n_params = grads_flat.shape[0]
        
        # For neural networks, always use diagonal approximation
        # Full Hessian is too expensive and often ill-conditioned
        if hessian_approx == 'diagonal' or n_params > 10:  # Small threshold for neural networks
            return self._compute_diagonal_hessian_safe(params, grads_flat)
        else:
            return self._compute_full_hessian_safe(params, grads_flat)

    def _compute_diagonal_hessian_safe(self, params: List[torch.Tensor], grads_flat: torch.Tensor) -> torch.Tensor:
        """Safe diagonal Hessian computation for neural networks"""
        n_params = grads_flat.shape[0]
        hessian_diag = torch.ones(n_params, device=grads_flat.device)  # Start with identity
        
        # Only compute diagonal for a subset to save computation
        # Or use a fixed fraction of parameters
        compute_indices = list(range(0, n_params, max(1, n_params // 10)))  # Sample 10%
        
        for i in compute_indices:
            try:
                grad2 = torch.autograd.grad(
                    grads_flat[i], 
                    params, 
                    retain_graph=True,
                    allow_unused=True
                )
                
                grad2_flat = torch.cat([
                    g.flatten() if g is not None else torch.zeros_like(p).flatten()
                    for g, p in zip(grad2, params)
                ])
                
                if i < grad2_flat.shape[0]:
                    hessian_diag[i] = grad2_flat[i].item()
                    # Ensure positive definiteness
                    hessian_diag[i] = max(hessian_diag[i], 1e-6)
            except:
                pass  # Keep default value of 1.0
        
        return torch.diag(hessian_diag)
    
    def _compute_full_hessian_safe(self, params: List[torch.Tensor], grads_flat: torch.Tensor) -> torch.Tensor:
        """Safer full Hessian computation"""
        n_params = grads_flat.shape[0]
        hessian = torch.zeros(n_params, n_params, device=grads_flat.device)
        
        for i in range(n_params):
            try:
                grad2 = torch.autograd.grad(
                    grads_flat[i], 
                    params, 
                    retain_graph=True,
                    allow_unused=True
                )
                
                grad2_flat = torch.cat([
                    g.flatten() if g is not None else torch.zeros_like(p).flatten() 
                    for g, p in zip(grad2, params)
                ])
                
                hessian[i, :] = grad2_flat
            except Exception as e:
                hessian[i, i] = 1.0
        
        return hessian
    
    @torch.no_grad()
    def _compute_newton_update(self, grads_flat: torch.Tensor, hessian: torch.Tensor, 
                             damping: float) -> torch.Tensor:
        """Compute Newton update: Δθ = -H⁻¹g"""
        hessian_reg = hessian + damping * torch.eye(hessian.shape[0], device=hessian.device)
        try:
            return torch.linalg.solve(hessian_reg, -grads_flat)
        except torch.linalg.LinAlgError:
            return torch.linalg.lstsq(hessian_reg, -grads_flat).solution

    @torch.no_grad()
    def _apply_update(self, update: torch.Tensor, params: List[torch.Tensor]):
        """Apply update to parameters: θ_new = θ_old + Δθ"""
        start_idx = 0
        for param in params:
            end_idx = start_idx + param.numel()
            param_update = update[start_idx:end_idx].reshape(param.shape)
            param.data.add_(param_update)
            start_idx = end_idx