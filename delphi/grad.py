"""
Gradients for truncated and untruncated latent variable models. 
"""

import torch as ch
from torch import sigmoid as sig
from torch.nn import Softmax
from torch.distributions import Gumbel, MultivariateNormal, Bernoulli
import math

from .utils.helpers import logistic, cov

softmax = Softmax(dim=1)

class TruncatedMultivariateNormalNLL(ch.autograd.Function):
    """
    Computes the truncated negative population log likelihood for truncated multivariate normal distribution with known truncation. 
    Function calculates the truncated negative log likelihood in the forward method and then calculates the 
    gradients with respect mu and cov in the backward method. When sampling from the conditional distribution, 
    we sample batch_size * num_samples samples, we then filter out the samples that remain in the truncation set, 
    and retainup to batch_size of the filtered samples. If there are fewer than batch_size number of samples remain,
    we provide a vector of zeros and calculate the untruncated log likelihood. 
    """
    @staticmethod
    def forward(ctx, T, v, data, phi, dims, trunc_multi_norm_score, sampler=None,
                 num_samples=100000, eps=1e-12):
        # reconstruct Sigma and mu
        sigma = ch.inverse(T)
        mu = (sigma @ v).view(dims)   # (d,)
        
        S = data[:, :dims]
        S_grad = data[:, dims:]

        sigma = T.inverse()
        L = ch.linalg.cholesky(sigma)

        # base Gaussian part
        quad = 0.5 * (S @ T * S).sum(dim=1)
        linear = (S @ v)
        base = quad - linear  # (batch,)

        # normalization stuff
        muT_T_mu = 0.5 * (mu @ v)
        const_term = 0.5 * dims * math.log(2 * math.pi)
        logdet_sigma = 2 * ch.log(ch.diagonal(L)).sum()
    
        if not sampler:             
            sampler = Sampler()   

        z = ch.Tensor([])
        num_sampled = 0

        while z.size(0) < num_samples:
            # draw samples per-observation so phi can depend on each sample individually if needed
            s = sampler.sample(mu, sigma, num_samples)
            mask = phi(s)
            z = ch.cat([z, s[mask.nonzero()[:,0]]])
            num_sampled += num_samples

        p_hat = ch.Tensor([z.size(0) / num_sampled])
        if p_hat < .01:
            print(f'acceptance rate: {p_hat.item()}')
                        
        log_p_hat = ch.log(p_hat + 1e-12)

        log_I = muT_T_mu + const_term + 0.5 * logdet_sigma + log_p_hat

        # final NLL 
        nll = base + log_I
        ctx.save_for_backward(S_grad, z)
        ctx.trunc_multi_norm_score = trunc_multi_norm_score
        ctx.dims = dims
        return nll.mean()

    @staticmethod
    def backward(ctx, grad_output):
        S_grad, s = ctx.saved_tensors
        grad = (-S_grad + ctx.trunc_multi_norm_score(s[:S_grad.size(0)]))

        cov_grad = grad[:,:ctx.dims**2].view(grad.size(0), ctx.dims, ctx.dims)
        loc_grad = grad[:,ctx.dims**2:]

        return  cov_grad / S_grad.size(0), loc_grad / S_grad.size(0), None, None, None, None, None, None, None


class TruncatedMultivariateNormalScore: 
    def __init__(self, known_cov=False): 
        self.known_cov = known_cov

    def __call__(self, x):
        if self.known_cov: 
            return ch.cat([ch.zeros(x.size(0), x.size(1)**2), x], 1)
        # calculates the negative log-likelihood for one sample of a censored normal
        return ch.cat([-.5*ch.bmm(x.unsqueeze(2), x.unsqueeze(1)).flatten(1), x], 1)

class PreSampler: 
    def __init__(self, dims: int, num_samples: int=1000000): 
        self.dims = dims 
        self.num_samples = num_samples 
        self.samples = ch.randn(self.num_samples, self.dims)
        
    def sample(self, mu: ch.Tensor, sigma: ch.Tensor, num_samples: int):
        if num_samples > self.num_samples: 
            raise Exception(f"num samples: ({num_samples}) greater than number of samples presampled: ({self.num_samples})")
        rand_perm = ch.randint(self.num_samples, ch.Size([self.num_samples]))
        L = ch.linalg.cholesky(sigma)
        s = mu + self.samples @ L.T 
        return s[rand_perm][:num_samples]

    
class Sampler: 
    def sample(self, mu: ch.Tensor, sigma: ch.Tensor, num_samples: int): 
        samples = ch.randn(num_samples, mu.size(0))
        L = ch.linalg.cholesky(sigma)
        s = mu + samples @ L.T 
        return s 

def rejection_sampling(mu, sigma, phi, dims, batch_size, num_samples=1000): 
    M = MultivariateNormal(ch.zeros(dims), ch.eye(dims))
    L = ch.linalg.cholesky(sigma)

    accepted_samples = ch.Tensor([])
    num_sampled = 0

    while accepted_samples.size(0) < batch_size:
        # draw samples per-observation so phi can depend on each sample individually if needed
        samples = M.sample((num_samples,))  # -> (num_samples, dims) 
        s = mu + samples @ L.T 
        mask = phi(s) 
        accepted_samples = ch.cat([accepted_samples, s[mask.nonzero()[:,0]]])
        num_sampled += num_samples

    p_hat = ch.Tensor([accepted_samples.size(0) / num_sampled])
    if p_hat < .01:
        print(f'acceptance rate: {p_hat.item()}')

    return accepted_samples[:batch_size], p_hat


class UnknownTruncationMultivariateNormalNLL(ch.autograd.Function):
    """
    Computes the negative population log likelihood for truncated multivariate normal distribution with unknown truncation.
    Calculates the population log-likelihood for the current batch in the forward step, and 
    then calculates its gradient in the backwards step.
    """
    @staticmethod
    # def forward(ctx, u, B, x, pdf, loc_grad, cov_grad, phi, exp_h):
    def forward(ctx, params, data, phi, exp_h, dims, known_cov=False):
        """
        Args: 
            params (torch.Tensor): size (dims + dims ** 2,) - current reparameterized mean and covariance matrix estimates concatenated together
            data: (torch.Tensor): size() - precomupted gradient values for both the mean and covariance matrix
            u (torch.Tensor): size (dims,) - current reparameterized mean estimate
            B (torch.Tensor): size (dims, dims) - current reparameterized covariance matrix estimate
            x (torch.Tensor): size (batch_size, dims) - batch of dataset samples 
            pdf (torch.Tensor): size (batch_size, 1) - batch of pdf for dataset samples
            loc_grad (torch.Tensor): (batch_size, dims) - precomputed gradient for mean for batch
            cov_grad (torch.Tensor): (batch_size, dims * dims) - precomputed gradient for covariance matrix for batch 
            phi (oracle.UnknownGaussian): oracle object for learning truncation set 
            exp_h (Exp_h): helper class object for calculating exponential in the gradient
            dims (int): the dimension number 
            known_cov (bool): whether the covariance matrix is known; if so, provide 0 as gradient for covariance matrix
        """
        u = params[:dims]
        B = params[dims:].resize(dims, dims)
        x = data[:,0][...,None]
        pdf = data[:,1][...,None]
        loc_grad = data[:,2][...,None]
        cov_grad = data[:,3][...,None]
        exp = exp_h(u, B, x)
        psi = phi.psi_k(x)
        loss = exp * pdf * psi
        ctx.save_for_backward(loss, loc_grad, cov_grad, ch.Tensor([known_cov]), ch.Tensor([dims]))
        return loss.mean(0)

    @staticmethod
    def backward(ctx, grad_output):
        loss, loc_grad, cov_grad, known_cov, dims = ctx.saved_tensors
        term_one = (loc_grad * loss).mean(0)
        term_two = ((cov_grad.flatten(1) * loss).unflatten(1, ch.Size([loc_grad.size(1), loc_grad.size(1)]))).mean(0) if not known_cov else ch.zeros((int(dims),)) 
        return ch.cat([term_one.flatten(), term_two.flatten()]), None, None, None, None, None


# class TruncatedMSE(ch.autograd.Function):
#     """
#     Computes the gradient of the negative population log likelihood for truncated regression
#     with known noise variance.
#     """
#     @staticmethod
#     def forward(ctx, 
#                 pred, 
#                 targ, 
#                 phi, 
#                 noise_var, 
#                 num_samples=10, 
#                 eps=1e-5):
#         """
#         Args: 
#             pred (torch.Tensor): size (batch_size, 1) matrix for regression model predictions
#             targ (torch.Tensor): size (batch_size, 1) matrix for regression target predictions
#             phi (oracle.oracle): dependent variable membership oracle
#             noise_var (float): noise distribution variance parameter
#             num_samples (int): number of samples to generate per sample in batch in rejection sampling procedure
#             eps (float): denominator error constant to avoid divide by zero errors
#         """
#         stacked = pred[None, ...].repeat(num_samples, 1, 1)
#         noised = stacked + math.sqrt(noise_var) * ch.randn(stacked.size())        
#         filtered = phi(noised)
#         z = (filtered * noised).sum(dim=0) / (filtered.sum(dim=0) + eps)
#         integrand = ch.exp(-0.5 * (noised - pred).pow(2))
    
#         masked_integrand = integrand * filtered 
    
#         # Monte Carlo estimate of the integral
#         integral_estimate = masked_integrand.mean(dim=0)
    
#         # log(integral)
#         eps = 1e-10
#         log_integral = ch.log(integral_estimate + eps)
#         quadratic_loss = -0.5 * (targ - pred).pow(2)

#         ctx.save_for_backward(pred, targ, z)
#         return (quadratic_loss + log_integral).mean()


#     @staticmethod
#     def backward(ctx, 
#                 grad_output):
#         pred, targ, z = ctx.saved_tensors
#         return (z - targ) / pred.size(0), targ / pred.size(0), None, None, None, None
    

class TruncatedMSE(ch.autograd.Function):
    @staticmethod
    def forward(ctx, pred, targ, phi, noise_var, num_samples=10000, eps=1e-5):
        N = pred.shape[0]

        # Sample latent points
        stacked = pred.unsqueeze(1).repeat(1, num_samples, 1)
        noise = (noise_var ** 0.5) * ch.randn_like(stacked)
        noised = stacked + noise

        # Apply truncation
        mask = phi(noised).float()

        # Conditional mean E[Y | Y ∈ S]
        z_num = (mask * noised).sum(dim=1)
        z_den = mask.sum(dim=1) + eps
        z = z_num / z_den  # (N, 1)

        # Probability mass P(Y in S)
        P_hat = mask.mean(dim=1)  # (N, 1)

        # Save for backward
        ctx.save_for_backward(pred, targ, noised, mask, P_hat, z, noise_var)

        # Compute log-likelihood
        quadratic_loss = -0.5 * (targ - pred).pow(2)
        log_integral = ch.log(math.sqrt(2 * math.pi * noise_var) * P_hat + eps)
        # loss = (quadratic_loss - log_integral).mean()

        return log_integral - quadratic_loss

    @staticmethod
    def backward(ctx, grad_output):
        pred, targ, noised, mask, P_hat, z, noise_var = ctx.saved_tensors
        eps = 1e-8

        # Mean-centered samples
        centered = noised 

        # Monte Carlo estimate of E[(z - pred) phi(z)]
        correction_term = (mask * centered).mean(dim=1) / (P_hat + eps)

        # Gradient wrt pred
        grad_pred = (targ - correction_term) / noise_var

        # Multiply by grad_output (from upstream)
        grad_pred = grad_pred 

        return - grad_pred / pred.size(0), None, None, None, None, None


# class TruncatedUnknownVarianceMSE(ch.autograd.Function):
#     """
#     Computes the gradient of negative population log likelihood for truncated linear regression
#     with unknown noise variance.
#     """
#     @staticmethod
#     def forward(ctx, pred, targ, lambda_, phi, num_samples=100, eps=1e-5):
#         """
#         Args: 
#             pred (torch.Tensor): size (batch_size, 1) matrix for regression model predictions
#             targ (torch.Tensor): size (batch_size, 1) matrix for regression target predictions
#             lambda_ (float): current reparameterized variance estimate for noise distribution
#             phi (oracle.oracle): dependent variable membership oracle
#             num_samples (int): number of samples to generate per sample in batch in rejection sampling procedure
#             eps (float): denominator error constant to avoid divide by zero errors
#         """
#         sigma = 1 / ch.sqrt(lambda_)
#         stacked = pred[..., None].repeat(1, num_samples, 1)

#         noised = stacked + sigma * ch.randn(stacked.size())
#         filtered = phi(noised)
#         out = noised * filtered
#         z = out.sum(dim=1) / (filtered.sum(dim=1) + eps)
#         z_2 = out.pow(2).sum(dim=1) / (filtered.sum(dim=1) + eps)
#         nll = 0.5 * lambda_ * targ.pow(2)  - lambda_ * targ * pred
#         const = -0.5 * lambda_ * z_2 + z * pred * lambda_

#         # Correct NLL: -(1/2)log(λ) + (λ/2)(y - pred)²
#         nll = -0.5 * ch.log(lambda_) + 0.5 * lambda_ * (targ - pred).pow(2)
    
#         # Expectation correction term
#         const = -0.5 * lambda_ * z_2 + lambda_ * z * pred - 0.5 * lambda_ * pred.pow(2)

#         ctx.save_for_backward(pred, targ, lambda_, z, z_2)
#         return (nll + const).mean(0)

#     @staticmethod
#     def backward(ctx, grad_output):
#         pred, targ, lambda_, z, z_2 = ctx.saved_tensors
#         """
#         multiply the v gradient by lambda, because autograd computes 
#         v_grad*x*variance, thus need v_grad*(1/variance) to cancel variance
#         factor
#         """
#         lambda_grad = .5 * (targ.pow(2) - z_2)
#         lambda_grad = grad_output * (0.5 / lambda_ - 0.5 * (targ - pred).pow(2) + 0.5 * (pred.pow(2) - z_2)) / pred.size(0)
#         return lambda_ * (z - targ) / pred.size(0), None, lambda_grad / pred.size(0), None, None, None

# class TruncatedUnknownVarianceMSE(ch.autograd.Function):
#     """Corrected: L = λ/2 * [y² - 2yμ - E[z²] + 2μE[z]]"""
#     @staticmethod
#     def forward(ctx, pred, targ, lambda_, phi, num_samples=100, eps=1e-5):
#         sigma = 1 / ch.sqrt(lambda_)
#         stacked = pred[..., None].repeat(1, num_samples, 1)
#         noised = stacked + sigma * ch.randn(stacked.size())
#         filtered = phi(noised)
#         out = noised * filtered
#         z = out.sum(dim=1) / (filtered.sum(dim=1) + eps)
#         z_2 = out.pow(2).sum(dim=1) / (filtered.sum(dim=1) + eps)
        
#         # Correct formulation: λ/2 * [y² - 2yμ - E[z²] + 2μE[z]]
#         loss = 0.5 * lambda_ * (targ.pow(2) - 2*targ*pred - z_2 + 2*pred*z)
        
#         ctx.save_for_backward(pred, targ, lambda_, z, z_2)
#         return loss.mean(0)

#     @staticmethod
#     def backward(ctx, grad_output):
#         pred, targ, lambda_, z, z_2 = ctx.saved_tensors
        
#         # ∂L/∂μ = λ/2 * [-2y + 2E[z]]= λ(E[z] - y)
#         pred_grad = grad_output * lambda_ * (z - targ) / pred.size(0)
        
#         # ∂L/∂λ = 1/2 * [y² - 2yμ - E[z²] + 2μE[z]]
#         lambda_grad = grad_output * 0.5 * (targ.pow(2) - 2*targ*pred - z_2 + 2*pred*z) / pred.size(0)
        
#         return pred_grad, None, lambda_grad, None, None, None

# import torch
# import torch.nn.functional as F
# import math

# class TruncatedUnknownVarianceMSE(torch.autograd.Function):
#     """
#     Maximum Likelihood Estimator for Truncated Gaussian Regression
#     using Monte Carlo (MC) estimation for the arbitrary truncation set.
    
#     Optimization Variables: mu (pred) and lambda_ (1/sigma^2).
#     """
    
#     @staticmethod
#     def forward(ctx, pred, targ, lambda_, phi, num_samples=100, eps=1e-10):
#         # pred is the conditional mean (mu)
#         # targ is the observed target (y)
#         # lambda_ is the precision (1/sigma^2)
        
#         # --- 0. Prepare for MC Sampling ---
# #         import pdb; pdb.set_trace()
#         sigma_sq = 1.0 / lambda_
#         sigma = torch.sqrt(sigma_sq)
        
#         # --- 1. MC Estimation of Conditional Moments and Probability ---
#         # Generate samples from the *latent* Gaussian distribution
#         stacked = pred.repeat(1, num_samples) # Shape: [Batch, num_samples]
#         noised = stacked + sigma * torch.randn_like(stacked)
        
#         # Apply the arbitrary truncation filter phi(y)
#         filtered = phi(noised)
        
#         # Calculate MC estimates of E[Y | Y in S] and E[Y^2 | Y in S]
#         # E[Y | S] approx= (Sum Y_i * phi(Y_i)) / (Sum phi(Y_i))
#         weighted_y = noised * filtered
        
#         sum_weights = filtered.sum(dim=1, keepdim=True) # Sum of phi(Y_i)
        
#         # Conditional Mean E[Y | S] approx= z (Used in backward pass)
#         z = weighted_y.sum(dim=1, keepdim=True) / (sum_weights + eps)
        
#         # Conditional Second Moment E[Y^2 | S] approx= z_2 (Used in backward pass)
#         z_2 = weighted_y.pow(2).sum(dim=1, keepdim=True) / (sum_weights + eps)
        
#         # Conditional Probability P(Y in S) approx= P_hat (Used in loss)
#         P_hat = sum_weights / num_samples
        
#         # --- 2. Negative Log-Likelihood (NLL) Calculation ---
        
#         # The NLL is L = - [ Log(f_N(y | mu, lambda)) - Log(P(Y in S)) ]
        
#         # a) Analytic Log-Density Term (for observed y)
#         # log(f_N(y)) = 0.5 * log(lambda) - 0.5 * log(2*pi) - 0.5 * lambda * (y - mu)^2
#         log_f_analytic = 0.5 * torch.log(lambda_) - 0.5 * math.log(2*math.pi) \
#                        - 0.5 * lambda_ * (targ - pred).pow(2)
        
#         # b) MC Log-Normalization Term (for the arbitrary set S)
#         log_P_hat = torch.log(P_hat + eps)
        
#         # Final Loss: NLL = - [ Log(f_analytic) - Log(P_hat) ]
#         loss = - (log_f_analytic - log_P_hat)
        
#         # --- 3. Save for Backward Pass ---
#         ctx.save_for_backward(pred, targ, lambda_, z, z_2)
#         return loss / pred.size(0)

#     @staticmethod
#     def backward(ctx, grad_output):
#         pred, targ, lambda_, z, z_2 = ctx.saved_tensors
        
#         # Note: The gradient is the difference (analytic - MC) due to the negative sign in NLL
#         mu_grad = lambda_ * (z - targ)
        
#         # --- 2. Gradient w.r.t. Precision (lambda_ = 1/sigma^2) ---
#         # Derived as: ∂L/∂λ ∝ 0.5 * (1/λ) * [ (y - μ)^2 - (E[Y^2|S] - 2μE[Y|S] + μ^2) ]

#         # Analytic component squared residual (y - mu)^2
#         analytic_sq_res = (targ - pred).pow(2) 

#         # MC component squared residual E[(Y - mu)^2 | S] = E[Y^2|S] - 2mu*E[Y|S] + mu^2
#         mc_sq_res = z_2 - 2 * pred * z + pred.pow(2)
        
#         # Full lambda gradient
#         # lambda_grad = 0.5 * (1.0 / lambda_) * (analytic_sq_res - mc_sq_res)

#         lambda_grad = 0.5 * (analytic_sq_res - mc_sq_res)
        
#         # --- 3. Return Gradients ---
#         return mu_grad / pred.size(0), \
#                None, \
#                lambda_grad.mean(0, keepdim=True), \
#                None, None, None
    

import torch
import torch.nn.functional as F
import math

class TruncatedUnknownVarianceMSE(torch.autograd.Function):
    """
    Maximum Likelihood Estimator for Truncated Gaussian Regression
    using Monte Carlo (MC) estimation for the arbitrary truncation set.
    
    Optimization Variables: mu (pred) and lambda_ (1/sigma^2).
    """
    
    @staticmethod
    def forward(ctx, pred, targ, lambda_, phi, num_samples=100, eps=1e-10):
        # pred is the conditional mean (mu)
        # targ is the observed target (y)
        # lambda_ is the precision (1/sigma^2)
        
        # --- 0. Prepare for MC Sampling ---
        sigma_sq = 1.0 / lambda_
        sigma = torch.sqrt(sigma_sq)
        
        # --- 1. MC Estimation of Conditional Moments and Probability ---
        # Generate samples from the *latent* Gaussian distribution
        stacked = pred.repeat(1, num_samples) # Shape: [Batch, num_samples]
        noised = stacked + sigma * torch.randn_like(stacked)

        # Apply the arbitrary truncation filter phi(y)
        filtered = phi(noised)
        
        # Calculate MC estimates of E[Y | Y in S] and E[Y^2 | Y in S]
        # E[Y | S] approx= (Sum Y_i * phi(Y_i)) / (Sum phi(Y_i))
        weighted_y = noised * filtered
        
        sum_weights = filtered.sum(dim=1, keepdim=True) # Sum of phi(Y_i)
        
        # Conditional Mean E[Y | S] approx= z (Used in backward pass)
        z = weighted_y.sum(dim=1, keepdim=True) / (sum_weights + eps)
        
        # Conditional Second Moment E[Y^2 | S] approx= z_2 (Used in backward pass)
        z_2 = weighted_y.pow(2).sum(dim=1, keepdim=True) / (sum_weights + eps)
        
        # Conditional Probability P(Y in S) approx= P_hat (Used in loss)
        P_hat = sum_weights / num_samples
        # --- 2. Negative Log-Likelihood (NLL) Calculation ---
        
        # The NLL is L = - [ Log(f_N(y | mu, lambda)) - Log(P(Y in S)) ]
        
        # a) Analytic Log-Density Term (for observed y)
        # log(f_N(y)) = 0.5 * log(lambda) - 0.5 * log(2*pi) - 0.5 * lambda * (y - mu)^2
        log_f_analytic = 0.5 * torch.log(lambda_) - 0.5 * math.log(2*math.pi) \
                       - 0.5 * lambda_ * (targ - pred).pow(2)
        
        # b) MC Log-Normalization Term (for the arbitrary set S)
        log_P_hat = torch.log(P_hat + eps)
        
        # Final Loss: NLL = - [ Log(f_analytic) - Log(P_hat) ]
        loss = - (log_f_analytic - log_P_hat)
        
        # --- 3. Save for Backward Pass ---
        ctx.save_for_backward(pred, targ, lambda_, z, z_2)
        return loss / pred.size(0)

    @staticmethod
    def backward(ctx, grad_output):
        pred, targ, lambda_, z, z_2 = ctx.saved_tensors
        
        # Note: The gradient is the difference (analytic - MC) due to the negative sign in NLL
        mu_grad = lambda_ * (z - targ)
        
        # --- 2. Gradient w.r.t. Precision (lambda_ = 1/sigma^2) ---
        # Derived as: ∂L/∂λ ∝ 0.5 * (1/λ) * [ (y - μ)^2 - (E[Y^2|S] - 2μE[Y|S] + μ^2) ]

        # Analytic component squared residual (y - mu)^2
        analytic_sq_res = (targ - pred).pow(2) 

        # MC component squared residual E[(Y - mu)^2 | S] = E[Y^2|S] - 2mu*E[Y|S] + mu^2
        mc_sq_res = z_2 - 2 * pred * z + pred.pow(2)
        
        # Full lambda gradient
        lambda_grad = 0.5 * (1.0 / lambda_) * (analytic_sq_res - mc_sq_res)
#         lambda_grad = 0.5 * (analytic_sq_res - mc_sq_res)
        lambda_grad = 0.5 * (targ.pow(2).mean(0) - z_2.mean(0))[...,None]

        
        # --- 3. Return Gradients ---
        return mu_grad / pred.size(0), \
               None, \
               lambda_grad, \
               None, None, None
    

# class TruncatedUnknownVarianceMSE(ch.autograd.Function):
#     """Corrected: L = λ/2 * [y² - 2yμ - E[z²] + 2μE[z]]"""
#     @staticmethod
#     def forward(ctx, pred, targ, lambda_, phi, num_samples=100, eps=1e-5):
#         # sigma = 1 / ch.sqrt(lambda_)
#         sigma = ch.sqrt(1/lambda_)
#         stacked = pred[..., None].repeat(1, num_samples, 1)
#         noised = stacked + sigma * ch.randn(stacked.size())
#         filtered = phi(noised)
#         out = noised * filtered
#         z = out.sum(dim=1) / (filtered.sum(dim=1) + eps)
#         z_2 = out.pow(2).sum(dim=1) / (filtered.sum(dim=1) + eps)
                
#         P_hat = filtered.float().mean(dim=1)
        
#         m = pred # Assume pred is m = w^T x
#         mu = lambda_ * m
#         sigma_sq = 1.0 / lambda_

#         # 1. Term A (Quadratic term)
#         quadratic_term = 0.5 * (lambda_ * targ.pow(2) - 2 * targ * mu)
        
#         # 2. Term B (Log-Integral I(mu, lambda))
        
#         # Part 1: (1/2) * m^2 * lambda
#         m_sq_lambda_term = 0.5 * m.pow(2) * lambda_ 
        
#         # Part 2: log[ sqrt(2*pi*sigma^2) * P(Z in S) ]
#         log_normalization = ch.log(ch.sqrt(2 * math.pi * sigma_sq) * P_hat + eps)
        
#         log_integral = m_sq_lambda_term + log_normalization
        
#         # Final Loss: L = Term A + log_integral (Term B)
#         loss = quadratic_term + log_integral
        
#         # The return value for a log-likelihood minimization is usually the mean of the loss:
#         ctx.save_for_backward(pred, targ, lambda_, z, z_2)
#         return loss / pred.size(0)

#     @staticmethod
#     def backward(ctx, grad_output):
#         pred, targ, lambda_, z, z_2 = ctx.saved_tensors
        
#         # ∂L/∂μ = λ/2 * [-2y + 2E[z]]= λ(E[z] - y)
#         pred_grad = lambda_ * (z - targ) / pred.size(0)
        
#         # ∂L/∂λ = 1/2 * [y² - 2yμ - E[z²] + 2μE[z]]
#         # lambda_grad =  0.5 * (targ.pow(2) - 2*targ*pred - z_2 + 2*pred*z) / pred.size(0)
#         lambda_grad =  0.5 * (targ.pow(2) - z_2) / pred.size(0)

        
#         return pred_grad, None, lambda_grad, None, None, None


def Test(mu, phi, c_gamma, alpha, T): 
  """
  Test function that checks which gradient to take 
  at timestep t. 
  Args: 
    :param mu: current conditional mean for LDS 
    :param phi: oracle 
    :param c_gamma: constant 
    :param alpha: survival probability
    :param T: number of timesteps in dataset
  """
  M = ch.distributions.MultivariateNormal(ch.zeros(mu[0].size(0)), ch.eye(mu[0].size(0)))

  # threshold constant
  gamma = (alpha / 2) ** c_gamma

  # number of samples
  k = int((4 / gamma) * math.log(T))
  stacked = mu.repeat(k, 1, 1)
  noise = M.sample(stacked.size()[:-1])
  ci = stacked + noise
  p = phi(ci).float().mean(0)
  """
  check whether the probability that a sample falls within the 
  truncation set is greater than the survival probability
  """
  return (p >= (2 * gamma))

class SwitchGrad(ch.autograd.Function):
    """
    Computes the gradient of the negative population log likelihood for truncated regression
    with known noise variance.
    """
    @staticmethod
    def forward(ctx, pred, targ, phi, c_gamma, alpha, T, noise_var, num_samples=10, eps=1e-5):
        """
        Args: 
            pred (torch.Tensor): size (batch_size, d) matrix for regression model predictions
            targ (torch.Tensor): size (batch_size, d) matrix for regression target predictions
            phi (oracle.oracle): dependent variable membership oracle
            c_gamma (float) : large constant >= 0
            alpha (float) : survival probability
            T (int) : number of samples within dataset 
            noise_var (float): noise distribution variance parameter
            num_samples (int): number of samples to generate per sample in batch in rejection sampling procedure
            eps (float): denominator error constant to avoid divide by zero errors
        """
        # make num_samples copies of pred, N x B x 1
        stacked = pred[None, ...].repeat(num_samples, 1, 1)
        '''
        test whether to use censor-aware or censor-oblivious function 
        for computing gradient
        '''
        M = ch.distributions.MultivariateNormal(ch.zeros(pred[0].size(0)), noise_var) 
        result = Test(pred, phi, c_gamma, alpha, T)

        # add random noise to each copy
        noised = stacked + M.sample(stacked.size()[:-1])
        
        # filter out copies where pred is in bounds
        filtered = phi(noised)
        # average across truncated indices
        z_ = (filtered * noised).sum(dim=0) / (filtered.sum(dim=0) + eps)

        """
        result and result_inv are masks, so that you keep the noised 
        and the unnoised samples
        """
        z = result.float()*z_ + (~result).float()*pred

        ctx.save_for_backward(pred, targ, z)
        loss = -.5 * (targ - pred).norm(p=2, keepdim=True, dim=-1).pow(2) + \
                .5 * (z - pred).norm(p=2, keepdim=True, dim=-1).pow(2)
        return loss.mean(0)

    @staticmethod
    def backward(ctx, grad_output):
        pred, targ, z = ctx.saved_tensors
        return (z - targ) / pred.size(0), targ / pred.size(0), None, \
        None, None, None, None, None, None, None


class TruncatedBCE(ch.autograd.Function):
    """
    Truncated binary cross entropy gradient for truncated binary classification tasks. 
    """
    @staticmethod
    def forward(ctx, pred, targ, phi, num_samples=10, eps=1e-5):
        """
        Args: 
            pred (torch.Tensor): size (batch_size, 1) matrix for regression model predictions
            targ (torch.Tensor): size (batch_size, 1) matrix for regression target predictions
            phi (oracle.oracle): dependent variable membership oracle
            num_samples (int): number of samples to generate per sample in batch in rejection sampling procedure
            eps (float): denominator error constant to avoid divide by zero errors
        """
        ctx.save_for_backward()
        bce_loss = ch.nn.BCEWithLogitsLoss()
        
        stacked = pred[None, ...].repeat(num_samples, 1, 1)
        rand_noise = logistic.sample(stacked.size())
        # add noise
        noised = stacked + rand_noise
        noised_labs = noised >= 0
        # filter
        filtered = phi(noised)
        mask = (noised_labs).eq(targ)
        nll = (filtered * mask * logistic.log_prob(rand_noise)).sum(0) / ((filtered * mask).sum(0) + eps)
        const = (filtered * logistic.log_prob(rand_noise)).sum(0) / (filtered.sum(0) + eps)
        ctx.save_for_backward(mask, filtered, rand_noise)
        ctx.eps = eps
        return -(nll - const) / pred.size(0)

    @staticmethod
    def backward(ctx, grad_output):
        mask, filtered, rand_noise = ctx.saved_tensors

        avg = 2*(sig(rand_noise) * mask * filtered).sum(0) / ((mask * filtered).sum(0) + ctx.eps) 
        norm_const = (2 * sig(rand_noise) * filtered).sum(0) / (filtered.sum(0) + ctx.eps)
        return -(avg - norm_const) / rand_noise.size(1), None, None, None, None


class TruncatedProbitMLE(ch.autograd.Function): 
    @staticmethod
    def forward(ctx, pred, targ, phi, num_samples=10, eps=1e-5): 
        """
        Args: 
            pred (torch.Tensor): size (batch_size, 1) matrix for regression model predictions
            targ (torch.Tensor): size (batch_size, 1) matrix for regression target predictions
            phi (oracle.oracle): dependent variable membership oracle
            num_samples (int): number of samples to generate per sample in batch in rejection sampling procedure
            eps (float): denominator error constant to avoid divide by zero errors
        """
        M = MultivariateNormal(ch.zeros(1,), ch.eye(1, 1))
        stacked = pred[None,...].repeat(num_samples, 1, 1)
        rand_noise = ch.randn(stacked.size())
        noised = stacked + rand_noise 
        noised_labs = noised >= 0
        mask = noised_labs.eq(targ)
        filtered = phi(noised)
        pdf = M.log_prob(rand_noise)[...,None]
        nll = (pdf * filtered * mask).sum(0) / ((filtered * mask).sum(0) + eps)
        const = (filtered * pdf).sum(0) / (filtered.sum(0) + eps)
        ctx.save_for_backward(rand_noise, filtered, mask)
        ctx.eps = eps
        return -(nll - const) / pred.size(0)

    @staticmethod
    def backward(ctx, grad_output): 
        rand_noise, filtered, mask = ctx.saved_tensors
        nll = (mask * filtered * rand_noise).sum(dim=0) / ((mask * filtered).sum(dim=0) + ctx.eps)
        const = (rand_noise * filtered).sum(dim=0) / (filtered.sum(dim=0) + ctx.eps)
        return -(nll - const) / rand_noise.size(1), None, None, None, None


class GumbelCE(ch.autograd.Function):
    @staticmethod
    def forward(ctx, pred, targ, num_samples=1000, eps=1e-5):
        ctx.save_for_backward(pred, targ)
        ce_loss = ch.nn.CrossEntropyLoss()
        ctx.num_samples = num_samples
        ctx.eps = eps
        return ce_loss(pred, targ)

    @staticmethod
    def backward(ctx, grad_output):
        pred, targ = ctx.saved_tensors
        # gumbel distribution
        gumbel = Gumbel(0, 1)
        # make num_samples copies of pred logits
        stacked = pred[None, ...].repeat(ctx.num_samples, 1, 1)
        # add gumbel noise to logits
        rand_noise = gumbel.sample(stacked.size())
        noised = stacked + rand_noise 
        noised_labs = noised.argmax(-1)
        # remove the logits from the trials, where the kth logit is not the largest value
        mask = noised_labs.eq(targ)[..., None]
        inner_exp = 1 - ch.exp(-rand_noise)
        avg = (inner_exp * mask).sum(0) / (mask.sum(0) + ctx.eps) / pred.size(0)
        return -avg , None, None, None


class TruncatedCE(ch.autograd.Function):
    @staticmethod
    def forward(ctx, pred, targ, phi, num_samples=1000, eps=1e-5):
        """
        Args: 
            pred (torch.Tensor): size (batch_size, 1) matrix for regression model predictions
            targ (torch.Tensor): size (batch_size, 1) matrix for regression target predictions
            phi (oracle.oracle): dependent variable membership oracle
            num_samples (int): number of samples to generate per sample in batch in rejection sampling procedure
            eps (float): denominator error constant to avoid divide by zero errors
        """
        ctx.save_for_backward(pred, targ)
        ctx.phi = phi
        ctx.num_samples = num_samples
        ctx.eps = eps
        ce_loss = ch.nn.CrossEntropyLoss()
        return ce_loss(pred, targ)
        '''
        # initialize gumbel distribution
        gumbel = Gumbel(0, 1)
        # make num_samples copies of pred logits
        stacked = pred[None, ...].repeat(num_samples, 1, 1)
        # add gumbel noise to logits
        rand_noise = gumbel.sample(stacked.size())
        noised = stacked + rand_noise 
        # truncate - if one of the noisy logits does not fall within the truncation set, remove it
        filtered = phi(noised)
        noised_labs = noised.argmax(-1)
        # mask takes care of invalid logits and truncation set
        mask = noised_labs.eq(targ)[..., None] * filtered
        
        nll = (gumbel.log_prob(rand_noise) * mask).sum(dim=-1)[...,None].sum(dim=0) / (mask.sum(dim=0) + eps)
        const = (gumbel.log_prob(rand_noise) * filtered).sum(dim=-1)[..., None].sum(dim=0) / (filtered.sum(dim=0) + eps) 
        return -(nll - const) / pred.size(0)
        '''
    @staticmethod
    def backward(ctx, grad_output):  
        #if isinstance(ctx.phi, oracle.Left_Distribution):
        #    import pdb; pdb.set_trace()
        pred, targ = ctx.saved_tensors
        # initialize gumbel distribution
        gumbel = Gumbel(0, 1)
        # make num_samples copies of pred logits
        stacked = pred[None, ...].repeat(ctx.num_samples, 1, 1)
        # add gumbel noise to logits
        rand_noise = gumbel.sample(stacked.size())
        noised = stacked + rand_noise 
        # truncate - if one of the noisy logits does not fall within the truncation set, remove it
        filtered = ctx.phi(noised)
        # noised_labs = softmax(stacked).argmax(-1)
        noised_labs = noised.argmax(-1)
        # mask takes care of invalid logits and truncation set
        mask = noised_labs.eq(targ)[..., None] * filtered
        inner_exp = (1 - ch.exp(-rand_noise))
        nll = ((inner_exp * mask).sum(0) / (mask.sum(0) + ctx.eps))
        const = ((inner_exp * filtered).sum(0) / (filtered.sum(0) + ctx.eps))
        return (-nll + const) / pred.size(0), None, None, None, None


class TruncatedBooleanProductNLL(ch.autograd.Function):
    """
    Computes the truncated negative population log likelihood for a truncated boolean product distribution. 
    Function calculates the truncated negative log likelihood in the forward method and then calculates the 
    gradients with respect to p in the backward method. When sampling from the conditional distribution, 
    we sample batch_size * num_samples samples, we then filter out the samples that remain in the truncation set, 
    and retainup to batch_size of the filtered samples. If there are fewer than batch_size number of samples remain,
    we provide a vector of zeros and calculate the untruncated log likelihood. 
    """
    @staticmethod
    def forward(ctx, z, x, phi, num_samples=10):
        """
        Args: 
            p (torch.Tensor): current logit probability vector estimate for truncated boolean product distribution
            x (torch.Tensor): batch_size * dims, sample batch 
            phi (delphi.oracle): oracle for truncated boolean product distribution
            num_samples (int): number of samples to sample for each sample in batch
        """
        # reparameterize distribution
        B = Bernoulli(logits=z)
        # sample num_samples * batch size samples from distribution
        s = B.sample([num_samples * x.size(0)])
        filtered = phi(s).nonzero(as_tuple=True)
        # z is a tensor of size batch size zeros, then fill with up to batch size num samples
        y = ch.zeros(x.size())
        elts = s[filtered][:x.size(0)]
        y[:elts.size(0)] = elts
        # standard negative log likelihood
        nll = -x*z
        const = ch.log(ch.exp(y*z).sum(0))
        ctx.save_for_backward(x, y)
        return (nll + const) / x.size(0)

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        # calculate gradient
        return (-x + y) / x.size(0), None, None, None, None
