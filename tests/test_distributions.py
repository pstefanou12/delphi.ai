# distribution tests 

import unittest
import torch as ch
from torch import Tensor
from torch.distributions import MultivariateNormal, Uniform
from torch.distributions.kl import kl_divergence
from torch.distributions.multivariate_normal import _batch_mahalanobis
from torch.linalg import cholesky, solve_triangular
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

class TestDistributions(unittest.TestCase): 
    """
    Test suite for the distribution module.
    """
    # right truncated normal distribution with known truncation and known variance
    def test_truncated_normal_known_variance(self):
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
        train_kwargs = Parameters({
                                'phi': phi_std_norm, 
                                'alpha': alpha,
                                'epochs': 3, 
                                'batch_size': 10, 
                                'covariance_matrix': ch.eye(1),
                                'trials': 1
                        }) 
        censored = distributions.CensoredNormal(train_kwargs)
        censored.fit(S_std_norm)
        # rescale distribution
        rescale_loc = censored.loc_ + emp_loc
        print(f"pred loc: {rescale_loc}")
        rescale_var = censored.variance_
        print(f"pred var: {rescale_var}")
        m = MultivariateNormal(rescale_loc, rescale_var)
        
        # check performance
        kl_censored = kl_divergence(m, M)
        self.assertTrue(kl_censored <= 1e-1)

    # right truncated normal distribution with known truncation
    def test_truncated_normal(self):
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
        train_kwargs = Parameters({
                                'phi': phi_std_norm, 
                                'alpha': alpha,
                                'epochs': 1, 
                                'batch_size': 10, 
                                'trials': 1,
                                'early_stopping': True,
                        }) 
        censored = distributions.CensoredNormal(train_kwargs)
        censored.fit(S_std_norm)
        # rescale distribution
        rescale_loc = censored.loc_ * emp_scale + emp_loc
        print(f"pred loc: {rescale_loc}")
        rescale_var = censored.variance_ * emp_var
        print(f"pred var: {rescale_var}")
        m = MultivariateNormal(rescale_loc, rescale_var)
        
        # check performance
        kl_censored = kl_divergence(m, M)
        self.assertTrue(kl_censored <= 1e-1)

    # sphere truncated multivariate normal distribution (10 D) with known truncation and covariance matrix
    def test_truncated_multivariate_normal_known_covariance_matrix(self):
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
        censored = distributions.CensoredMultivariateNormal(train_kwargs)
        censored.fit(S_norm)
        # rescale distribution
        rescale_loc = censored.loc_ + emp_loc
        print(f'pred loc: {rescale_loc}')
        print(f'pred covariance matrix: {M.covariance_matrix}')
        m = MultivariateNormal(rescale_loc, censored.covariance_matrix_)
        
        # check performance
        kl_censored = kl_divergence(m, M)
        self.assertTrue(kl_censored <= 2e-1)

    # sphere truncated multivariate normal distribution (10 D) with known truncation
    def test_truncated_multivariate_normal(self):
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
            print(f"Min eigenvalue: {eigenvalues.min()}")
            print(f"Max eigenvalue: {eigenvalues.max()}")
            print(f"Condition number: {eigenvalues.max() / eigenvalues.max(0)[0]}")
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
        censored = distributions.CensoredMultivariateNormal(train_kwargs)
        censored.fit(S_norm)
        # rescale distribution
        rescale_loc = censored.loc_ @ emp_scale + emp_loc
        rescale_cov = emp_scale @ censored.covariance_matrix_ @ emp_scale.T
        print(f'pred loc: {rescale_loc}')
        print(f'pred covariance matrix: {rescale_cov}')
        m = MultivariateNormal(rescale_loc, rescale_cov)
        
        # check performance
        kl_censored = kl_divergence(m, M)
        self.assertTrue(kl_censored <= 2e-1)

    # right truncated 1D normal distribution with unknown truncation
    def test_unknown_truncation_normal(self):
        M = MultivariateNormal(ch.zeros(1), ch.eye(1)) 
        samples = M.rsample([1000,])
        # generate ground-truth data
        phi = oracle.Right_Distribution(Tensor([0.0]))
        indices = phi(samples).nonzero()[:,0]
        S = samples[indices]
        alpha = S.size(0) / samples.size(0)
        emp_loc = S.mean(0)
        emp_var = S.var(0, keepdim=True)
        emp_scale = ch.sqrt(emp_var)
        S_norm = (S - emp_loc) / emp_scale

        # train algorithm
        train_kwargs = Parameters({'alpha': alpha,
                                'epochs': 25}) 
        truncated = distributions.TruncatedNormal(train_kwargs)
        truncated.fit(S_norm)
        # rescale distribution
        rescale_loc = truncated.loc_ @ emp_scale + emp_loc
        rescale_var = truncated.variance_ @ emp_var
        m = MultivariateNormal(rescale_loc, rescale_var)
        
        # check distribution parameter estimates
        kl_censored = kl_divergence(m, M)
        self.assertTrue(kl_censored <= 1e-1) 

    # 10D sphere truncated multivariate normal distribution with unknown truncation
    def test_unknown_truncation_multivariate_normal(self):
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
        truncated = distributions.TruncatedMultivariateNormal(train_kwargs)
        truncated.fit(S)
        # rescale distribution
        rescale_loc = truncated.loc_
        rescale_var = truncated.covariance_matrix_
        m = MultivariateNormal(rescale_loc, rescale_var)
        
        # check distribution parameter estimates
        kl_censored = kl_divergence(m, M)
        self.assertTrue(kl_censored <= 1e-1)  


    def test_truncated_bernoulli(self): 
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


if __name__ == '__main__':
    unittest.main(verbosity=2)  # Adding verbosity to see more test details
