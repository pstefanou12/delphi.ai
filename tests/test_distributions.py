# distribution tests 

import unittest
import torch as ch
from torch import Tensor
from torch.distributions import MultivariateNormal, Uniform
from torch.distributions.kl import kl_divergence
from torch.distributions.multivariate_normal import _batch_mahalanobis
from scipy.linalg import sqrtm

from delphi import distributions 
from delphi import oracle
from delphi.utils.helpers import Parameters, cov


class UnNorm_Sphere(oracle.oracle):
    """
    Spherical truncation
    """
    def __init__(self, covariance_matrix, centroid, radius, loc, cov):
        self._unbroadcasted_scale_tril = covariance_matrix.cholesky()
        self.centroid = centroid
        self.radius = radius
        self.loc, self.cov = loc, cov
        self.scale = ch.from_numpy(sqrtm(self.cov))

    def __call__(self, x):
        x_rescale = x @ self.scale + self.loc
        diff = x_rescale - self.centroid
        dist = ch.sqrt(_batch_mahalanobis(self._unbroadcasted_scale_tril, diff))
        return (dist < self.radius).float().flatten()

    def __str__(self): 
        return 'sphere'

class TestDistributions(unittest.TestCase): 
    """
    Test suite for the distribution module.
    """
    # right truncated normal distribution with known truncation
    def test_censored_normal(self):
        M = MultivariateNormal(ch.zeros(1), ch.eye(1)) 
        samples = M.rsample([1000,])
        # generate ground-truth data
        phi = oracle.Left_Distribution(Tensor([0.0]))
        # truncate
        indices = phi(samples).nonzero()[:,0]
        S = samples[indices]
        alpha = S.size(0) / samples.size(0)
        emp_loc = S.mean(0)
        emp_var = S.var(0)[...,None]
        emp_scale = ch.sqrt(emp_var)
    
        S_norm = (S - emp_loc) / emp_scale
        phi_norm = oracle.Left_Distribution(((phi.left - emp_loc) / emp_scale).flatten())
    
        # train algorithm
        train_kwargs = Parameters({'phi': phi_norm, 
                                'alpha': alpha,
                                'epochs': 5}) 
        censored = distributions.CensoredNormal(train_kwargs)
        censored.fit(S_norm)
        # rescale distribution
        rescale_loc = censored.loc_ * emp_scale + emp_loc
        rescale_var = censored.variance_ * emp_var
        m = MultivariateNormal(rescale_loc, rescale_var)
        
        # check performance
        kl_censored = kl_divergence(m, M)
        self.assertTrue(kl_censored <= 1e-1)

    # sphere truncated multivariate normal distribution (10 D) with known truncation
    def test_censored_multivariate_normal(self):
        M = MultivariateNormal(ch.zeros(10), ch.eye(10)) 
        samples = M.rsample([5000,])
        # generate ground-truth data
        W = Uniform(-.5, .5)
        centroid = W.sample([10,])
        phi = oracle.Sphere(M.covariance_matrix, centroid, 3.0)
        indices = phi(samples).nonzero()[:,0]
        S = samples[indices]
        alpha = S.size(0) / samples.size(0)
        emp_loc = S.mean(0)
        emp_cov = cov(S)
        emp_scale = ch.from_numpy(sqrtm(emp_cov))
        S_norm = (S - emp_loc) @ emp_scale.inverse()
        phi_norm = UnNorm_Sphere(M.covariance_matrix, phi.centroid, phi.radius, emp_loc, emp_cov)

        # train algorithm
        train_kwargs = Parameters({'phi': phi_norm, 
                                'alpha': alpha,
                                'epochs': 10}) 
        censored = distributions.CensoredMultivariateNormal(train_kwargs)
        censored.fit(S_norm)
        # rescale distribution
        rescale_loc = censored.loc_ @ emp_scale + emp_loc
        rescale_cov = censored.covariance_matrix_ @ emp_cov
        m = MultivariateNormal(rescale_loc, rescale_cov)
        
        # check performance
        kl_censored = kl_divergence(m, M)
        self.assertTrue(kl_censored <= 2e-1)

    # sphere truncated multivariate normal distribution (10 D) with known truncation
    def test_truncated_normal(self):
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
                                'epochs': 10}) 
        truncated = distributions.TruncatedNormal(train_kwargs)
        truncated.fit(S_norm)
        # rescale distribution
        rescale_loc = truncated.loc_ @ emp_scale + emp_loc
        rescale_var = truncated.variance_ @ emp_var
        m = MultivariateNormal(rescale_loc, rescale_var)
        
        # check distribution parameter estimates
        kl_censored = kl_divergence(m, M)
        self.assertTrue(kl_censored <= 1e-1) 




if __name__ == '__main__':
    unittest.main()
