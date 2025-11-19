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

ch.set_printoptions(precision=4, sci_mode=False)


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
    print(f"emp loc:\n {emp_loc}")
    print(f"known variance:\n {ch.eye(1)}")
    
    S_std_norm = (S - emp_loc) 
    phi_std_norm = oracle.Left_Distribution((phi.left - emp_loc).flatten())
    
    # train algorithm
    args = Parameters({
                        'batch_size': 10, 
                        'trials': 1, 
                        'verbose': True,
                        'lr': 1e-1,
                        'num_samples': 10000,
                        'early_stopping': True, 
                        'tol': 5e-2,
                        'gradient_steps': 1500,
                    }) 
    truncated = distributions.TruncatedNormal(args,
                                              phi_std_norm, 
                                              alpha, 
                                              1,
                                              variance=ch.eye(1))
    truncated.fit(S_std_norm)
    
    # rescale distribution
    rescale_loc = truncated.best_loc_ + emp_loc
    print(f"pred loc:\n {rescale_loc}")
    rescale_var = truncated.best_variance_
    print(f"pred var:\n {rescale_var}")
    m = MultivariateNormal(rescale_loc, rescale_var)
        
    # check performance
    kl_truncated = kl_divergence(m, M)
    kl_emp = kl_divergence(MultivariateNormal(emp_loc, M.covariance_matrix), M)
    print(f'empirical kl divergence: {kl_emp.item():.3f}')
    print(f'truncated kl divergence: {kl_truncated.item():.3f}')
    msg = f'kl divergence between estimated and true underlying distribution is greater than 1e-1. truncated kl divergence is {kl_truncated}'
    assert kl_truncated <= 1e-1, msg

# right truncated normal distribution with known truncation
def test_truncated_normal():
    M = MultivariateNormal(ch.zeros(1), ch.eye(1)) 
    samples = M.rsample([5000,])
    print(f'num total samples: {samples.size(0)}')
    # generate ground-truth data
    phi = oracle.Left_Distribution(Tensor([0.0]))
    # truncate
    indices = phi(samples).nonzero()[:,0]
    S = samples[indices]
    print(f'num truncated samples: {S.size(0)}')
    alpha = S.size(0) / samples.size(0)
    emp_loc = S.mean(0)
    print(f"emp loc:\n {emp_loc}")
    emp_var = S.var(0)[...,None]
    print(f"emp var:\n {emp_var}")
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
                        'num_samples': 10000,
                        'early_stopping': True,
                        'tol': 1e-3,
                        'var_lr': 1e-1,
                    }) 
    truncated = distributions.TruncatedNormal(args,
                                              phi_std_norm, 
                                              alpha, 
                                              1)
    truncated.fit(S_std_norm)

    # rescale distribution
    rescale_loc = truncated.best_loc_ * emp_scale + emp_loc
    print(f"pred loc:\n {rescale_loc}")
    rescale_var = truncated.best_variance_ * emp_var
    print(f"pred var:\n {rescale_var}")
    m = MultivariateNormal(rescale_loc, rescale_var)
        
    # check performance
    kl_truncated = kl_divergence(m, M)
    kl_emp = kl_divergence(MultivariateNormal(emp_loc, M.covariance_matrix), M)
    print(f'empirical kl divergence: {kl_emp.item():.3f}')
    print(f'truncated kl divergence: {kl_truncated.item():.3f}')
    msg = f'kl divergence between estimated and true underlying distribution is greater than 1e-1. truncated kl divergence is {kl_truncated}'
    assert kl_truncated <= 1e-1, msg

def test_truncated_2_dim_multivariate_normal_known_covariance_matrix():
    dims = 2
    M = MultivariateNormal(ch.zeros(dims), 2 * ch.eye(dims)) 
    samples = M.rsample([5000,])

    phi = lambda x: x[:, 0] > 0
    # generate ground-truth data
    alpha = 0
    while alpha < .3: 
        indices = phi(samples).nonzero()[:,0]
        S = samples[indices]
        alpha = S.size(0) / samples.size(0)

    print(f'alpha: {alpha}')
    print(f'num total samples: {samples.size(0)}')
    print(f'num truncated samples: {S.size(0)}')

    emp_loc = S.mean(0, keepdim=True)
    print(f'empirical mean:\n {emp_loc.T}')

    # train algorithm
    args = Parameters({
                        'epochs': 10, 
                        'trials': 1,
                        'batch_size': 50,
                        'num_samples': 10000, 
                        'verbose': True, 
                        'lr': 1e-1,
                    }) 
    truncated = distributions.TruncatedMultivariateNormal(args, 
                                                          phi, 
                                                          alpha, 
                                                          dims,
                                                          covariance_matrix=M.covariance_matrix)
    truncated.fit(S)
    # rescale distribution
    rescale_loc = truncated.best_loc_
    print(f'pred mean:\n {rescale_loc.T}')
    print(f'pred covariance matrix:\n {M.covariance_matrix}')
    m = MultivariateNormal(rescale_loc, truncated.best_covariance_matrix_)
    # check performance
    kl_truncated = kl_divergence(m, M)
    kl_emp = kl_divergence(MultivariateNormal(emp_loc, M.covariance_matrix), M)
    print(f'empirical kl divergence: {kl_emp.item():.3f}')
    print(f'truncated kl divergence: {kl_truncated.item():.3f}')
    msg = f'kl divergence between estimated and true underlying distribution is greater than 1e-1. truncated kl divergence is {kl_truncated}'
    assert kl_truncated <= 1e-1, msg

def test_truncated_2_dim_multivariate_normal():
    dims = 2
    M = MultivariateNormal(ch.zeros(dims), 2 * ch.eye(dims)) 
    samples = M.rsample([5000,])

    phi = lambda x: x[:, 0] > 0
    # generate ground-truth data
    alpha = 0
    while alpha < .3: 
        indices = phi(samples).nonzero()[:,0]
        S = samples[indices]
        alpha = S.size(0) / samples.size(0)

    print(f'alpha: {alpha}')
    print(f'num total samples: {samples.size(0)}')
    print(f'num truncated samples: {S.size(0)}')

    emp_loc = S.mean(0, keepdim=True)
    emp_covariance_matrix = cov(S)

    print(f'empicial mean:\n {emp_loc.T}')
    print(f'empirical covariance matrix:\n {emp_covariance_matrix}')

    # train algorithm
    args = Parameters({
                        'epochs': 10, 
                        'trials': 1,
                        'batch_size': 50,
                        'num_samples': 10000, 
                        'verbose': True, 
                        'lr': 1e-1,
                    }) 
    truncated = distributions.TruncatedMultivariateNormal(args, 
                                                          phi, 
                                                          alpha, 
                                                          dims)
    truncated.fit(S)
    # rescale distribution
    rescale_loc = truncated.best_loc_
    rescale_covariance_matrix = truncated.best_covariance_matrix_
    print(f'pred loc:\n {rescale_loc.T}')
    print(f'pred covariance matrix:\n {rescale_covariance_matrix}')
    m = MultivariateNormal(rescale_loc, rescale_covariance_matrix)
        
    # check performance
    kl_truncated = kl_divergence(m, M)
    kl_emp = kl_divergence(MultivariateNormal(emp_loc, M.covariance_matrix), M)
    print(f'empirical kl divergence: {kl_emp.item():.3f}')
    print(f'truncated kl divergence: {kl_truncated.item():.3f}')
    msg = f'kl divergence between estimated and true underlying distribution is greater than 1e-1. truncated kl divergence is {kl_truncated}'
    assert kl_truncated <= 1e-1, msg

def test_truncated_10_dim_multivariate_normal_known_covariance_matrix():
    dims = 10
    M = MultivariateNormal(ch.zeros(dims), 10 * ch.eye(dims)) 
    samples = M.rsample([10000,])

    phi = lambda x: x[:, 0] > 0
    # generate ground-truth data
    alpha = 0
    while alpha < .3: 
        indices = phi(samples).nonzero()[:,0]
        S = samples[indices]
        alpha = S.size(0) / samples.size(0)

    print(f'alpha: {alpha}')
    print(f'num total samples: {samples.size(0)}')
    print(f'num truncated samples: {S.size(0)}')

    emp_loc = S.mean(0, keepdim=True)
    print(f'empirical mean:\n {emp_loc.T}')

    # train algorithm
    args = Parameters({
                        'epochs': 10, 
                        'trials': 1,
                        'batch_size': 10,
                        'num_samples': 10000, 
                        'verbose': True, 
                        'lr': 1e-1,
                    }) 
    truncated = distributions.TruncatedMultivariateNormal(args, 
                                                          phi, 
                                                          alpha, 
                                                          dims,
                                                          covariance_matrix=M.covariance_matrix)
    truncated.fit(S)
    # rescale distribution
    rescale_loc = truncated.best_loc_
    print(f'pred loc:\n {rescale_loc.T}')
    m = MultivariateNormal(rescale_loc, truncated.best_covariance_matrix_)
        
    # check performance
    kl_truncated = kl_divergence(m, M)
    kl_emp = kl_divergence(MultivariateNormal(emp_loc, M.covariance_matrix), M)
    print(f'empirical kl divergence: {kl_emp.item():.3f}')
    print(f'truncated kl divergence: {kl_truncated.item():.3f}')
    msg = f'kl divergence between estimated and true underlying distribution is greater than 1e-1. truncated kl divergence is {kl_truncated}'
    assert kl_truncated <= 1e-1, msg

def test_truncated_10_dim_multivariate_normal():
    dims = 10
    M = MultivariateNormal(ch.zeros(dims), 10 * ch.eye(dims)) 
    samples = M.rsample([20000,])

    phi = lambda x: x[:, 0] > 0
    # generate ground-truth data
    alpha = 0
    while alpha < .3: 
        indices = phi(samples).nonzero()[:,0]
        S = samples[indices]
        alpha = S.size(0) / samples.size(0)

    print(f'alpha: {alpha}')
    print(f'num total samples: {samples.size(0)}')
    print(f'num truncated samples: {S.size(0)}')

    emp_loc = S.mean(0, keepdim=True)
    print(f'empirical mean:\n {emp_loc.T}')

    # train algorithm
    args = Parameters({
                        'epochs': 20, 
                        'trials': 1,
                        'batch_size': 100,
                        'num_samples': 10000, 
                        'verbose': True, 
                        'lr': 1e-1,
                        'covariance_matrix_lr': 1e-2,
                    }) 
    truncated = distributions.TruncatedMultivariateNormal(args, 
                                                          phi, 
                                                          alpha, 
                                                          dims) 
    truncated.fit(S)
    # rescale distribution
    rescale_loc = truncated.best_loc_ 
    rescale_cov = truncated.best_covariance_matrix_
    print(f'pred loc:\n {rescale_loc.T}')
    print(f'pred covariance matrix: {rescale_cov}')
    m = MultivariateNormal(rescale_loc, rescale_cov)
        
    # check performance
    kl_truncated = kl_divergence(m, M)
    kl_emp = kl_divergence(MultivariateNormal(emp_loc, M.covariance_matrix), M)
    print(f'empirical kl divergence: {kl_emp.item():.3f}')
    print(f'truncated kl divergence: {kl_truncated.item():.3f}')
    msg = f'kl divergence between estimated and true underlying distribution is greater than 1e-1. truncated kl divergence is {kl_truncated}'
    assert kl_truncated <= 1e-1, msg

# right truncated 1D normal distribution with unknown truncation and known variance
def test_unknown_truncation_normal_known_variance():
    dims = 1
    # generate ground-truth data
    M = MultivariateNormal(ch.zeros(dims), ch.eye(dims)) 
    samples = M.rsample([10000,])
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
    print(f'emp var: {emp_var}')
    print(f'known variance: {M.covariance_matrix}')
    k = 5
    print(f'k: {k}')

    # train algorithm
    args = Parameters({
                        'epochs': 2, 
                        'trials': 1,
                        'batch_size': 100,
                        'lr': 1e-1, 
                        'early_stopping': True,
                        'verbose': True,
                    }) 
    truncated = distributions.UnknownTruncationNormal(args, k, alpha, dims, covariance_matrix=M.covariance_matrix)
    truncated.fit(S)
    # rescale distribution
    rescale_loc = truncated.best_loc_
    print(f'pred loc: {rescale_loc}')
    print(f'pred variance: {truncated.best_covariance_matrix_}')
    m = MultivariateNormal(rescale_loc, truncated.best_covariance_matrix_)
                
    # check distribution parameter estimates
    kl_truncated = kl_divergence(m, M)
    kl_emp = kl_divergence(MultivariateNormal(emp_loc, M.covariance_matrix), M)
    print(f'empirical kl divergence: {kl_emp.item():.3f}')
    print(f'truncated kl divergence: {kl_truncated.item():.3f}')
    msg = f'kl divergence between estimated and true underlying distribution is greater than 1e-1. truncated kl divergence is {kl_truncated}'
    assert kl_truncated <= 1e-1, msg

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