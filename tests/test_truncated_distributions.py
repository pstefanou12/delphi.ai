# distribution tests 

import torch as ch
from torch import Tensor
from torch.distributions import MultivariateNormal, Uniform, Bernoulli, Exponential, Poisson, Weibull
from torch.distributions.kl import kl_divergence

from delphi import distributions 
from delphi import oracle
from delphi.utils.helpers import Parameters, cov 

ch.set_printoptions(precision=4, sci_mode=False)

def generate_truncated_dataset(dist, phi, num_samples): 
    num_accepted, num_sampled = 0, 0

    S = []
    while num_accepted < num_samples: 
        samples = dist.sample([num_samples,])
        indices = phi(samples).nonzero()[:,0]
        S.append(samples[indices])
        num_accepted +=  indices.size(0)
        num_sampled += num_samples
    S = ch.cat(S)[:num_samples]
    alpha = num_accepted / num_sampled

    return S, alpha


# right truncated normal distribution with known truncation and known variance
def test_truncated_normal_known_variance():
    M = MultivariateNormal(ch.zeros(1), ch.eye(1)) 
    phi = oracle.Left_Distribution(Tensor([0.0]))
    num_samples = 1000
    S, alpha = generate_truncated_dataset(M, phi, num_samples)
    print(f'alpha: {alpha}')
    print(f'num_samples: {num_samples}')

    emp_loc = S.mean(0)
    print(f"emp loc:\n {emp_loc}")
    print(f"known variance:\n {ch.eye(1)}")
    S_std_norm = (S - emp_loc) 
    phi_std_norm = oracle.Left_Distribution((phi.left - emp_loc).flatten())

    args = Parameters({
                        'batch_size': 10, 
                        'trials': 1, 
                        'verbose': True,
                        'lr': 1e-1,
                        'num_samples': 1000,
                        'early_stopping': True, 
                        'tol': 5e-2,
                        'iterations': 1500,
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
    m = MultivariateNormal(rescale_loc, ch.eye(1))
        
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

    phi = oracle.Left_Distribution(Tensor([0.0]))
    num_samples = 1000
    S, alpha = generate_truncated_dataset(M, phi, num_samples)
    print(f'alpha: {alpha}')
    print(f'num_samples: {num_samples}')

    emp_loc = S.mean(0)
    print(f"emp loc:\n {emp_loc}")
    emp_var = S.var(0)[...,None]
    print(f"emp var:\n {emp_var}")
    emp_scale = ch.sqrt(emp_var) 

    S_std_norm = (S - emp_loc) / emp_scale
    phi_std_norm = oracle.Left_Distribution(((phi.left - emp_loc) / emp_scale).flatten())

    args = Parameters({
                        'iterations': 1500, 
                        'batch_size': 10, 
                        'trials': 1, 
                        'verbose': True,
                        'lr': 1e-2,
                        'num_samples': 1000,
                        'early_stopping': True,
                        'tol': 1e-3,
                        'val_interval': 100
                    }) 
    truncated = distributions.TruncatedNormal(args,
                                              phi_std_norm, 
                                              alpha, 
                                              1)
    truncated.fit(S_std_norm)
    # rescale distribution
    rescale_best_loc = truncated.best_loc_ * emp_scale + emp_loc
    print(f"pred best loc:\n {rescale_best_loc}")
    rescale_best_var = truncated.best_variance_ * emp_var
    print(f"pred best var:\n {rescale_best_var}")
    rescale_ema_loc = truncated.ema_loc_ * emp_scale + emp_loc
    print(f"pred ema loc:\n {rescale_ema_loc}")
    rescale_ema_var = truncated.ema_covariance_matrix_ * emp_var
    print(f"pred ema var:\n {rescale_ema_var}")
    rescale_avg_loc = truncated.avg_loc_ * emp_scale + emp_loc
    print(f"pred avg loc:\n {rescale_avg_loc}")
    rescale_avg_var = truncated.avg_variance_ * emp_var
    print(f"pred avg var:\n {rescale_avg_var}")

    m = MultivariateNormal(rescale_best_loc, rescale_best_var)
        
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

    num_samples = 5000
    phi = lambda x: x[:, 0] > 0
    # generate ground-truth data
    S, alpha = generate_truncated_dataset(M, phi, num_samples)

    print(f'alpha: {alpha}')
    print(f'num_samples: {num_samples}')

    emp_loc = S.mean(0, keepdim=True)
    print(f'empirical mean:\n {emp_loc.T}')

    # train algorithm
    args = Parameters({
                        'iterations': 2500, 
                        'trials': 1,
                        'batch_size': 1,
                        'num_samples': 10000, 
                        'verbose': True, 
                        'lr': 1e-2,
                    }) 
    truncated = distributions.TruncatedMultivariateNormal(args, 
                                                          phi, 
                                                          alpha, 
                                                          dims,
                                                          covariance_matrix=M.covariance_matrix)

    truncated.fit(S)

    best_loc = truncated.best_loc_ 
    print(f'best loc:\n {best_loc.T}')
    best_m = MultivariateNormal(best_loc, M.covariance_matrix)
    kl_truncated = kl_divergence(best_m, M)

    ema_loc = truncated.ema_loc_ 
    print(f'ema loc:\n {ema_loc.T}')
    ema_m = MultivariateNormal(ema_loc, M.covariance_matrix)
    ema_kl_div = kl_divergence(ema_m, M)
    print(f'ema kl divergence: {ema_kl_div}')

    avg_loc = truncated.avg_loc_ 
    print(f'avg loc:\n {avg_loc.T}')
    avg_m = MultivariateNormal(avg_loc, M.covariance_matrix)
    avg_kl_div = kl_divergence(avg_m, M)
    print(f'avg kl divergence: {avg_kl_div}')

    kl_emp = kl_divergence(MultivariateNormal(emp_loc, M.covariance_matrix), M)
    print(f'empirical kl divergence: {kl_emp.item():.3f}')
    print(f'truncated kl divergence: {kl_truncated.item():.3f}')
    msg = f'kl divergence between estimated and true underlying distribution is greater than 1e-1. truncated kl divergence is {kl_truncated}'
    assert kl_truncated <= 1e-1, msg

def test_truncated_2_dim_multivariate_normal():
    dims = 2
    M = MultivariateNormal(ch.zeros(dims), 2 * ch.eye(dims)) 
    num_samples = 1000
    phi = lambda x: x[:, 0] > 0
    S, alpha = generate_truncated_dataset(M, phi, num_samples)
    
    print(f'alpha: {alpha}')
    print(f'num truncated samples: {S.size(0)}')

    emp_loc = S.mean(0, keepdim=True)
    emp_covariance_matrix = cov(S)
    emp_var = S.var(0)
    emp_sigma_diag = ch.diag(ch.sqrt(emp_var))
    S_std_norm = (S - emp_loc) / ch.sqrt(emp_var)

    print(f'empicial mean:\n {emp_loc.T}')
    print(f'empirical covariance matrix:\n {emp_covariance_matrix}')

    phi_std_norm = lambda x: x[:,0] > (0 - emp_loc[0,0]) / ch.sqrt(emp_var[0])
    S_std_norm = (S - emp_loc) / ch.sqrt(emp_var)

    # train algorithm
    args = Parameters({
                        'iterations': 5000, 
                        'batch_size': 10,
                        'num_samples': 1000, 
                        'verbose': True, 
                        'lr': 1e-1,
                        'optimizer': 'sgd',
                        'covariance_matrix_lr': 1e-2,
                    }) 
    truncated = distributions.TruncatedMultivariateNormal(args, 
                                                          phi_std_norm, 
                                                          alpha, 
                                                          dims)
    truncated.fit(S_std_norm)
    best_loc = truncated.best_loc_ * ch.sqrt(emp_var) + emp_loc
    best_covariance_matrix = emp_sigma_diag @ truncated.best_covariance_matrix_ @ emp_sigma_diag 
    print(f'best loc:\n {best_loc.T}')
    print(f'best covariance matrix:\n {best_covariance_matrix}')
    best_m = MultivariateNormal(best_loc, best_covariance_matrix)

    ema_loc = truncated.ema_loc_ * ch.sqrt(emp_var) + emp_loc
    ema_covariance_matrix = emp_sigma_diag @ truncated.ema_covariance_matrix_ @ emp_sigma_diag 
    print(f'ema loc:\n {ema_loc.T}')
    print(f'ema covariance matrix:\n {ema_covariance_matrix}')
    ema_m = MultivariateNormal(ema_loc, ema_covariance_matrix)
    ema_kl_div = kl_divergence(ema_m, M)
    print(f'ema kl divergence: {ema_kl_div}')

    avg_loc = truncated.avg_loc_ * ch.sqrt(emp_var) + emp_loc
    avg_covariance_matrix = emp_sigma_diag @ truncated.avg_covariance_matrix_ @ emp_sigma_diag 
    print(f'avg loc:\n {avg_loc.T}')
    print(f'avg covariance matrix:\n {avg_covariance_matrix}')
    avg_m = MultivariateNormal(avg_loc, avg_covariance_matrix)
    avg_kl_div = kl_divergence(avg_m, M)
    print(f'avg kl divergence: {avg_kl_div}')
        
    # check performance
    kl_truncated = kl_divergence(best_m, M)
    kl_emp = kl_divergence(MultivariateNormal(emp_loc, emp_covariance_matrix), M)
    print(f'empirical kl divergence: {kl_emp.item():.3f}')
    print(f'truncated kl divergence: {kl_truncated.item():.3f}')
    msg = f'kl divergence between estimated and true underlying distribution is greater than 1e-1. truncated kl divergence is {kl_truncated}'
    assert kl_truncated <= 1e-1, msg

def test_truncated_10_dim_multivariate_normal_known_covariance_matrix():
    dims = 10
    M = MultivariateNormal(ch.zeros(dims), 10 * ch.eye(dims)) 

    phi = lambda x: x[:, 0] > 0
    num_samples = 1000
    S, alpha = generate_truncated_dataset(M, phi, num_samples)
    print(f'alpha: {alpha}')
    print(f'num truncated samples: {S.size(0)}')

    emp_loc = S.mean(0, keepdim=True)
    emp_covariance_matrix = cov(S)
    emp_var = M.covariance_matrix.diag()
    S_std_norm = (S - emp_loc) / ch.sqrt(emp_var)

    print(f'empicial mean:\n {emp_loc.T}')
    print(f'empirical covariance matrix:\n {emp_covariance_matrix}')

    phi_std_norm = lambda x: x[:,0] > ((0 - emp_loc[0,0]) / ch.sqrt(emp_var[0]))
    S_std_norm = (S - emp_loc) / ch.sqrt(emp_var)

    # train algorithm
    args = Parameters({
                        'epochs': 2, 
                        'batch_size': 1,
                        'num_samples': 1000, 
                        'verbose': True, 
                        'optimizer': 'sgd',
                        'lr': 1e-2,
                    }) 
    
    scaled_cov = ch.diag(ch.sqrt(emp_var)).inverse() * M.covariance_matrix * ch.diag(ch.sqrt(emp_var)).inverse()
    truncated = distributions.TruncatedMultivariateNormal(args, 
                                                          phi_std_norm, 
                                                          alpha, 
                                                          dims,
                                                          covariance_matrix=scaled_cov)
    truncated.fit(S_std_norm)
    best_loc = truncated.best_loc_ * ch.sqrt(emp_var) + emp_loc
    print(f'best loc:\n {best_loc.T}')
    best_m = MultivariateNormal(best_loc, M.covariance_matrix)

    ema_loc = truncated.ema_loc_ * ch.sqrt(emp_var) + emp_loc
    print(f'ema loc:\n {ema_loc.T}')
    ema_m = MultivariateNormal(ema_loc, M.covariance_matrix)
    ema_kl_div = kl_divergence(ema_m, M)
    print(f'ema kl divergence: {ema_kl_div}')

    avg_loc = truncated.avg_loc_ * ch.sqrt(emp_var) + emp_loc
    print(f'avg loc:\n {avg_loc.T}')
    avg_m = MultivariateNormal(avg_loc, M.covariance_matrix)
    avg_kl_div = kl_divergence(avg_m, M)
    print(f'avg kl divergence: {avg_kl_div}')
        
    # check performance
    kl_truncated = kl_divergence(best_m, M)
    kl_emp = kl_divergence(MultivariateNormal(emp_loc, M.covariance_matrix), M)
    print(f'empirical kl divergence: {kl_emp.item():.3f}')
    print(f'truncated kl divergence: {kl_truncated.item():.3f}')
    msg = f'kl divergence between estimated and true underlying distribution is greater than 1e-1. truncated kl divergence is {kl_truncated}'
    assert kl_truncated <= 1e-1, msg

def test_truncated_10_dim_multivariate_normal():
    dims = 10
    M = MultivariateNormal(ch.zeros(dims), 10 * ch.eye(dims)) 

    phi = lambda x: x[:, 0] > 0
    num_samples = 5000
    S, alpha = generate_truncated_dataset(M, phi, num_samples)
    print(f'alpha: {alpha}')
    print(f'num truncated samples: {S.size(0)}')

    emp_loc = S.mean(0, keepdim=True)
    emp_covariance_matrix = cov(S)
    emp_var = S.var(0)
    emp_sigma_diag = ch.diag(ch.sqrt(emp_var))
    S_std_norm = (S - emp_loc) / ch.sqrt(emp_var)

    print(f'empicial mean:\n {emp_loc.T}')
    print(f'empirical covariance matrix:\n {emp_covariance_matrix}')

    phi_std_norm = lambda x: x[:,0] > ((0 - emp_loc[0,0]) / ch.sqrt(emp_var[0]))
    S_std_norm = (S - emp_loc)  / ch.sqrt(emp_var)

    # train algorithm
    args = Parameters({
                        'iterations': 5000, 
                        'trials': 1,
                        'batch_size': 1,
                        'num_samples': 1000, 
                        'verbose': True, 
                        'optimizer': 'sgd',
                        'lr': 1e-1,
                        'covariance_matrix_lr': 1e-2,
                    }) 
    truncated = distributions.TruncatedMultivariateNormal(args, 
                                                          phi_std_norm, 
                                                          alpha, 
                                                          dims) 
    truncated.fit(S_std_norm)
    # rescale distribution
    best_loc = truncated.best_loc_ * ch.sqrt(emp_var) + emp_loc
    best_covariance_matrix = emp_sigma_diag @ truncated.best_covariance_matrix_ @ emp_sigma_diag 
    print(f'best loc:\n {best_loc.T}')
    print(f'best covariance matrix:\n {best_covariance_matrix}')
    best_m = MultivariateNormal(best_loc, best_covariance_matrix)

    ema_loc = truncated.ema_loc_ * ch.sqrt(emp_var) + emp_loc
    ema_covariance_matrix = emp_sigma_diag @ truncated.ema_covariance_matrix_ @ emp_sigma_diag 
    print(f'ema loc:\n {ema_loc.T}')
    print(f'ema covariance matrix:\n {ema_covariance_matrix}')
    ema_m = MultivariateNormal(ema_loc, ema_covariance_matrix)
    ema_kl_div = kl_divergence(ema_m, M)
    print(f'ema kl divergence: {ema_kl_div}')

    avg_loc = truncated.avg_loc_ * ch.sqrt(emp_var) + emp_loc
    avg_covariance_matrix = emp_sigma_diag @ truncated.avg_covariance_matrix_ @ emp_sigma_diag 
    print(f'avg loc:\n {avg_loc.T}')
    print(f'avg covariance matrix:\n {avg_covariance_matrix}')
    avg_m = MultivariateNormal(avg_loc, avg_covariance_matrix)
    avg_kl_div = kl_divergence(avg_m, M)
    print(f'avg kl divergence: {avg_kl_div}')
        
    # check performance
    kl_truncated = kl_divergence(best_m, M)
    kl_emp = kl_divergence(MultivariateNormal(emp_loc, emp_covariance_matrix), M)
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

def test_truncated_boolean_product_2_dims():
    dims = 2
    p = ch.Tensor([.5, .75])
    print(f'true p: {p}')
    dist = Bernoulli(p)

    def phi(z):
        return ~((z[:,0] == 1) * (z[:,1] == 1))

    num_samples = 10000
    S, alpha = generate_truncated_dataset(dist, phi, num_samples)

    emp_p = S.mean(0)
    print(f'alpha: {alpha}')
    print(f'num truncated samples: {S.size(0)}')
    print(f'emp_p: {emp_p}') 

    args = Parameters({
                    'iterations': 2500, 
                    'trials': 1,
                    'batch_size': 1,
                    'num_samples': 1000, 
                    'verbose': True, 
                    'optimizer': 'sgd',
                    'lr': 1e-1,
                    'max_phases': 1000000,
                    'rate': 1.5,
                }) 
    
    truncated = distributions.TruncatedBooleanProduct(args,
                                           phi, 
                                           alpha, 
                                           dims)
    truncated.fit(S)

    best_p = truncated.best_p_ 
    print(f'best p:\n {best_p.T}')
    best_m = Bernoulli(best_p)

    ema_p = truncated.ema_p_ 
    print(f'ema p:\n {ema_p.T}')
    ema_m = Bernoulli(ema_p)
    ema_kl_div = kl_divergence(ema_m, dist).sum()
    print(f'ema kl divergence: {ema_kl_div}')

    avg_p = truncated.avg_p_
    print(f'avg p:\n {avg_p.T}')
    avg_m = Bernoulli(avg_p)
    avg_kl_div = kl_divergence(avg_m, dist).sum()
    print(f'avg kl divergence: {avg_kl_div}')
        
    # check performance
    kl_truncated = kl_divergence(best_m, dist).sum()
    kl_emp = kl_divergence(Bernoulli(emp_p), dist).sum()
    print(f'empirical kl divergence: {kl_emp.item():.3f}')
    print(f'truncated kl divergence: {kl_truncated.item():.3f}')
    msg = f'kl divergence between estimated and true underlying distribution is greater than 1e-1. truncated kl divergence is {kl_truncated}'
    assert kl_truncated <= 1e-1, msg

def test_truncated_boolean_product_20_dims():
    dims = 20
    p = ch.Tensor([.5, .75, .3, .4, .8, .5, .75, .3, .4, .8, .5, .75, .3, .4, .8, .5, .75, .3, .4, .8])
    print(f'true p: {p}')
    dist = Bernoulli(p)

    def phi(z):
        return ~((z[:,0] == 1) * (z[:,1] == 1))

    num_samples = 10000

    S, alpha = generate_truncated_dataset(dist, phi, num_samples)
    emp_p = S.mean(0)
    print(f'alpha: {alpha}')
    print(f'num truncated samples: {S.size(0)}')
    print(f'emp_p: {emp_p}') 

    args = Parameters({
                    'iterations': 2500, 
                    'trials': 1,
                    'batch_size': 1,
                    'num_samples': 1000, 
                    'verbose': True, 
                    'optimizer': 'sgd',
                    'lr': 1e-1,
                }) 
    
    truncated = distributions.TruncatedBooleanProduct(args,
                                           phi, 
                                           alpha, 
                                           dims)
    truncated.fit(S)

    best_p = truncated.best_p_ 
    print(f'best p:\n {best_p.T}')
    best_m = Bernoulli(best_p)

    ema_p = truncated.ema_p_ 
    print(f'ema p:\n {ema_p.T}')
    ema_m = Bernoulli(ema_p)
    ema_kl_div = kl_divergence(ema_m, dist).sum()
    print(f'ema kl divergence: {ema_kl_div}')

    avg_p = truncated.avg_p_
    print(f'avg p:\n {avg_p.T}')
    avg_m = Bernoulli(avg_p)
    avg_kl_div = kl_divergence(avg_m, dist).sum()
    print(f'avg kl divergence: {avg_kl_div}')
        
    # check performance
    kl_truncated = kl_divergence(best_m, dist).sum()
    kl_emp = kl_divergence(Bernoulli(emp_p), dist).sum()
    print(f'empirical kl divergence: {kl_emp.item():.3f}')
    print(f'truncated kl divergence: {kl_truncated.item():.3f}')
    msg = f'kl divergence between estimated and true underlying distribution is greater than 1e-1. truncated kl divergence is {kl_truncated}'
    assert kl_truncated <= 1e-1, msg

def test_truncated_exponential():
    dims = 1
    lambda_ = ch.Tensor([1])
    print(f'true lambda: {lambda_}')
    dist = Exponential(lambda_)

    def phi(z):
        return z > 1

    num_samples = 10000
    S, alpha = generate_truncated_dataset(dist, phi, num_samples)

    emp_lambda = 1.0/S.mean(0)
    print(f'alpha: {alpha}')
    print(f'num truncated samples: {S.size(0)}')
    print(f'emp_lambda: {emp_lambda}') 

    args = Parameters({
                    'iterations': 2500, 
                    'trials': 1,
                    'batch_size': 1,
                    'num_samples': 1000, 
                    'verbose': True, 
                    'optimizer': 'sgd',
                    'lr': 1e-2,
                    'max_phases': 1000000,
                    'rate': 1.5,
                }) 
    
    truncated = distributions.TruncatedExponential(args,
                                           phi, 
                                           alpha, 
                                           dims)
    truncated.fit(S)
    
    best_lambda = truncated.best_lambda_ 
    print(f'best lambda:\n {best_lambda.T}')
    best_m = Exponential(best_lambda)

    ema_lambda = truncated.ema_lambda_ 
    print(f'ema lambda:\n {ema_lambda.T}')
    ema_m = Exponential(ema_lambda)
    ema_kl_div = kl_divergence(ema_m, dist).sum()
    print(f'ema kl divergence: {ema_kl_div}')

    avg_lambda = truncated.avg_lambda_
    print(f'avg lambda:\n {avg_lambda.T}')
    avg_m = Exponential(avg_lambda)
    avg_kl_div = kl_divergence(avg_m, dist).sum()
    print(f'avg kl divergence: {avg_kl_div}')
        
    # check performance
    kl_truncated = kl_divergence(best_m, dist).sum()
    kl_emp = kl_divergence(Exponential(emp_lambda), dist).sum()
    print(f'empirical kl divergence: {kl_emp.item():.3f}')
    print(f'truncated kl divergence: {kl_truncated.item():.3f}')
    msg = f'kl divergence between estimated and true underlying distribution is greater than 1e-1. truncated kl divergence is {kl_truncated}'
    assert kl_truncated <= 1e-1, msg

def test_truncated_exponential_2_dims():
    dims = 2
    lambda_ = ch.Tensor([1, 2.0])
    print(f'true lambda: {lambda_}')
    dist = Exponential(lambda_)

    def phi(z):
        return z[:,0] > .5

    num_samples = 10000
    S, alpha = generate_truncated_dataset(dist, phi, num_samples)

    emp_lambda = 1.0/S.mean(0)
    print(f'alpha: {alpha}')
    print(f'num truncated samples: {S.size(0)}')
    print(f'emp_lambda: {emp_lambda}') 

    args = Parameters({
                    'iterations': 2500, 
                    'trials': 1,
                    'batch_size': 10,
                    'num_samples': 1000, 
                    'verbose': True, 
                    'optimizer': 'sgd',
                    'lr': 1e-2,
                }) 
    
    truncated = distributions.TruncatedExponential(args,
                                           phi, 
                                           alpha, 
                                           dims)
    truncated.fit(S)
    
    best_lambda = truncated.best_lambda_ 
    print(f'best lambda:\n {best_lambda.T}')
    best_m = Exponential(best_lambda)

    ema_lambda = truncated.ema_lambda_ 
    print(f'ema lambda:\n {ema_lambda.T}')
    ema_m = Exponential(ema_lambda)
    ema_kl_div = kl_divergence(ema_m, dist).sum()
    print(f'ema kl divergence: {ema_kl_div}')

    avg_lambda = truncated.avg_lambda_
    print(f'avg lambda:\n {avg_lambda.T}')
    avg_m = Exponential(avg_lambda)
    avg_kl_div = kl_divergence(avg_m, dist).sum()
    print(f'avg kl divergence: {avg_kl_div}')
        
    # check performance
    kl_truncated = kl_divergence(best_m, dist).sum()
    kl_emp = kl_divergence(Exponential(emp_lambda), dist).sum()
    print(f'empirical kl divergence: {kl_emp.item():.3f}')
    print(f'truncated kl divergence: {kl_truncated.item():.3f}')
    msg = f'kl divergence between estimated and true underlying distribution is greater than 1e-1. truncated kl divergence is {kl_truncated}'
    assert kl_truncated <= 1e-1, msg

def test_truncated_exponential_20_dims():
    dims = 20
    lambda_ = 2*ch.ones(20,)
    print(f'true lambda: {lambda_}')
    dist = Exponential(lambda_)

    def phi(z):
        return z[:,0] > .25

    num_samples = 10000
    S, alpha = generate_truncated_dataset(dist, phi, num_samples)

    emp_lambda = 1.0/S.mean(0)
    print(f'alpha: {alpha}')
    print(f'num truncated samples: {S.size(0)}')
    print(f'emp_lambda: {emp_lambda}') 

    args = Parameters({
                    'iterations': 1500, 
                    'trials': 1,
                    'batch_size': 10,
                    'num_samples': 1000, 
                    'verbose': True, 
                    'optimizer': 'sgd',
                    'lr': 1e-1,
                }) 
    
    truncated = distributions.TruncatedExponential(args,
                                                   phi, 
                                                   alpha, 
                                                   dims)
    truncated.fit(S)
    
    best_lambda = truncated.best_lambda_ 
    print(f'best lambda:\n {best_lambda.T}')
    best_m = Exponential(best_lambda)

    ema_lambda = truncated.ema_lambda_ 
    print(f'ema lambda:\n {ema_lambda.T}')
    ema_m = Exponential(ema_lambda)
    ema_kl_div = kl_divergence(ema_m, dist).sum()
    print(f'ema kl divergence: {ema_kl_div}')

    avg_lambda = truncated.avg_lambda_
    print(f'avg lambda:\n {avg_lambda.T}')
    avg_m = Exponential(avg_lambda)
    avg_kl_div = kl_divergence(avg_m, dist).sum()
    print(f'avg kl divergence: {avg_kl_div}')
    
    # check performance
    kl_truncated = kl_divergence(best_m, dist).sum()
    kl_emp = kl_divergence(Exponential(emp_lambda), dist).sum()
    print(f'empirical kl divergence: {kl_emp.item():.3f}')
    print(f'truncated kl divergence: {kl_truncated.item():.3f}')
    msg = f'kl divergence between estimated and true underlying distribution is greater than 1e-1. truncated kl divergence is {kl_truncated}'
    assert kl_truncated <= 1e-1, msg

def test_truncated_poisson():
    dims = 1
    lambda_ = ch.Tensor([1])
    print(f'true lambda: {lambda_}')
    dist = Poisson(lambda_)

    def phi(z):
        return z > 1

    num_samples = 10000
    S, alpha = generate_truncated_dataset(dist, phi, num_samples)

    emp_lambda = S.mean(0)
    print(f'alpha: {alpha}')
    print(f'num truncated samples: {S.size(0)}')
    print(f'emp_lambda: {emp_lambda}') 

    args = Parameters({
                    'iterations': 2500, 
                    'trials': 1,
                    'batch_size': 1,
                    'num_samples': 1000, 
                    'verbose': True, 
                    'optimizer': 'sgd',
                    'lr': 1e-2,
                    'max_phases': 1000000,
                    'rate': 1.5,
                }) 
    
    truncated = distributions.TruncatedPoisson(args,
                                           phi, 
                                           alpha, 
                                           dims)
    truncated.fit(S)
    
    best_lambda = truncated.best_lambda_ 
    print(f'best lambda:\n {best_lambda.T}')
    best_m = Poisson(best_lambda)

    ema_lambda = truncated.ema_lambda_ 
    print(f'ema lambda:\n {ema_lambda.T}')
    ema_m = Poisson(ema_lambda)
    ema_kl_div = kl_divergence(ema_m, dist).sum()
    print(f'ema kl divergence: {ema_kl_div}')

    avg_lambda = truncated.avg_lambda_
    print(f'avg lambda:\n {avg_lambda.T}')
    avg_m = Poisson(avg_lambda)
    avg_kl_div = kl_divergence(avg_m, dist).sum()
    print(f'avg kl divergence: {avg_kl_div}')
        
    # check performance
    kl_truncated = kl_divergence(best_m, dist).sum()
    kl_emp = kl_divergence(Poisson(emp_lambda), dist).sum()
    print(f'empirical kl divergence: {kl_emp.item():.3f}')
    print(f'truncated kl divergence: {kl_truncated.item():.3f}')
    msg = f'kl divergence between estimated and true underlying distribution is greater than 1e-1. truncated kl divergence is {kl_truncated}'
    assert kl_truncated <= 1e-1, msg

def test_truncated_poisson_2_dims():
    dims = 2
    lambda_ = ch.Tensor([1.0, 2.0])
    print(f'true lambda: {lambda_}')
    dist = Poisson(lambda_)

    def phi(z):
        return z[:,0] > .25

    num_samples = 10000
    S, alpha = generate_truncated_dataset(dist, phi, num_samples)

    emp_lambda = S.mean(0)
    print(f'alpha: {alpha}')
    print(f'num truncated samples: {S.size(0)}')
    print(f'emp_lambda: {emp_lambda}') 

    args = Parameters({
                    'iterations': 2500, 
                    'trials': 1,
                    'batch_size': 1,
                    'num_samples': 1000, 
                    'verbose': True, 
                    'optimizer': 'sgd',
                    'lr': 1e-2,
                }) 
    
    truncated = distributions.TruncatedPoisson(args,
                                           phi, 
                                           alpha, 
                                           dims)
    truncated.fit(S)
    
    best_lambda = truncated.best_lambda_ 
    print(f'best lambda:\n {best_lambda.T}')
    best_m = Poisson(best_lambda)

    ema_lambda = truncated.ema_lambda_ 
    print(f'ema lambda:\n {ema_lambda.T}')
    ema_m = Poisson(ema_lambda)
    ema_kl_div = kl_divergence(ema_m, dist).sum()
    print(f'ema kl divergence: {ema_kl_div}')

    avg_lambda = truncated.avg_lambda_
    print(f'avg lambda:\n {avg_lambda.T}')
    avg_m = Poisson(avg_lambda)
    avg_kl_div = kl_divergence(avg_m, dist).sum()
    print(f'avg kl divergence: {avg_kl_div}')
        
    # check performance
    kl_truncated = kl_divergence(best_m, dist).sum()
    kl_emp = kl_divergence(Poisson(emp_lambda), dist).sum()
    print(f'empirical kl divergence: {kl_emp.item():.3f}')
    print(f'truncated kl divergence: {kl_truncated.item():.3f}')
    msg = f'kl divergence between estimated and true underlying distribution is greater than 1e-1. truncated kl divergence is {kl_truncated}'
    assert kl_truncated <= 1e-1, msg

def test_truncated_poisson_20_dims():
    dims = 20
    lambda_ = 2*ch.ones(20,)
    print(f'true lambda: {lambda_}')
    dist = Poisson(lambda_)

    def phi(z):
        return z[:,0] > 1

    num_samples = 10000
    S, alpha = generate_truncated_dataset(dist, phi, num_samples)

    emp_lambda = S.mean(0)
    print(f'alpha: {alpha}')
    print(f'num truncated samples: {S.size(0)}')
    print(f'emp_lambda: {emp_lambda}') 

    args = Parameters({
                    'iterations': 2500, 
                    'trials': 1,
                    'batch_size': 10,
                    'num_samples': 1000, 
                    'verbose': True, 
                    'optimizer': 'sgd',
                    'lr': 1e-2,
                }) 
    
    truncated = distributions.TruncatedPoisson(args,
                                           phi, 
                                           alpha, 
                                           dims)
    truncated.fit(S)
    
    best_lambda = truncated.best_lambda_ 
    print(f'best lambda:\n {best_lambda.T}')
    best_m = Poisson(best_lambda)

    ema_lambda = truncated.ema_lambda_ 
    print(f'ema lambda:\n {ema_lambda.T}')
    ema_m = Poisson(ema_lambda)
    ema_kl_div = kl_divergence(ema_m, dist).sum()
    print(f'ema kl divergence: {ema_kl_div}')

    avg_lambda = truncated.avg_lambda_
    print(f'avg lambda:\n {avg_lambda.T}')
    avg_m = Poisson(avg_lambda)
    avg_kl_div = kl_divergence(avg_m, dist).sum()
    print(f'avg kl divergence: {avg_kl_div}')
        
    # check performance
    kl_truncated = kl_divergence(best_m, dist).sum()
    kl_emp = kl_divergence(Poisson(emp_lambda), dist).sum()
    print(f'empirical kl divergence: {kl_emp.item():.3f}')
    print(f'truncated kl divergence: {kl_truncated.item():.3f}')
    msg = f'kl divergence between estimated and true underlying distribution is greater than 1e-1. truncated kl divergence is {kl_truncated}'
    assert kl_truncated <= 1e-1, msg

def test_truncated_weibull():
    dims = 1
    k = ch.Tensor([2.0])
    lambda_ = ch.Tensor([1])
    print(f'known k: {k}')
    print(f'true lambda: {lambda_}')

    dist = Weibull(lambda_, k)

    def phi(z):
        return z > 1.5

    num_samples = 10000
    S, alpha = generate_truncated_dataset(dist, phi, num_samples)

    emp_lambda_ = S.pow(k).mean(0).pow(1.0/k) 
    print(f'alpha: {alpha}')
    print(f'num truncated samples: {S.size(0)}')
    print(f'emp_lambda: {emp_lambda_}') 

    args = Parameters({
                    'iterations': 1500, 
                    'trials': 1,
                    'batch_size': 1,
                    'num_samples': 1000, 
                    'verbose': True, 
                    'optimizer': 'sgd',
                    'lr': 1e-2,
                    'max_phases': 1000000,
                    'rate': 1.5,
                }) 
    
    truncated = distributions.TruncatedWeibull(args,
                                           phi, 
                                           alpha, 
                                           dims, 
                                           k)
    truncated.fit(S)
    
    best_lambda = truncated.best_lambda_ 
    print(f'best lambda:\n {best_lambda.T}')
    best_l2_err = (best_lambda - lambda_).norm(p=2)
    print(f'truncated l2 error: {best_l2_err.item():.3f}')

    ema_lambda = truncated.ema_lambda_ 
    print(f'ema lambda:\n {ema_lambda.T}')
    ema_l2_err = (ema_lambda - lambda_).norm(p=2)
    print(f'ema l2 error: {ema_l2_err}')

    avg_lambda = truncated.avg_lambda_
    print(f'avg lambda:\n {avg_lambda.T}')
    avg_l2_err = (avg_lambda - lambda_).norm(p=2)
    print(f'avg l2 error: {avg_l2_err}')
        
    # check performance
    emp_l2_err = (emp_lambda_ - lambda_).norm(p=2)
    print(f'empirical l2 error: {emp_l2_err.item():.3f}')
    msg = f'l2 norm between estimated and true underlying distribution is greater than 1e-1. truncated l2 norm is {best_l2_err}'
    assert best_l2_err <= 1e-1, msg

def test_truncated_weibull_2_dims():
    dims = 2
    k = ch.Tensor([1.0, 1.0])
    lambda_ = ch.Tensor([1.0, 2.0])
    print(f'known k: {k}')
    print(f'true lambda: {lambda_}')

    dist = Weibull(lambda_, k)

    def phi(z):
        return z[:,0] > 1.0

    num_samples = 10000
    S, alpha = generate_truncated_dataset(dist, phi, num_samples)

    emp_lambda_ = S.pow(k).mean(0).pow(1.0/k) 
    print(f'alpha: {alpha}')
    print(f'num truncated samples: {S.size(0)}')
    print(f'emp_lambda: {emp_lambda_}') 

    args = Parameters({
                    'iterations': 2500, 
                    'trials': 1,
                    'batch_size': 10,
                    'num_samples': 1000, 
                    'verbose': True, 
                    'optimizer': 'sgd',
                    'lr': 1e-2,
                }) 
    
    truncated = distributions.TruncatedWeibull(args,
                                           phi, 
                                           alpha, 
                                           dims, 
                                           k)
    truncated.fit(S)
    
    best_lambda = truncated.best_lambda_ 
    print(f'best lambda:\n {best_lambda.T}')
    best_l2_err = (best_lambda - lambda_).norm(p=2)
    print(f'truncated l2 error: {best_l2_err.item():.3f}')

    ema_lambda = truncated.ema_lambda_ 
    print(f'ema lambda:\n {ema_lambda.T}')
    ema_l2_err = (ema_lambda - lambda_).norm(p=2)
    print(f'ema l2 error: {ema_l2_err}')

    avg_lambda = truncated.avg_lambda_
    print(f'avg lambda:\n {avg_lambda.T}')
    avg_l2_err = (avg_lambda - lambda_).norm(p=2)
    print(f'avg l2 error: {avg_l2_err}')
        
    # check performance
    emp_l2_err = (emp_lambda_ - lambda_).norm(p=2)
    print(f'empirical l2 error: {emp_l2_err.item():.3f}')
    msg = f'l2 norm between estimated and true underlying distribution is greater than 1e-1. truncated l2 norm is {best_l2_err}'
    assert best_l2_err <= 1e-1, msg

def test_truncated_weibull_2_dims_diff_scale():
    dims = 2
    k = ch.Tensor([2.0, 3.0])
    lambda_ = ch.Tensor([1.0, 2.0])
    print(f'known k: {k}')
    print(f'true lambda: {lambda_}')

    dist = Weibull(lambda_, k)

    def phi(z):
        return z[:,0] > 1.0

    num_samples = 10000
    S, alpha = generate_truncated_dataset(dist, phi, num_samples)

    emp_lambda_ = S.pow(k).mean(0).pow(1.0/k) 
    print(f'alpha: {alpha}')
    print(f'num truncated samples: {S.size(0)}')
    print(f'emp_lambda: {emp_lambda_}') 

    args = Parameters({
                    'iterations': 2500, 
                    'trials': 1,
                    'batch_size': 10,
                    'num_samples': 1000, 
                    'verbose': True, 
                    'optimizer': 'sgd',
                    'lr': 1e-2,
                }) 
    
    truncated = distributions.TruncatedWeibull(args,
                                           phi, 
                                           alpha, 
                                           dims, 
                                           k)
    truncated.fit(S)
    
    best_lambda = truncated.best_lambda_ 
    print(f'best lambda:\n {best_lambda.T}')
    best_l2_err = (best_lambda - lambda_).norm(p=2)
    print(f'truncated l2 error: {best_l2_err.item():.3f}')

    ema_lambda = truncated.ema_lambda_ 
    print(f'ema lambda:\n {ema_lambda.T}')
    ema_l2_err = (ema_lambda - lambda_).norm(p=2)
    print(f'ema l2 error: {ema_l2_err}')

    avg_lambda = truncated.avg_lambda_
    print(f'avg lambda:\n {avg_lambda.T}')
    avg_l2_err = (avg_lambda - lambda_).norm(p=2)
    print(f'avg l2 error: {avg_l2_err}')
        
    # check performance
    emp_l2_err = (emp_lambda_ - lambda_).norm(p=2)
    print(f'empirical l2 error: {emp_l2_err.item():.3f}')
    msg = f'l2 norm between estimated and true underlying distribution is greater than 1e-1. truncated l2 norm is {best_l2_err}'
    assert best_l2_err <= 1e-1, msg

def test_truncated_weibull_20_dims_diff_scale():
    dims = 20
    k = ch.Tensor([2.0, 3.0, 1.0, 2.0, 1.0]).repeat(4)
    lambda_ = ch.Tensor([1.0, 2.0, 3.0, 1.0, 5.0]).repeat(4)
    print(f'known k: {k}')
    print(f'true lambda: {lambda_}')

    dist = Weibull(lambda_, k)

    def phi(z):
        return z[:,0] > 1.0

    num_samples = 10000
    S, alpha = generate_truncated_dataset(dist, phi, num_samples)

    emp_lambda_ = S.pow(k).mean(0).pow(1.0/k) 
    print(f'alpha: {alpha}')
    print(f'num truncated samples: {S.size(0)}')
    print(f'emp_lambda: {emp_lambda_}') 

    args = Parameters({
                    'iterations': 2500, 
                    'trials': 1,
                    'batch_size': 10,
                    'num_samples': 1000, 
                    'verbose': True, 
                    'optimizer': 'sgd',
                    'lr': 1e-2,
                }) 
    
    truncated = distributions.TruncatedWeibull(args,
                                           phi, 
                                           alpha, 
                                           dims, 
                                           k)
    truncated.fit(S)
    
    best_lambda = truncated.best_lambda_ 
    print(f'best lambda:\n {best_lambda.T}')
    best_l2_err = (best_lambda - lambda_).norm(p=2)
    print(f'truncated l2 error: {best_l2_err.item():.3f}')

    ema_lambda = truncated.ema_lambda_ 
    print(f'ema lambda:\n {ema_lambda.T}')
    ema_l2_err = (ema_lambda - lambda_).norm(p=2)
    print(f'ema l2 error: {ema_l2_err}')

    avg_lambda = truncated.avg_lambda_
    print(f'avg lambda:\n {avg_lambda.T}')
    avg_l2_err = (avg_lambda - lambda_).norm(p=2)
    print(f'avg l2 error: {avg_l2_err}')
        
    # check performance
    emp_l2_err = (emp_lambda_ - lambda_).norm(p=2)
    print(f'empirical l2 error: {emp_l2_err.item():.3f}')
    msg = f'l2 norm between estimated and true underlying distribution is greater than 1e-1. truncated l2 norm is {best_l2_err}'
    assert best_l2_err <= 1e-1, msg
