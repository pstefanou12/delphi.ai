# distribution tests 

import unittest
import torch as ch
from torch import Tensor
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.kl import kl_divergence

from delphi import distributions 
from delphi import oracle
from delphi.utils.helpers import Parameters

class TestDistributions(unittest.TestCase): 
    """
    Test suite for the distribution module.
    """

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


if __name__ == '__main__':
    unittest.main()
