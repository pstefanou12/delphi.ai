# distribution tests 

import unittest
import numpy as np
import torch as ch
from torch import Tensor
from torch.distributions import MultivariateNormal, Uniform
from torch.distributions.kl import kl_divergence
from torch.distributions.multivariate_normal import _batch_mahalanobis
from torch.nn import MSELoss
from scipy.linalg import sqrtm
import torch.linalg as LA
from sklearn.linear_model import LinearRegression

from delphi import stats 
from delphi import oracle
from delphi.utils.helpers import Parameters, cov

# CONSTANT
mse_loss =  MSELoss()


class TestStats(unittest.TestCase): 
    """
    Test suite for the stats module.
    """
    # right truncated normal distribution with known truncation
    def test_truncated_regression(self):
        W = Uniform(-1, 1)
        M = Uniform(-10, 10)
        X = M.rsample([10000, 10])
        # generate ground truth
        gt = ch.nn.Linear(in_features=10, out_features=1)
        gt.weight = ch.nn.Parameter(W.sample(ch.Size([1, 10])))
        gt.bias = ch.nn.Parameter(W.sample(ch.Size([1, 1])))
        noise_var = Tensor([10.0])[...,None]
        with ch.no_grad():
            # generate data
            y = gt(X) + ch.sqrt(noise_var) * ch.randn(X.size(0), 1) 
        # generate ground-truth data
        phi = oracle.Left_Regression(Tensor([0.0]))
        # truncate
        indices = phi(y).nonzero()[:,0]
        x_trunc, y_trunc = X[indices], y[indices]
        alpha = x_trunc.size(0) / X.size(0)
        # normalize input features
        l_inf = LA.norm(x_trunc, dim=-1, ord=float('inf')).max()
        beta = l_inf * (10 ** .5)
        x_trunc /= beta
        X /= beta 

        gt_norm = LinearRegression()
        gt_norm.fit(X, y)
        gt_ = ch.from_numpy(np.concatenate([gt_norm.coef_.flatten(), gt_norm.intercept_]))

        # scale y features
        y_trunc_scale = y_trunc / ch.sqrt(noise_var)
        phi_scale = oracle.Left_Regression(phi.left / ch.sqrt(noise_var))
    
        # train algorithm
        train_kwargs = Parameters({'phi': phi_scale, 
                                'alpha': alpha,
                                'epochs': 10, 
                                'batch_size': 10,
                                'normalize': False,
                                'noise_var': 1.0}) 
        trunc_reg = stats.TruncatedLinearRegression(train_kwargs)
        trunc_reg.fit(x_trunc, y_trunc_scale)
        w_ = ch.cat([(trunc_reg.coef_).flatten(), trunc_reg.intercept_]) * ch.sqrt(noise_var)
        known_mse_loss = mse_loss(gt_, w_)

        self.assertTrue(known_mse_loss <= 3e-1)

if __name__ == '__main__':
    unittest.main()
