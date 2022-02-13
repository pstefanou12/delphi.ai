# distribution tests 

from re import A
import unittest
import numpy as np
import torch as ch
from torch import Tensor
from torch.distributions import MultivariateNormal, Uniform
from torch.distributions.kl import kl_divergence
from torch.distributions.multivariate_normal import _batch_mahalanobis
from torch.distributions.transforms import SigmoidTransform
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.nn import CosineSimilarity
from torch.nn import MSELoss
import torch.linalg as LA
from sklearn.linear_model import LinearRegression, LogisticRegression

from delphi import stats 
from delphi import oracle
from delphi.utils.helpers import Parameters, cov

# CONSTANT
mse_loss =  MSELoss()
base_distribution = Uniform(0, 1)
transforms_ = [SigmoidTransform().inv]
logistic = TransformedDistribution(base_distribution, transforms_)
cos_sim = CosineSimilarity()


class TestStats(unittest.TestCase): 
    """
    Test suite for the stats module.
    """
    ## right truncated normal distribution with known truncation
    #def test_truncated_regression(self):
    #    W = Uniform(-1, 1)
    #    M = Uniform(-10, 10)
    #    X = M.rsample([10000, 10])
    #    # generate ground truth
    #    gt = ch.nn.Linear(in_features=1, out_features=1)
    #    gt.weight = ch.nn.Parameter(W.sample(ch.Size([1, 10])))
    #    gt.bias = ch.nn.Parameter(W.sample(ch.Size([1, 1])))
    #    noise_var = Tensor([10.0])[...,None]
    #    with ch.no_grad():
    #        # generate data
    #        y = gt(X) + ch.sqrt(noise_var) * ch.randn(X.size(0), 1) 
    #    # generate ground-truth data
    #    phi = oracle.Left_Regression(Tensor([0.0]))
    #    # truncate
    #    indices = phi(y).nonzero()[:,0]
    #    x_trunc, y_trunc = X[indices], y[indices]
    #    alpha = x_trunc.size(0) / X.size(0)
    #    # normalize input features
    #    l_inf = LA.norm(x_trunc, dim=-1, ord=float('inf')).max()
    #    beta = l_inf * (10 ** .5)
    #    x_trunc /= beta
    #    X /= beta 

    #    gt_norm = LinearRegression()
    #    gt_norm.fit(X, y)
    #    gt_ = ch.from_numpy(np.concatenate([gt_norm.coef_.flatten(), gt_norm.intercept_]))

    #    # calculate empirical noise variance for regression 
    #    ols_trunc = LinearRegression()
    #    ols_trunc.fit(x_trunc, y_trunc)
    #    emp_noise_var = ch.from_numpy(ols_trunc.predict(X) - y.numpy()).var(0)

    #    # scale y features
    #    y_trunc_scale = y_trunc / ch.sqrt(noise_var)
    #    phi_scale = oracle.Left_Regression(phi.left / ch.sqrt(noise_var))
    #    # train algorithm
    #    train_kwargs = Parameters({'phi': phi_scale, 
    #                            'alpha': alpha,
    #                            'epochs': 10, 
    #                            'batch_size': 100,
    #                            'normalize': False,
    #                            'noise_var': 1.0}) 
    #    trunc_reg = stats.TruncatedLinearRegression(train_kwargs)
    #    trunc_reg.fit(x_trunc, y_trunc_scale)
    #    w_ = ch.cat([(trunc_reg.coef_).flatten(), trunc_reg.intercept_]) * ch.sqrt(noise_var)
    #    known_mse_loss = mse_loss(gt_, w_.flatten())
    #    self.assertTrue(known_mse_loss <= 3e-1, f'known mse loss: {known_mse_loss}')
    #    
    #   # scale y features by empirical noise variance
    #    y_trunc_emp_scale = y_trunc / ch.sqrt(emp_noise_var)
    #    phi_emp_scale = oracle.Left_Regression(phi.left / ch.sqrt(emp_noise_var))
    #    # train algorithm
    #    train_kwargs = Parameters({'phi': phi_emp_scale, 
    #                            'alpha': alpha,
    #                            'epochs': 10, 
    #                            'batch_size': 10,
    #                            'normalize': False,})
    #    unknown_trunc_reg = stats.TruncatedLinearRegression(train_kwargs)
    #    unknown_trunc_reg.fit(x_trunc, y_trunc_emp_scale)
    #    w_ = ch.cat([(unknown_trunc_reg.coef_).flatten(), unknown_trunc_reg.intercept_]) * ch.sqrt(emp_noise_var)
    #    noise_var_ = unknown_trunc_reg.variance_ * emp_noise_var
    #    unknown_mse_loss = mse_loss(gt_, w_.flatten())
    #    unknown_var_l1 = ch.abs(noise_var - noise_var_)
    #    self.assertTrue(unknown_mse_loss <= 3e-1, f'unknown mse loss: {unknown_mse_loss}')
#   #     self.assertTrue(unknown_var_l1 <= 3e-1)

    def test_truncated_logistic_regression(self):
        d, k = 10, 1
        # ground-truth logistic regression model 
        gt = ch.nn.Linear(in_features=d, out_features=k, bias=True)
        gt.weight = ch.nn.Parameter(ch.ones(k, d))
        gt.bias = ch.nn.Parameter(ch.ones(1, k))

        # input features
        M = MultivariateNormal(ch.zeros(d), ch.eye(d)) 
        X = M.sample([10000])
        # latent variables
        z = gt(X) + logistic.sample([X.size(0), 1])
        # classification
        y = (z > 0).float()
        # generate ground-truth data
        phi = oracle.Left_Regression(Tensor([-.5]))
        phi = oracle.Identity()
        indices = phi(z).flatten().nonzero().flatten()
        x_trunc = X[indices]
        y_trunc = y[indices]
        alpha = x_trunc.size(0) / X.size(0)
        print(f'alpha: {alpha}')
        
        log_reg = LogisticRegression(penalty='none', fit_intercept=True)
        log_reg.fit(X, y.flatten())
        log_reg_ = ch.from_numpy(np.concatenate([log_reg.coef_.flatten(), log_reg.intercept_]))
        pred = log_reg.predict(X)
        acc = np.equal(pred, y.flatten()).sum() / len(y)
        print(f'sklearn acc: {acc}')

        trunc_sklearn = LogisticRegression(penalty='none', fit_intercept=True)
        trunc_sklearn.fit(x_trunc, y_trunc.flatten())
        trunc_sklearn_ = ch.from_numpy(np.concatenate([trunc_sklearn.coef_.flatten(), trunc_sklearn.intercept_]))
        trunc_sklearn_cos_sim = cos_sim(trunc_sklearn_[None,...], log_reg_[None,...])
        pred = trunc_sklearn.predict(X)
        acc = np.equal(pred, y.flatten()).sum() / len(y)
        print(f'trunc sklearn acc: {acc}')
        print(f'trunc sklearn cos sim: {trunc_sklearn_cos_sim}')
        train_kwargs = Parameters({'phi': phi,
                            'alpha': alpha,
                            'fit_intercept': True, 
                            'normalize': False, 
                            'batch_size': 100,
                            'epochs': 30,
                            'trials': 1, 
                            'verbose': True,
                            'num_samples': 100})
        trunc_log_reg = stats.TruncatedLogisticRegression(train_kwargs)
        trunc_log_reg.fit(x_trunc, y_trunc) 
        trunc_log_reg_ = ch.cat([trunc_log_reg.coef_.flatten(), trunc_log_reg.intercept_])
        trunc_cos_sim = float(cos_sim(trunc_log_reg_[None,...], log_reg_[None,...]))
        trunc_log_reg_pred = trunc_log_reg.predict(X)
        trunc_log_reg_acc = trunc_log_reg_pred.eq(y).sum() / len(y)
        print(f'trunc log reg accuracy: {trunc_log_reg_acc}')
        print(f'trunc cos sim: {trunc_cos_sim}')
        self.assertTrue(trunc_cos_sim >= .8, f'trunc cos sim: {trunc_cos_sim}')
        
        train_kwargs = Parameters({'phi': phi,
                            'alpha': alpha,
                            'fit_intercept': True, 
                            'normalize': False, 
                            'batch_size': 100,
                            'epochs': 30,
                            'trials': 1,
                            'multi_class': 'multinomial', 
                            'verbose': True,
                            'num_samples': 100})
        trunc_multi_log_reg = stats.TruncatedLogisticRegression(train_kwargs)
        trunc_multi_log_reg.fit(x_trunc, y_trunc.flatten().long()) 
        trunc_multi_log_reg_ = ch.cat([trunc_multi_log_reg.coef_[1] - trunc_multi_log_reg.coef_[0], 
        (trunc_multi_log_reg.intercept_[1] - trunc_multi_log_reg.intercept_[0])[...,None]])
        trunc_multi_cos_sim = float(cos_sim(trunc_multi_log_reg_[None,...], log_reg_[None,...]))
        trunc_multi_log_reg_pred = trunc_multi_log_reg.predict(X)
        trunc_multi_log_reg_acc = trunc_multi_log_reg_pred.eq(y.flatten()).sum() / len(y)
        print(f'trunc multi log reg accuracy: {trunc_multi_log_reg_acc}')
        print(f'trunc multi cos sim: {trunc_multi_cos_sim}')
        self.assertTrue(trunc_cos_sim >= .8, f'trunc multi cos sim: {trunc_multi_cos_sim}')

if __name__ == '__main__':
    unittest.main()
