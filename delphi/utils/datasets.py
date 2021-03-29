import torch as ch
from torch import Tensor
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.linear_model import LinearRegression, LogisticRegression
import copy
import warnings

from .helpers import censored_sample_nll, cov


class CensoredNormalDataset(Dataset):
    def __init__(
            self,
            S: Tensor):
        # empirical mean and variance
        self._loc = ch.mean(S, dim=0)
        self._var = ch.var(S, dim=0)
        # normalize data and apply gradient
        self.S = censored_sample_nll(S) 

    def __len__(self): 
        return self.S.size(0)
    
    def __getitem__(self, idx):
        return [self.S[idx],]
    
    @property
    def loc(self): 
        return self._loc.clone()
    
    @property
    def var(self): 
        return self._var.clone()
    
    
class CensoredMultivariateNormalDataset(Dataset):
    def __init__(
            self,
            S: Tensor):
        # empirical mean and variance
        self._loc = S.mean(0)
        self._covariance_matrix = cov(S)
        # apply gradient to data
        self.S = censored_sample_nll(S) 

    def __len__(self): 
        return self.S.size(0)
    
    def __getitem__(self, idx):
        return [self.S[idx],]
    
    @property
    def loc(self): 
        return self._loc.clone()
    
    @property
    def covariance_matrix(self): 
        return self._covariance_matrix.clone()


class TruncatedNormalDataset(Dataset): 
    def __init__(
            self,
            S: Tensor):
        self.S = S
        # samples 
        self._loc = ch.mean(S, dim=0)
        self._var = ch.var(S, dim=0)
        # compute gradients
        pdf = ch.exp(Normal(ch.zeros(1), ch.eye(1).flatten()).log_prob(self.S))
        self.loc_grad = pdf*(self._loc - self.S)
        self.var_grad = .5*pdf*(ch.bmm(self.S.unsqueeze(2), self.S.unsqueeze(1)) - self._var - self._loc.unsqueeze(0).matmul(self._loc.unsqueeze(1))).flatten(1)
        
    def __len__(self): 
        return self.S.size(0)
    
    def __getitem__(self, idx):
        """
        :returns: (sample, sample pdf, sample mean coeffcient, sample covariance matrix coeffcient)
        """
        return self.S[idx], self.loc_grad[idx], self.var_grad[idx]
    
    @property
    def loc(self): 
        return self._loc.clone()
    
    @property
    def var(self): 
        return self._var.clone()


class TruncatedMultivariateNormalDataset(Dataset):
    def __init__(
            self,
            S: Tensor):
        # samples 
        self.S = S
        self._loc = ch.mean(S, dim=0)
        self._covariance_matrix = cov(S)
        # compute gradients
        pdf = ch.exp(MultivariateNormal(ch.zeros(self.S.size(1)).double(), ch.eye(self.S.size(1)).double()).log_prob(self.S)).unsqueeze(1)
        # pdf of each sample
        self.loc_grad = pdf*(self._loc - self.S)
        self.cov_grad = (.5*pdf*((ch.bmm(S.unsqueeze(2), S.unsqueeze(1)) - self._covariance_matrix - self._loc.unsqueeze(0).matmul(self._loc.unsqueeze(1))).flatten(1))).unflatten(1, self._covariance_matrix.size())

    def __len__(self): 
        return self.S.size(0)
    
    def __getitem__(self, idx):
        """
        :returns: (sample, sample pdf, sample mean coeffcient, sample covariance matrix coeffcient)
        """
        return self.S[idx], self.loc_grad[idx], self.cov_grad[idx]
    
    @property
    def loc(self): 
        return self._loc.clone()
    
    @property
    def covariance_matrix(self): 
        return self._covariance_matrix.clone()


class TruncatedRegressionDataset(Dataset):
    def __init__(
            self,
            X: Tensor,
            y: Tensor,
            bias: bool=True,
            batch_size: int=100,
            unknown: bool=True):
        self.X = X
        self.y = y
        self.bias = bias
        self.batch_size = batch_size
        self.unknown = unknown

        # empirical estimates for dataset
        self._reg = LinearRegression(fit_intercept=self.bias).fit(self.X, self.y)

        if self.unknown:
            # calculate empirical variance of the residuals
            self.var = ch.var(ch.from_numpy(self._reg.predict(self.X)) - self.y, dim=0).unsqueeze(0)
            self._lambda_ = self.var.inverse()
            self._v = ch.from_numpy(self._reg.coef_) * self.lambda_
            self._v0 = ch.from_numpy(self._reg.intercept_) * self._lambda_ if self._reg.intercept_ else None

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.X)

    @property
    def reg(self):
        return copy.deepcopy(self._reg)

    @property
    def w(self):
        return ch.from_numpy(self._reg.coef_).clone().t()

    @property
    def w0(self):
        return ch.from_numpy(self._reg.intercept_).clone()

    # properties for regression with unknown noise variance
    @property
    def lambda_(self):
        if self._lambda_:
            return self._lambda_.clone()
        warnings.warn("variance is known, so lambda is not defined.")

    @property
    def v(self):
        if self._v is not None:
            return self._v.clone().t()
        warnings.warn("variance in known, so v is not defined")

    @property
    def v0(self):
        if self._v0:
            return self._v0.clone()
        if self._v:
            warnings.warn("no bias term, so v0 is not defined")
        else:
            warnings.warn("variance is known, so v0 is not defined")


class TruncatedLogisticRegressionDataset(Dataset):
    def __init__(
            self,
            X: Tensor,
            y: Tensor,
            bias: bool=True):
        self.X = X
        self.y = y
        self.bias = bias

        # empirical estimates for dataset
        self._log_reg = LogisticRegression()
        self._log_reg.fit(self.X, self.y.flatten())

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.X)

    @property
    def log_reg(self):
        return copy.deepcopy(self._log_reg)

    @property
    def w(self):
        return Tensor(self._log_reg.coef_)

    @property
    def w0(self):
        return Tensor(self._log_reg.intercept_)

