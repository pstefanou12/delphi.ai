"""
Censored multivariate normal distribution with oracle access (ie. known truncation set).
"""

import torch as ch
from torch import Tensor
from torch.distributions.multivariate_normal import MultivariateNormal
from cox.utils import Parameters
import config

from .stats import stats
from ..oracle import oracle
from ..train import train_model
from ..utils.datasets import DataSet, CENSORED_MULTIVARIATE_NORMAL_REQUIRED_ARGS,\
    CENSORED_MULTIVARIATE_NORMAL_OPTIONAL_ARGS, CensoredMultivariateNormal
from ..grad import CensoredMultivariateNormalNLL
from ..utils import defaults
from ..utils.helpers import cov


class MultivariateNormal(stats):
    """
    Censored multivariate distribution class.
    """
    def __init__(
            self,
            phi: oracle,
            alpha: Tensor,
            args: Parameters,
            **kwargs):
        """
        """
        super(censored_multivariate_normal, self).__init__()
        # check that algorithm hyperparameters
        config.args = defaults.check_and_fill_args(args, defaults.CENSOR_ARGS, CensoredMultivariateNormal)
        # add oracle and survival prob to parameters
        config.args.__setattr__('phi', phi)
        config.args.__setattr__('alpha', alpha)
        self._multivariate_normal = None
        # intialize loss function and add custom criterion to hyperparameters
        self.criterion = CensoredMultivariateNormalNLL.apply
        config.args.__setattr__('custom_criterion', self.criterion)
        # create instance variables for empirical estimates
        self.emp_loc, self.emp_covariance_matrix = None, None
        self.projection_set = None

    def fit(self, S: Tensor):
        """
        """
        # create dataset and dataloader
        ds_kwargs = {
            'custom_class_args': {
                'S': S},
            'custom_class': CensoredMultivariateNormal,
            'transform_train': None,
            'transform_test': None,
            'label_mapping': None}
        ds = DataSet('censored_multivariate_normal', CENSORED_MULTIVARIATE_NORMAL_REQUIRED_ARGS,
                     CENSORED_MULTIVARIATE_NORMAL_OPTIONAL_ARGS, data_path=None, **ds_kwargs)
        loaders = ds.make_loaders(workers=config.args.workers, batch_size=config.args.batch_size)
        # initialize model with empiricial estimates
        self._multivariate_normal = MultivariateNormal(loaders[0].dataset.loc, loaders[0].dataset.covariance_matrix)
        # keep track of gradients for mean and covariance matrix
        self._multivariate_normal.loc.requires_grad, self._multivariate_normal.covariance_matrix.requires_grad = True, True
        # initialize projection set and add iteration hook to hyperparameters
        self.projection_set = CensoredMultivariateNormalProjectionSet(self._multivariate_normal.loc,
                                                                      self._multivariate_normal.covariance_matrix)
        config.args.__setattr__('iteration_hook', self.projection_set)
        # run PGD to predict actual estimates
        return train_model(config.args, self._multivariate_normal, loaders,
                           update_params=[self._multivariate_normal.loc, self._multivariate_normal.covariance_matrix])


class CensoredMultivariateNormalProjectionSet(CensoredNormalProjectionSet):
    """
    Censored multivariate normal projection set
    """
    def __init__(self, emp_loc, emp_covariance_matrix):
        """
        Args:
            emp_loc (torch.Tensor): empirical mean
            emp_covariance_matrix (torch.Tensor): empirical covariance
        """
        super().__init__(emp_loc, emp_covariance_matrix.svd()[1])

    def __call__(self, M, i, loop_type, inp, target):
        if config.args.clamp:
            u, s, v = M.covariance_matrix.svd()  # decompose covariance estimate
            M.loc.data = ch.cat([ch.clamp(M.loc[i], self.loc_bounds.lower[i], self.loc_bounds.upper[i]).unsqueeze(0) for i in range(M.loc.shape[0])])
            M.covariance_matrix.data = u.matmul(ch.diag(ch.cat([ch.clamp(s[i], self.scale_bounds.lower[i], self.scale_bounds.upper[i]).unsqueeze(0) for i in range(s.shape[0])]))).matmul(v.t())
        else:
            pass


# HELPER FUNCTIONS
def cov(m, rowvar=False):
    '''
    Estimate a covariance matrix given data.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.

    Returns:
        The covariance matrix of the variables.
    '''
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    # clone array so that data is not manipulated in-place
    m_ = m.clone().detach()
    if m_.dim() < 2:
        m_ = m_.view(1, -1)
    if not rowvar and m_.size(0) != 1:
        m_ = m_.t()
    # m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (m_.size(1) - 1)
    m_ -= ch.mean(m_, dim=1, keepdim=True)
    mt = m_.t()  # if complex: mt = m.t().conj()
    return fact * m_.matmul(mt)