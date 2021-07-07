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

