
"""
Truncated multivariate normal distribution without oracle access (ie. unknown truncation set)
"""

import torch as ch
from torch import Tensor
from torch.distributions.multivariate_normal import MultivariateNormal
from cox.utils import Parameters
import config

from .stats import stats
from .unknown_truncation_normal import TruncatedMultivariateNormalNLL, TruncatedNormalProjectionSet
from ..oracle import oracle
from ..train import train_model
from ..utils.helpers import Exp_h
from ..utils.datasets import TRUNCATED_MULTIVARIATE_NORMAL_REQUIRED_ARGS, TRUNCATED_MULTIVARIATE_NORMAL_OPTIONAL_ARGS, \
    TruncatedMultivariateNormal, DataSet
from ..utils import defaults


class truncated_multivariate_normal(stats):
    """
    Truncated multivariate normal distribution class.
    """
    def __init__(
            self,
            phi: oracle,
            alpha: float,
            args: Parameters,
            device: str = 'cpu',
            **kwargs):
        super(truncated_multivariate_normal, self).__init__()
        # check algorithm hyperparameters
        config.args = defaults.check_and_fill_args(args, defaults.HERMITE_ARGS, TruncatedMultivariateNormal)
        # add oracle and survival prob to parameters
        config.args.__setattr__('phi', phi)
        config.args.__setattr__('alpha', alpha)
        self._multivariate_normal = None
        # intialize loss function and add custom criterion to hyperparameters
        self.criterion = TruncatedMultivariateNormalNLL.apply
        config.args.__setattr__('custom_criterion', self.criterion)

    def fit(self, S: Tensor):
        # create dataset and dataloader
        ds_kwargs = {
            'custom_class_args': {
                'S': S},
            'custom_class': TruncatedMultivariateNormal,
            'transform_train': None,
            'transform_test': None,
            'label_mapping': None}
        ds = DataSet('truncated_normal', TRUNCATED_MULTIVARIATE_NORMAL_REQUIRED_ARGS,
                     TRUNCATED_MULTIVARIATE_NORMAL_OPTIONAL_ARGS, data_path=None, **ds_kwargs)
        loaders = ds.make_loaders(workers=config.args.workers, batch_size=config.args.batch_size)
        # initialize model with empiricial estimates
        self._multivariate_normal = MultivariateNormal(loaders[0].dataset.loc, loaders[0].dataset.covariance_matrix)
        # keep track of gradients for mean and covariance matrix
        self._multivariate_normal.loc.requires_grad, self._multivariate_normal.covariance_matrix.requires_grad = True, True
        # initialize projection set and add iteration hook to hyperparameters
        self.projection_set = TruncatedMultivariateNormalProjectionSet(self._multivariate_normal.loc,
                                                                       self._multivariate_normal.covariance_matrix)
        config.args.__setattr__('iteration_hook', self.projection_set)
        # exponent class
        self.exp_h = Exp_h(self._multivariate_normal.loc, self._multivariate_normal.covariance_matrix)
        config.args.__setattr__('exp_h', self.exp_h)
        # run PGD to predict actual estimates
        return train_model(config.args, self._multivariate_normal, loaders,
                           update_params=[self._multivariate_normal.loc, self._multivariate_normal.covariance_matrix])


class TruncatedMultivariateNormalProjectionSet(TruncatedNormalProjectionSet):
    """
    Truncated multivariate normal distribution with unknown truncation projection set.
    """

    def __init__(self, emp_loc, emp_covariance_matrix):
        """
        Args:
            emp_loc (torch.Tensor): empirical mean
            emp_scale (torch.Tensor): empirical variance
            alpha (torch.Tensor): lower bound on survival probability for distribution
            r (float): projection set radius
            clamp (bool): boolean for clamp heuristic
        """
        super().__init__(emp_loc, emp_covariance_matrix.svd()[1])

    def __call__(self, M, i, loop_type, inp, target):
        if config.args.clamp:
            u, s, v = M.covariance_matrix.svd()  # decompose covariance estimate
            M.loc.data = ch.cat(
                [ch.clamp(M.loc[i], float(self.loc_bounds.lower[i]), float(self.loc_bounds.upper[i])).unsqueeze(0) for i in
                 range(M.loc.shape[0])])
            M.covariance_matrix.data = u.matmul(ch.diag(ch.cat(
                [ch.clamp(s[i], float(self.scale_bounds.lower[i]), float(self.scale_bounds.upper[i])).unsqueeze(0) for i in
                 range(s.shape[0])]))).matmul(v.t())
        else:
            pass