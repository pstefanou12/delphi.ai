"""
Helper code (functions, classes, etc.)
"""


import torch as ch
from torch import Tensor
from torch import nn
import cox
from typing import NamedTuple
import os
import git


from .constants import CKPT_NAME


def censored_sample_nll(x):
    return ch.cat([-.5*ch.bmm(x.unsqueeze(2), x.unsqueeze(1)).flatten(1), x], 1)


def ckpt_at_epoch(num):
    return '%s_%s' % (num, CKPT_NAME)


def setup_store_with_metadata(args, store):
    '''
    Sets up a store for training according to the arguments object. See the
    argparse object above for options.
    '''
    # Add git commit to args
    try:
        repo = git.Repo(path=os.path.dirname(os.path.realpath('__file__')),
                            search_parent_directories=True)
        version = repo.head.object.hexsha
    except git.exc.InvalidGitRepositoryError:
        version = __version__
    args.version = version

    args_dict = args.as_dict()
    schema = cox.store.schema_from_dict(args_dict)
    store.add_table('metadata', schema)
    store['metadata'].append_row(args_dict)


def has_attr(obj, k):
    """Checks both that obj.k exists and is not equal to None"""
    try:
        return (getattr(obj, k) is not None)
    except KeyError as e:
        return False
    except AttributeError as e:
        return False


def accuracy(output, target, topk=(1,), exact=False):
    """
        Computes the top-k accuracy for the specified values of k

        Args:
            output (ch.Tensor) : model output (N, classes) or (N, attributes)
                for sigmoid/multitask binary classification
            target (ch.Tensor) : correct labels (N,) [multiclass] or (N,
                attributes) [multitask binary]
            topk (tuple) : for each item "k" in this tuple, this method
                will return the top-k accuracy
            exact (bool) : whether to return aggregate statistics (if
                False) or per-example correctness (if True)

        Returns:
            A list of top-k accuracies.
    """
    with ch.no_grad():
        # Binary Classification
        if len(target.shape) > 1:
            assert output.shape == target.shape, \
                "Detected binary classification but output shape != target shape"
            return [ch.round(ch.sigmoid(output)).eq(ch.round(target)).float().mean()], [-1.0]

        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        res_exact = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float()
            ck_sum = correct_k.sum(0, keepdim=True)
            res.append(ck_sum.mul_(100.0 / batch_size))
            res_exact.append(correct_k)

        if not exact:
            return res
        else:
            return res_exact


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


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


class Bounds(NamedTuple):
    lower: Tensor
    upper: Tensor


class Exp_h:
    def __init__(self, emp_loc, emp_cov):
        self.emp_loc = emp_loc
        self.emp_cov = emp_cov
        self.pi_const = (self.emp_loc.size(0) / 2.0) * ch.log(
            2.0 * Tensor([ch.acos(ch.zeros(1)).item() * 2]).unsqueeze(0))

    def __call__(self, u, B, x):
        """
        returns: evaluates exponential function
        """
        cov_term = ch.bmm(x.unsqueeze(1).matmul(B), x.unsqueeze(2)).flatten(1) / 2.0
        trace_term = ch.trace((B - ch.eye(u.size(0))) * (self.emp_cov + self.emp_loc.matmul(self.emp_loc))).unsqueeze(0)
        loc_term = (x - self.emp_loc).matmul(u.unsqueeze(1))
        return ch.exp(cov_term - trace_term - loc_term + self.pi_const)


class LinearUnknownVariance(nn.Module):
    """
    Linear layer with unknown noise variance. Used for regression models.
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool=True):
        """
        :param lambda_: 1/empirical variance
        :param v: empirical weight*lambda_ estimate
        :param bias: (optional) empirical bias*lambda_ estimate
        """
        super(LinearUnknownVariance, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.v = Parameter(Tensor(out_features, in_features))
        self.lambda_ = Parameter(Tensor(out_features))
        if bias:
            self.bias = Parameter(Tensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        var = self.lambda_.clone().detach().inverse()
        w = self.v*var
        if self.bias.nelement() > 0:
            return x.matmul(w) + self.bias * var
        return x.matmul(w)


def init_process(args, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=args.rank, world_size=args.size)


def setup_store_with_metadata(args):
    '''
    Sets up a store for training according to the arguments object. See the
    argparse object above for options.
    '''
    # Add git commit to args
    try:
        repo = git.Repo(path=os.path.dirname(os.path.realpath(__file__)),
                            search_parent_directories=True)
        version = repo.head.object.hexsha
    except git.exc.InvalidGitRepositoryError:
        version = __version__
    args.version = version

    # Create the store
    store = cox.store.Store(args.out_dir, args.exp_name)
    args_dict = args.__dict__
    schema = cox.store.schema_from_dict(args_dict)
    store.add_table('metadata', schema)
    store['metadata'].append_row(args_dict)

    return store
