"""
Helper code (functions, classes, etc.)
"""


import torch as ch
from torch import Tensor
from torch.distributions import Uniform
from torch.distributions.transforms import SigmoidTransform
from torch.distributions.transformed_distribution import TransformedDistribution
import torch.nn as nn
import cox
from typing import NamedTuple
import pprint

from . import constants as consts

# CONSTANTS
JUPYTER = 'jupyter'
TERMINAL = 'terminal'
IPYTHON = 'ipython'
ZMQ='zmqshell'
COLAB='google.colab'

class Parameters():
    '''
    Parameters class, just a nice way of accessing a dictionary

    .. code-block:: python

        ps = Parameters({"a": 1, "b": 3})
        ps.A # returns 1
    '''
    def __init__(self, params):
        super().__setattr__('params', params)

        # ensure no overlapping (in case) params
        collisions = set()
        for k in self.params.keys():
            collisions.add(k.lower())

        assert len(collisions) == len(self.params.keys())

    def as_dict(self):
        return self.params

    def __getattr__(self, x):
        if x in vars(self):
            return vars(self)[x]

        k = x.lower()
        if k not in self.params:
            return None

        return self.params[k]

    def __setattr__(self, x, v):
        # Fix for some back-compatibility with some pickling bugs
        if x == 'params':
            super().__setattr__(x, v)
            return

        if x in vars(self):
            vars(self)[x.lower()] = v

        self.params[x.lower()] = v

    def __delattr__ (self, key):
        del self.params[key]

    def __iter__ (self):
        return iter(self.params)

    def __len__ (self):
        return len(self.params)

    def __str__(self):
        pp = pprint.PrettyPrinter()
        return pp.pformat(self.params)

    def __repr__(self):
        return str(self)

    def __getstate__(self):
        return self.params

    def __contains__(self, x):
        return x in self.params

    def __setstate__(self, x):
        self.params = x



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


def censored_sample_nll(x):
    # calculates the negative log-likelihood for one sample of a censored normal
    return ch.cat([-.5*ch.bmm(x.unsqueeze(2), x.unsqueeze(1)).flatten(1), x], 1)


def type_of_script():
    """
    Check the program's running environment.
    """
    try:
        ipy_str = str(type(get_ipython()))
        if ZMQ in ipy_str:
            return JUPYTER
        if TERMINEL in ipy_str:
            return IPYTHON
        if COLAB in ipy_str: 
            return COLAB
    except:
        return TERMINAL



def calc_est_grad(func, x, y, rad, num_samples):
    B, *_ = x.shape
    Q = num_samples//2
    N = len(x.shape) - 1
    with ch.no_grad():
        # Q * B * C * H * W
        extender = [1]*N
        queries = x.repeat(Q, *extender)
        noise = ch.randn_like(queries)
        norm = noise.view(B*Q, -1).norm(dim=-1).view(B*Q, *extender)
        noise = noise / norm
        noise = ch.cat([-noise, noise])
        queries = ch.cat([queries, queries])
        y_shape = [1] * (len(y.shape) - 1)
        l = func(queries + rad * noise, y.repeat(2*Q, *y_shape)).view(-1, *extender)
        grad = (l.view(2*Q, B, *extender) * noise.view(2*Q, B, *noise.shape[1:])).mean(dim=0)
    return grad


class InputNormalize(ch.nn.Module):
    '''
    A module (custom layer) for normalizing the input to have a fixed
    mean and standard deviation (user-specified).
    '''
    def __init__(self, new_mean, new_std):
        super(InputNormalize, self).__init__()
        new_std = new_std[..., None, None]
        new_mean = new_mean[..., None, None]

        self.register_buffer("new_mean", new_mean)
        self.register_buffer("new_std", new_std)

    def forward(self, x):
        x = ch.clamp(x, 0, 1)
        x_normalized = (x - self.new_mean)/self.new_std
        return x_normalized


def ckpt_at_epoch(num):
    return '%s_%s' % (num, consts.CKPT_NAME)


def setup_store_with_metadata(args, store):
    '''
    Sets up a store for training according to the arguments object. See the
    argparse object above for options.
    '''
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
            return [ch.round(ch.sigmoid(output)).eq(ch.round(target)).float().mean(), -1.0] 
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


class Bounds(NamedTuple): 
    lower: Tensor
    upper: Tensor


class FakeReLU(ch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class FakeReLUM(nn.Module):
    def forward(self, x):
        return FakeReLU.apply(x)


class SequentialWithArgs(ch.nn.Sequential):
    def forward(self, input, *args, **kwargs):
        vs = list(self._modules.values())
        l = len(vs)
        for i in range(l):
            if i == l-1:
                input = vs[i](input, *args, **kwargs)
            else:
                input = vs[i](input)
        return input


class DataPrefetcher():
    def __init__(self, loader, stop_after=None):
        self.loader = loader
        self.dataset = loader.dataset
        self.stream = ch.cuda.Stream()
        self.stop_after = stop_after
        self.next_input = None
        self.next_target = None

    def __len__(self):
        return len(self.loader)

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loaditer)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with ch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)

    def __iter__(self):
        count = 0
        self.loaditer = iter(self.loader)
        self.preload()
        while self.next_input is not None:
            ch.cuda.current_stream().wait_stream(self.stream)
            input = self.next_input
            target = self.next_target
            self.preload()
            count += 1
            yield input, target
            if type(self.stop_after) is int and (count > self.stop_after):
                break


# logistic distribution
base_distribution = Uniform(0, 1)
transforms_ = [SigmoidTransform().inv]
logistic = TransformedDistribution(base_distribution, transforms_)


class ProcedureComplete(Exception): 
    def __init__(self, message='procedure complete'): 
        super(ProcedureComplete, self).__init__(message)


class PSDError(Exception): 
    def __init__(self, message='psd error'): 
        super(PSDError, self).__init__(message)
        self.message = message


