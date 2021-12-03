"""
Helper code (functions, classes, etc.)
"""


import torch as ch
from torch import Tensor
import torch.linalg as LA
from torch.distributions import Uniform
from torch.distributions.transforms import SigmoidTransform
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import cox
from typing import NamedTuple
import os
import git
import math
import pprint

from . import constants as consts

# CONSTANTS
REQ = 'required'
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


class LinearUnknownVariance(nn.Module):
    """
    Linear layer with unknown noise variance.
    """
    def __init__(self, in_features, out_features, bias=True):
        """
        :param in_features: number of in features
        :param out_features: number of out features 
        :param bias: bias term or not
        """
        # choose initial noise variance from a normal distribution
        super(LinearUnknownVariance, self).__init__()
        self.in_features, self.out_features = in_features, out_features

        # layer parameters
        self.weight = ch.nn.Parameter(Tensor(self.in_features, self.out_features))
        self.bias = ch.nn.Parameter(Tensor(out_features)) if bias else None
        self.lambda_ = ch.nn.Parameter(Tensor(1, 1))

        # initialize weights and biases
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5)) # weight init
        if self.bias: 
            nn.init.uniform_(self.bias, -5, 5)  # bias init 
        nn.init.uniform_(self.lambda_, -5, 5)  # lambda init 

    def forward(self, x):
        # reparamaterize weight and variance estimates
        var = self.lambda_.clone().detach().inverse()
        w = self.weight * var
        if self.bias is not None:
            return ch.add(x@w.T, self.bias * var)
        return x@w.T


class Normalize:
    """
    Normalizes the input covariate features for truncated
    regression.
    """

    def __init__(self):
        '''
        Args:
            X (torch.Tensor): regression input features; shape expected to be n (number of samples) by d (number of dimensions)
        '''
        super(Normalize).__init__()
        self._l_inf, self._beta = None, None

    def fit_transform(self, X):
        '''
        Normalize input features truncated regression
        '''
        # normalize input features
        self._l_inf = LA.norm(X, dim=-1, ord=float('inf')).max()
        self._beta = self._l_inf * (X.size(1) ** .5)
        return self

    def transform(self, X):      
        return X / self._beta

    @property
    def beta(self):
        return self._beta

    @property
    def l_inf(self):
        return self._l_inf


def make_train_and_val(args, X, y): 
    # separate into training and validation set
    rand_indices = ch.randperm(X.size(0))
    val = int(args.val * X.size(0))
    train_indices, val_indices = rand_indices[val:], rand_indices[:val]
    X_train,y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]

    # normalize input covariates
    if args.normalize:
        train_norm = Normalize().fit_transform(X_train)
        X_train = train_norm.transform(X_train)
        val_norm = Normalize().fit_transform(X_val)
        X_val = val_norm.transform(X_val)

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=args.workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.workers)

    return train_loader, val_loader

def make_train_and_val_distr(args, S): 
    # separate into training and validation set
    rand_indices = ch.randperm(S.size(0))
    val = int(args.val * X.size(0))
    train_indices, val_indices = rand_indices[args.val:], rand_indices[:args.val]
    X_train = S[train_indices]
    X_val = S[val_indices]
    train_ds = CensoredNormalDataset(X_train)
    val_ds = CensoredNormalDataset(X_val)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size)
    val_loader = DataLoader(val_ds, batch_size=len(val_ds))

    return train_loader, val_loader

def check_and_fill_args(args, defaults): 
        '''
        Checks args (algorithm hyperparameters) and makes sure that all required parameters are 
        given.
        '''
        for arg_name, (arg_type, arg_default) in defaults.items():
            if has_attr(args, arg_name):
                # check to make sure that hyperparameter inputs are the same type
                if isinstance(args.__getattr__(arg_name), arg_type): continue
                raise ValueError('Arg: {} is not correct type: {}. Fix args dict and run again.'.format(arg_name, arg_type))
            if arg_default == REQ: raise ValueError(f"{arg_name} required")
            elif arg_default is not None: 
                setattr(args, arg_name, arg_default)
        return args

        
# logistic distribution
base_distribution = Uniform(0, 1)
transforms_ = [SigmoidTransform().inv]
logistic = TransformedDistribution(base_distribution, transforms_)


class ProcedureComplete(Exception): 
    def __init__(self, message='procedure complete'): 
        super(ProcedureComplete, self).__init__(message)

