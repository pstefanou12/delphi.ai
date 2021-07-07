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
import os
import git
import math

from . import constants as consts


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

        
# logistic distribution
base_distribution = Uniform(0, 1)
transforms_ = [SigmoidTransform().inv]
logistic = TransformedDistribution(base_distribution, transforms_)


class ProcedureComplete(Exception): 
    def __init__(self, message='procedure complete'): 
        super(ProcedureComplete, self).__init__(message)

