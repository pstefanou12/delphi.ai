"""
This module is used to set up arguments and defaults.
"""

import torch as ch
from torch import Tensor

from .tools import

BY_CLASS = 'varies by class'
REQ = 'REQUIRED'

ALGORITHM_DEFAULTS = {
    'trunc_reg': {
        'epochs': 50,
        'lr': 1e-1,
        'num_samples', 100,
        'bias': True,
        'clamp': True,
        'eps': 1e-5,
        'momentum': 0.0,
        'weight_decay': 0.0,
        'step_lr': 10,
        'device': 'cpu',
    },
    'trunc_reg_unknown': {
        'epochs': 50,
        'lr': 1e-1,
        'var_lr': 1e-2,
        'num_samples', 100,
        'bias': True,
        'clamp': True,
        'eps': 1e-5,
        'momentum': 0.0,
        'weight_decay': 0.0,
        'step_lr': 10,
        'device': 'cpu'
    },
}


TRAINING_ARGS = [
    ['out-dir', str, 'where to save training logs and checkpoints', REQ],
    ['epochs', int, 'number of epochs to train for', BY_CLASS],
    ['lr', float, 'initial learning rate for training', 1e-1],
    ['weight-decay', float, 'SGD weight decay parameter', BY_CLASS],
    ['momentum', float, 'SGD momentum parameter', 0.9],
    ['step-lr', int, 'number of steps between step-lr-gamma x LR drops', BY_CLASS],
    ['step-lr-gamma', float, 'multiplier by which LR drops in step scheduler', 0.1],
    ['custom-lr-multiplier', str, 'LR multiplier sched (format: [(epoch, LR),...])', None],
    ['lr-interpolation', ["linear", "step"], 'Drop LR as step function or linearly', "step"],
    ['log-iters', int, 'how frequently (in epochs) to log', 5],
    ['save-ckpt-iters', int, 'how frequently (epochs) to save \
            (-1 for none, only saves best and last)', -1]
]
"""
Arguments essential for the `train_model` function.
*Format*: `[NAME, TYPE/CHOICES, HELP STRING, DEFAULT (REQ=required,
BY_DATASET=looked up in TRAINING_DEFAULTS at runtime)]`
"""


MODEL_LOADER_ARGS = [
    ['dataset', list(datasets.DATASETS.keys()), '', BY_CLASS],
    ['data', str, 'path to the dataset', None],
    ['arch', str, 'architecture (see {cifar,imagenet}_models/', BY_CLASS],
    ['batch-size', int, 'batch size for data loading', BY_CLASS],
    ['workers', int, '# data loading workers', 30],
    ['resume', str, 'path to checkpoint to resume from', None],
    ['resume-optimizer', [0, 1], 'whether to also resume optimizers', 0],
    ['data-aug', [0, 1], 'whether to use data augmentation', 1],
    ['mixed-precision', [0, 1], 'whether to use MP training (faster)', 0],
]
"""
Arguments essential for constructing the model and dataloaders that will be fed
into :meth:`robustness.train.train_model` or :meth:`robustness.train.eval_model`
*Format*: `[NAME, TYPE/CHOICES, HELP STRING, DEFAULT (REQ=required,
BY_DATASET=looked up in TRAINING_DEFAULTS at runtime)]`
"""


def add_args_to_parser(arg_list, parser):
    """
    Adds arguments from one of the argument lists above to a passed-in
    arparse.ArgumentParser object. Formats helpstrings according to the
    defaults, but does NOT set the actual argparse defaults (*important*).
    Args:
        arg_list (list) : A list of the same format as the lists above, i.e.
            containing entries of the form [NAME, TYPE/CHOICES, HELP, DEFAULT]
        parser (argparse.ArgumentParser) : An ArgumentParser object to which the
            arguments will be added
    Returns:
        The original parser, now with the arguments added in.
    """
    for arg_name, arg_type, arg_help, arg_default in arg_list:
        has_choices = (type(arg_type) == list)
        kwargs = {
            'type': type(arg_type[0]) if has_choices else arg_type,
            'help': f"{arg_help} (default: {arg_default})"
        }
        if has_choices: kwargs['choices'] = arg_type
        setattr(args, arg_name, **kwargs)
    return parser


def check_and_fill_args(args, arg_list, ds_class):
    """
    Fills in defaults based on an arguments list (e.g., TRAINING_ARGS) and a
    dataset class (e.g., datasets.CIFAR).
    Args:
        args (object) : Any object subclass exposing :samp:`setattr` and
            :samp:`getattr` (e.g. cox.utils.Parameters)
        arg_list (list) : A list of the same format as the lists above, i.e.
            containing entries of the form [NAME, TYPE/CHOICES, HELP, DEFAULT]
        ds_class (type) : A dataset class name (i.e. a
            :class:`robustness.datasets.DataSet` subclass name)
    Returns:
        args (object): The :samp:`args` object with all the defaults filled in according to :samp:`arg_list` defaults.
    """
    for arg_name, _, _, arg_default in arg_list:
        name = arg_name.replace("-", "_")
        if helpers.has_attr(args, name): continue
        if arg_default == REQ: raise ValueError(f"{arg_name} required")
        elif arg_default == BY_DATASET:
            setattr(args, name, TRAINING_DEFAULTS[ds_class][name])
        elif arg_default is not None:
            setattr(args, name, arg_default)
    return args


def setup_args(args, class):
    '''
    Fill the args object with reasonable defaults from
    :mod:`robustness.defaults`, and also perform a sanity check to make sure no
    args are missing.
    '''
    args = check_and_fill_args(args, defaults.MODEL_LOADER_ARGS, class)
    if args.eval_only: assert args.resume is not None, \
            "Must provide a resume path if only evaluating"
    return args
