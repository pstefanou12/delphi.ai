"""
This module is used to set up arguments and defaults.
"""

import torch as ch
from torch import Tensor

from .Function import TruncatedMSE, TruncatedUnknownVarianceMSE

BY_DATASET = 'varies by dataset'
REQ = 'REQUIRED'

REGRESSION_DEFAULTS = {
    'known': {
        'custom_criterion': TruncatedMSE.apply,
        'epochs': 50,
        'lr': 1e-1,
        'num_samples', 100,
        'bias': True,
        'clamp': True,
        'eps': 1e-5,
        'momentum': 0.0,
        'weight_decay': 0.0,
        'step_lr': 10,
        'step_lr_gamma', .1,
        'device': 'cpu',
    },
    'unknown': {
        'custom_criterion': TruncatedUnknownVarianceMSE.apply,
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
        'step_lr_gamma', .1,
        'device': 'cpu'
    },
}


CONFIG_ARGS = [
    ['config-path', str, 'config path for loading in parameters', None],
    ['exp-name', str, 'where to save in (inside out_dir)', None]
]


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
        parser.add_argument(f'--{arg_name}', **kwargs)
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
