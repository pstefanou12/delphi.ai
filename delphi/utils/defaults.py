"""
This module is used to set up arguments and defaults.
"""

from torch.utils.data import TensorDataset


from . import datasets
from .helpers import has_attr
from . import constants as consts

BY_DATASET = 'varies by dataset'
REQ = 'REQUIRED'

TRAINING_DEFAULTS = {
    datasets.CIFAR: {
        "epochs": 150,
        "batch_size": 128,
        "weight_decay": 5e-4,
        "step_lr": 50,
        "momentum": .9,
        "score": False,
    },
    datasets.ImageNet: {
        "epochs": 200,
        "batch_size":256,
        "weight_decay":1e-4,
        "step_lr": 50, 
        'momentum': .9,
        "score": False,
    },
    datasets.CensoredNormal: {
        "steps": 1000,
        "batch_size": 10,
        "weight_decay": 5e-4,
        "momentum": 0.0,
        "num_samples": 100,
        "radius": 2.0,
        "step_lr": 10,
        "score": True,
        "tol": 1e-2,
    },
    datasets.CensoredMultivariateNormal: {
        "steps": 1000,
        "batch_size": 10,
        "weight_decay": 5e-4,
        "momentum": 0.0,
        "num_samples": 100,
        "radius": 2.0,
        "step_lr": 10,
        "score": True,
        "tol": 1e-2,
    },
    datasets.TruncatedNormal: {
        "steps": 1000,
        "batch_size": 10,
        "weight_decay": 5e-4,
        "momentum": 0.0,
        "num_samples": 100,
        "radius": 2.0,
        "step_lr": 10,
        "score": True,
        "tol": 1e-2,
        "d": 100,
    },
    datasets.TruncatedMultivariateNormal: {
        "steps": 1000,
        "batch_size": 10,
        "weight_decay": 5e-4,
        "momentum": 0.0,
        "num_samples": 100,
        "radius": 2.0,
        "step_lr": 10,
        "score": True,
        "tol": 1e-2,
        "d": 100,
    },
    datasets.TensorDataset: {
        "steps": 1000,
        "batch_size": 100,
        "weight_decay": 0.0,
        "momentum": 0.0,
        "num_samples": 100,
        "radius": 2.0,
        "step_lr": 10,
        "score": True,
        "tol": 1e-2,
        "clamp": True, 
    },
}
"""
Default hyperparameters for training by dataset (tested for resnet50).
Parameters can be accessed as `TRAINING_DEFAULTS[dataset_class][param_name]`
"""

# default arguments for training deep neural networks
TRAINING_ARGS = [
    ['out-dir', str, 'where to save training logs and checkpoints', REQ],
    ['epochs', int, 'number of epochs to train for', BY_DATASET],
    ['lr', float, 'initial learning rate for training', 0.1],
    ['weight-decay', float, 'SGD weight decay parameter', BY_DATASET],
    ['momentum', float, 'SGD momentum parameter', BY_DATASET],
    ['step-lr', int, 'number of steps between step-lr-gamma x LR drops', BY_DATASET],
    ['step-lr-gamma', float, 'multiplier by which LR drops in step scheduler', 0.1],
    ['custom-lr-multiplier', str, 'LR multiplier sched (format: [(epoch, LR),...])', None],
    ['lr-interpolation', ["linear", "step"], 'Drop LR as step function or linearly', "step"],
    ['adv-train', [0, 1], 'whether to train adversarially', REQ],
    ['adv-eval', [0, 1], 'whether to adversarially evaluate', None],
    ['log-iters', int, 'how frequently (in epochs) to log', 5],
    ['save-ckpt-iters', int, 'how frequently (epochs) to save \
            (-1 for none, only saves best and last)', -1],
    ['parallel', [0, 1], 'train model on multiple GPUs', 0], 
    ['accuracy', [0, 1], 'check model accuracy during training procedure', 1],
]
"""
Arguments essential for the `train_model` function.
*Format*: `[NAME, TYPE/CHOICES, HELP STRING, DEFAULT (REQ=required,
BY_DATASET=looked up in TRAINING_DEFAULTS at runtime)]`
"""

# default arguments for censored algorithm
CENSOR_ARGS = [
    ['steps', int, 'number of epochs to train for', BY_DATASET],
    ['lr', float, 'initial learning rate for training', 0.1],
    ['weight-decay', float, 'SGD weight decay parameter', BY_DATASET],
    ['momentum', float, 'SGD momentum parameter', BY_DATASET],
    ['step-lr', int, 'number of steps between step-lr-gamma x LR drops', BY_DATASET],
    ['step-lr-gamma', float, 'multiplier by which LR drops in step scheduler', 0.1],
    ['custom-lr-multiplier', str, 'LR multiplier sched (format: [(epoch, LR),...])', None],
    ['num_samples', int, 'number of samples to sample at once from a truncated distribution', BY_DATASET],
    ['radius', float, 'initial radius size for PSGD', BY_DATASET],
    ['score', bool, 'determine convergence based off of score', BY_DATASET],
    ['lr-interpolation', ["linear", "step"], 'Drop LR as step function or linearly', "step"],
    ['tol', float, 'gradient tolerance for algorithm convergence', BY_DATASET]
]

# default arguments for unknown truncation algorithm
HERMITE_ARGS = [
    ['steps', int, 'number of epochs to train for', BY_DATASET],
    ['lr', float, 'initial learning rate for training', 0.1],
    ['weight-decay', float, 'SGD weight decay parameter', BY_DATASET],
    ['momentum', float, 'SGD momentum parameter', BY_DATASET],
    ['step-lr', int, 'number of steps between step-lr-gamma x LR drops', BY_DATASET],
    ['step-lr-gamma', float, 'multiplier by which LR drops in step scheduler', 0.1],
    ['custom-lr-multiplier', str, 'LR multiplier sched (format: [(epoch, LR),...])', None],
    ['num_samples', int, 'number of samples to sample at once from a truncated distribution', BY_DATASET],
    ['radius', float, 'initial radius size for PSGD', BY_DATASET],
    ['score', bool, 'determine convergence based off of score', BY_DATASET],
    ['lr-interpolation', ["linear", "step"], 'Drop LR as step function or linearly', "step"],
    ['tol', float, 'gradient tolerance for algorithm convergence', BY_DATASET]
]

# default arguments for truncated regression algorithm
REGRESSION_ARGS = [
    ['steps', int, 'number of epochs to train for', BY_DATASET],
    ['lr', float, 'initial learning rate for training', 0.1],
    ['weight-decay', float, 'SGD weight decay parameter', BY_DATASET],
    ['momentum', float, 'SGD momentum parameter', BY_DATASET],
    ['step-lr', int, 'number of steps between step-lr-gamma x LR drops', BY_DATASET],
    ['step-lr-gamma', float, 'multiplier by which LR drops in step scheduler', 0.1],
    ['custom-lr-multiplier', str, 'LR multiplier sched (format: [(epoch, LR),...])', None],
    ['num_samples', int, 'number of samples to sample at once from a truncated distribution', BY_DATASET],
    ['radius', float, 'initial radius size for PSGD', BY_DATASET],
    ['score', bool, 'determine convergence based off of score', BY_DATASET],
    ['lr-interpolation', ["linear", "step"], 'Drop LR as step function or linearly', "step"],
    ['tol', float, 'gradient tolerance for algorithm convergence', BY_DATASET], 
    ['clamp', bool, 'use clamp method for projection set', BY_DATASET], 
    ['shuffle', [0, 1], 'shuffle dataset running procedure', 1], 
    ['eps', float, 'fuzzy error value', 1e-5]
]

# default arguments for truncated logistic regression algorithm
LOGISTIC_ARGS = [
    ['steps', int, 'number of epochs to train for', BY_DATASET],
    ['lr', float, 'initial learning rate for training', 0.1],
    ['weight-decay', float, 'SGD weight decay parameter', BY_DATASET],
    ['momentum', float, 'SGD momentum parameter', BY_DATASET],
    ['step-lr', int, 'number of steps between step-lr-gamma x LR drops', BY_DATASET],
    ['step-lr-gamma', float, 'multiplier by which LR drops in step scheduler', 0.1],
    ['custom-lr-multiplier', str, 'LR multiplier sched (format: [(epoch, LR),...])', None],
    ['num_samples', int, 'number of samples to sample at once from a truncated distribution', BY_DATASET],
    ['radius', float, 'initial radius size for PSGD', BY_DATASET],
    ['score', bool, 'determine convergence based off of score', BY_DATASET],
    ['lr-interpolation', ["linear", "step"], 'Drop LR as step function or linearly', "step"],
    ['tol', float, 'gradient tolerance for algorithm convergence', BY_DATASET]
]

MODEL_LOADER_ARGS = [
    ['dataset', list(datasets.DATASETS.keys()), '', REQ],
    ['data', str, 'path to the dataset', '/tmp/'],
    ['arch', str, 'architecture (see {cifar,imagenet}_models/', REQ],
    ['batch-size', int, 'batch size for data loading', BY_DATASET],
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

CONFIG_ARGS = [
    ['config-path', str, 'config path for loading in parameters', None],
    ['eval-only', [0, 1], 'just run evaluation (no training)', 0],
    ['exp-name', str, 'where to save in (inside out_dir)', None]
]
"""
Arguments for main.py specifically
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
            :class:`delphi.datasets.DataSet` subclass name)
    Returns:
        args (object): The :samp:`args` object with all the defaults filled in according to :samp:`arg_list` defaults.
    """
    for arg_name, _, _, arg_default in arg_list:
        name = arg_name.replace("-", "_")
        if has_attr(args, name): continue
        if arg_default == REQ: raise ValueError(f"{arg_name} required")
        elif arg_default == BY_DATASET:
            setattr(args, name, TRAINING_DEFAULTS[ds_class][name])
        elif arg_default is not None:
            setattr(args, name, arg_default)
    return args