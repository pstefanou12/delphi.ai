"""
Default parameters for running algorithms in delphi.ai.
"""

import torch as ch
from typing import Callable, Iterable

from .helpers import has_attr


# CONSTANTS
REQ = 'required'


TRAINER_DEFAULTS = { 
    'num_trials': (int, 3),
    'epochs': (int, 20),
    'trials': (int, 3),
    'tol': (float, 1e-3),
    'early_stopping': (bool, False), 
    'n_iter_no_change': (int, 5),
    'verbose': (bool, False),
    'disable_no_grad': (bool, False), 
    'epoch_step': (bool, False),
}

DATASET_DEFAULTS = {
        'workers': (int, 1), 
        'batch_size': (int, 100), 
        'val': (float, .2), 
        'normalize': (bool, False),
}

DELPHI_DEFAULTS = { 
    'lr': (float, 1e-1), 
    'step_lr': (int, 100),
    'step_lr_gamma': (float, .9), 
    'custom_lr_multiplier': (str, None), 
    'momentum': (float, 0.0), 
    'weight_decay': (float, 0.0), 
    'device': (str, 'cpu')
}

TRUNC_REG_DEFAULTS = {
        'phi': (Callable, REQ),
        'noise_var': (ch.Tensor, None), 
        'fit_intercept': (bool, True), 
        'val': (float, .2),
        'var_lr': (float, 1e-2), 
        'l1': (float, 0.0),
        'weight_decay': (float, 0.0), 
        'eps': (float, 1e-5),
        'r': (float, 1.0), 
        'rate': (float, 1.5), 
        'batch_size': (int, 50),
        'workers': (int, 0),
        'num_samples': (int, 50),
        'shuffle': (bool, True)
}

TRUNC_LDS_DEFAULTS = {
        'phi': (Callable, REQ),
        'noise_var': (ch.Tensor, None), 
        'fit_intercept': (bool, False), 
        'val': (float, .2),
        'l1': (float, 0.0),
        'weight_decay': (float, 0.0), 
        'eps': (float, 1e-5),
        'r': (float, 1.0), 
        'rate': (float, 1.5), 
        'batch_size': (int, 50),
        'workers': (int, 0),
        'num_samples': (int, 50),
        'c_s': (float, 100.0),
        'shuffle': (bool, False), 
        'constant': (bool, True),
}

TRUNC_LOG_REG_DEFAULTS = {
        'phi': (Callable, REQ),
        'epochs': (int, 1),
        'fit_intercept': (bool, True), 
        'trials': (int, 3),
        'val': (float, .2),
        'l1': (float, 0.0), 
        'eps': (float, 1e-5),
        'r': (float, 1.0), 
        'rate': (float, 1.5), 
        'batch_size': (int, 10),
        'workers': (int, 0),
        'num_samples': (int, 10),
        'multi_class': ({'multinomial', 'ovr'}, 'ovr'),
}


TRUNC_PROB_REG_DEFAULTS = {
        'phi': (Callable, REQ), 
        'fit_intercept': (bool, True), 
        'trials': (int, 3),
        'val': (float, .2),
        'l1': (float, 0.0), 
        'eps': (float, 1e-5),
        'r': (float, 1.0), 
        'rate': (float, 1.5), 
        'batch_size': (int, 10),
        'tol': (float, 1e-3),
        'workers': (int, 0),
        'num_samples': (int, 10),
}


CENSOR_MULTI_NORM_DEFAULTS = {
        'phi': (Callable, REQ),
        'val': (float, .2),
        'eps': (float, 1e-5),
        'r': (float, 1.0), 
        'rate': (float, 1.5), 
        'batch_size': (int, 10),
        'tol': (float, 1e-1),
        'workers': (int, 0),
        'num_samples': (int, 10),
        'covariance_matrix': (ch.Tensor, None),
}


TRUNC_MULTI_NORM_DEFAULTS = {
        'val': (float, .2),
        'eps': (float, 1e-5),
        'r': (float, 1.0), 
        'rate': (float, 1.5), 
        'batch_size': (int, 10),
        'tol': (float, 1e-1),
        'workers': (int, 0),
        'num_samples': (int, 10),
        'covariance_matrix': (ch.Tensor, None), 
        'd': (int, 100),
}


TRUNC_BOOL_PROD_DEFAULTS = {
        'phi': (Callable, REQ),
        'val': (float, .2),
        'eps': (float, 1e-5),
        'r': (float, 1.0), 
        'rate': (float, 1.5), 
        'batch_size': (int, 10),
        'tol': (float, 1e-1),
        'workers': (int, 0),
        'num_samples': (int, 10),
        'step_lr_gamma': 1.0,
}

TRUNCATED_LQR_DEFAULTS =  { 
        'target_thickness': (float, float('inf')), 
        'num_traj_phase_one': (int, float('inf')),
        'num_traj_phase_two': (int, float('inf')), 
        'num_traj_gen_samples_B': (int, float('inf')), 
        'num_traj_gen_samples_A': (int, float('inf')),
        'T_phase_one': (int, float('inf')), 
        'T_phase_two': (int, float('inf')), 
        'T_gen_samples_B': (int, float('inf')),  
        'T_gen_samples_A': (int, float('inf')),
        'R': (float, REQ),
        'U_A': (float, REQ), 
        'U_B': (float, REQ), 
        'delta': (float, REQ), 
        'eps1': (float, .9), 
        'eps2': (float, .9),
        'repeat': (int, None), 
        'gamma': (float, REQ) 
}

def check_and_fill_args(args, defaults): 
        '''
        Checks args (algorithm hyperparameters) and makes sure that all required parameters are 
        given.
        '''
        # assign all of the default arguments and check that all necessary arguments are provided
        for arg_name, (arg_type, arg_default) in defaults.items():
                if has_attr(args, arg_name):
                # check to make sure that hyperparameter inputs are the same type
                        if isinstance(arg_type, Iterable):
                                if args.__getattr__(arg_name) in arg_type: continue 
                                raise ValueError('arg: {} is not correct type: {}. fix hyperparameters and run again.'.format(arg_name, arg_type))
                        if isinstance(args.__getattr__(arg_name), arg_type): continue
                        raise ValueError('arg: {} is not correct type: {}. fix hyperparameters and run again.'.format(arg_name, arg_type))
                if arg_default == REQ: 
                        raise ValueError(f"{arg_name} required")
                elif arg_default is not None: 
                        # use default arugment
                        setattr(args, arg_name, arg_default)
        
        return args

