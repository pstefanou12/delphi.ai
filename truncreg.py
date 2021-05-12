"""
Script for running regression batch job experiments, so that they can be compared against truncreg.
"""

import subprocess
import torch as ch
from torch import Tensor
from torch.distributions import Uniform
import pandas as pd
import numpy as np
import csv
import json
from cox.utils import Parameters
from cox.store import Store
import os
from sklearn.linear_model import LinearRegression

from delphi.stats.linear_regression import TruncatedLinearRegression
from delphi.oracle import Left
from delphi.utils import constants as consts

# SETUP ENVIRONMENT VARIABLES TO ENABLE WRITING TO STORES
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
# CONSTANTS 
STORE_PATH = '/home/gridsan/stefanou/Regression/TruncReg'
STORE_TABLE_NAME = 'results'
# commands and arguments
COMMAND = 'Rscript'
PATH2SCRIPT = '/home/gridsan/stefanou/delphi/truncreg.R'
TMP_FILE = '/home/gridsan/stefanou/truncreg/tmp.csv'
RESULT_FILE = '/home/gridsan/stefanou/truncreg/result.csv'
# ARGUMENTS FOR TRUNCREG
ARGS = ['0.0', 'left']

# ORACLE
LEFT = Left(Tensor([0.0]))

# procedure hyperparameters
args = Parameters({ 
    'workers': 8, 
    'batch_size': 100,
    'bias': True,
    'num_samples': 1000,
    'samples': 10000, 
    'in_features': 1, 
    'clamp': True, 
    'radius': 5.0, 
    'var_lr': 1e-2,
    'lr': 1e-2,
    'shuffle': False, 
    'eps': 1e-5, 
    'custom_lr_multiplier': consts.COSINE,
    'trials': 10,
    'tol': 1e-2,
    'lower': -1, 
    'upper': 1,
    'x_lower': -5, 
    'x_upper': 5,
    'device': 'cuda' if ch.cuda.is_available() else 'cpu',
    'phi': LEFT, 
    'trials': 10,
})

# distribution for generating ground truth
U = Uniform(args.lower, args.upper)
U_ = Uniform(args.x_lower, args.x_upper)


for i in range(args.trials):
    # create store and add table
    store = Store(STORE_PATH)

    store.add_table(STORE_TABLE_NAME, { 
        'known_param_mse': float,
        'unknown_param_mse': float,
        'unknown_var_mse': float,
        'ols_param_mse': float,
        'ols_var_mse': float,
        'trunc_reg_param_mse': float, 
        'trunc_var_mse': float,
        'alpha': float, 
        'var': float, 
    })

    # increase variance up to 20
    for var in range(1, 21):
        
        ch.manual_seed(1)
        # generate ground truth
        ground_truth = ch.nn.Linear(in_features=args.in_features, out_features=1, bias=args.bias)
        ground_truth.weight = ch.nn.Parameter(U.sample(ch.Size([1, args.in_features]))) 
        ground_truth.bias = ch.nn.Parameter(U.sample(ch.Size([1, 1])))

        # remove synthetic data from the computation graph
        with ch.no_grad():
            # generate data
            X = U_.sample(ch.Size([args.samples, args.in_features]))
            y = ground_truth(X) + ch.sqrt(Tensor([var])) * ch.randn(X.size(0), 1)
            # truncate
            indices = args.phi(y).nonzero(as_tuple=False).flatten()
            y_trunc, x_trunc = y[indices], X[indices]
            alpha = Tensor([y_trunc.size(0) / args.samples])

        # empirical linear regression
        ols = LinearRegression() 
        ols.fit(x_trunc, y_trunc)
        ols_var = ch.var(Tensor(ols.predict(x_trunc)) - y_trunc, dim=0)[..., None]

        # truncated linear regression with known noise variance
        trunc_reg = TruncatedLinearRegression(phi=args.phi, alpha=alpha, args=args, bias=args.bias, var=ols_var)
        results = trunc_reg.fit(x_trunc, y_trunc)
        w_, w0_ = results.weight.detach().cpu(), results.bias.detach().cpu()

        # truncated linear regression with unknown noise variance
        trunc_reg = TruncatedLinearRegression(phi=args.phi, alpha=alpha, args=args, bias=args.bias)
        results = trunc_reg.fit(x_trunc, y_trunc)
        var_ = results.lambda_.inverse().detach()
        w, w0 = (results.v.detach()*var_).cpu(), (results.bias.detach()*var_).cpu()

        # spawn subprocess to run truncreg experiment
        concat = ch.cat([x_trunc, y_trunc], dim=1).numpy()
        """
        DATA FORMAT:
            -First n-1 columns are independent variables
            -nth column is dependent variable
        """
        concat_df = pd.DataFrame(concat)
        concat_df.to_csv(TMP_FILE) # save data to csv
        """
        Arguments
        - c - truncation point (float)
        - dir - left or right -> type of truncation (str)
        """
        cmd = [COMMAND, PATH2SCRIPT] + ARGS

        # check_output will run the command and store the result
        result = subprocess.check_output(cmd, universal_newlines=True)
        trunc_res = Tensor(pd.read_csv(RESULT_FILE)['x'].to_numpy())[None,...]

        # parameter estimates 
        known_params = ch.cat([w_.T, w0_[..., None]])
        real_params = ch.cat([ground_truth.weight.T, ground_truth.bias])
        ols_params = ch.cat([Tensor(ols.coef_).T, Tensor(ols.intercept_)[..., None]])
        unknown_params = ch.cat([w, w0])
        trunc_reg_params = ch.cat([trunc_res[1:-1].flatten(), trunc_res[0]])

        # metrics
        known_param_mse = ch.nn.MSELoss()(known_params, real_params)
        unknown_param_mse = ch.nn.MSELoss()(unknown_params, real_params)
        unknown_var_mse = ch.nn.MSELoss()(var_, Tensor([var]))
        ols_param_mse = ch.nn.MSELoss()(Tensor(ols_params), Tensor(real_params))
        ols_var_mse = ch.nn.MSELoss()(ols_var, Tensor([var]))
        trunc_reg_param_mse = ch.nn.MSELoss()(trunc_reg_params, real_params)
        trunc_var_mse = ch.nn.MSELoss()(trunc_reg_params[-1], Tensor([var]))

        # add results to store
        store[STORE_TABLE_NAME].append_row({ 
            'known_param_mse': known_param_mse,
            'unknown_param_mse': unknown_param_mse,
            'unknown_var_mse': unknown_var_mse,
            'ols_param_mse': ols_param_mse,
            'ols_var_mse': ols_var_mse,
            'trunc_reg_param_mse': trunc_reg_param_mse, 
            'trunc_var_mse': trunc_var_mse,
            'alpha': float(alpha.flatten()),
            'var': float(var), 
        })

    # close current store
    store.close()
