"""
Script for running random.
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
from sklearn.metrics import r2_score
from argparse import ArgumentParser
import datetime

from delphi.stats.linear_regression import TruncatedLinearRegression
from delphi.oracle import Left
from delphi.utils import constants as consts
from delphi.utils.helpers import setup_store_with_metadata

# CONSTANTS 
TABLE_NAME = 'results'

# commands and arguments
COMMAND = 'Rscript'
PATH2SCRIPT = '/home/gridsan/stefanou/delphi/truncreg.R'
TMP_FILE = 'tmp.csv'
RESULT_FILE = 'result.csv'

# experiment argument parser
parser = ArgumentParser()
parser.add_argument('--dims', type=int, required=True, help='dimensions to run regression alg')
parser.add_argument('--bias', action='store_true', help='bias term for regression')
parser.add_argument('--out-dir', required=True, help='experiment output')
parser.add_argument('--samples', type=int, required=False, default=10000,  help='number of samples to generate for ground-truth')
parser.add_argument('--c', type=float, required=False, default=0, help='truncation parameter for experiment')
parser.add_argument('--batch_size', type=int, required=False, default=100, help='batch size for procedure')
parser.add_argument('--lr', type=float, required=False, default=1e-1, help='learning rate for weight params')
parser.add_argument('--var-lr', type=float, required=False, default=1e-2, help='learning rate for variance parameter')
parser.add_argument('--var_', type=int, required=False, default=20, help='maximum variance to run for experiment')
parser.add_argument('--trials', type=int, required=False, default=10, help='number of trials to run')
parser.add_argument('--norm', action='store_true', help='normalize gradient')


def main(args):
    # MSE Loss
    mse_loss = ch.nn.MSELoss()

    # distribution for generating ground truth
    U = Uniform(args.lower, args.upper)
    U_ = Uniform(args.x_lower, args.x_upper)

    # # set experiment manual seed 
    # ch.manual_seed(0)
    for i in range(args.trials):
        # create store and add table
        store = Store(args.out_dir)
        store.add_table(TABLE_NAME, { 
            'ols_r2': float,
            'ols_param_mse': float,
            'ols_var_mse': float,
            'known_emp_r2': float,
            'known_emp_param_mse': float,
            'known_emp_time': int,
            'known_r2': float,
            'known_param_mse': float,
            'known_time': int,
            'unknown_r2': float, 
            'unknown_param_mse': float,
            'unknown_var_mse': float,
            'unknown_time': int,
            'trunc_reg_r2': float,
            'trunc_reg_param_mse': float, 
            'trunc_reg_var_mse': float,
            'trunc_reg_time': int,
            'alpha': float, 
            'num_samples': int,
            'noise_scale': float, 
        })

        # generate ground truth
        ground_truth = ch.nn.Linear(in_features=args.dims, out_features=1, bias=args.bias)
        ground_truth.weight = ch.nn.Parameter(U.sample(ch.Size([1, args.dims]))) 
        # bias term 
        if args.bias: 
            ground_truth.bias = ch.nn.Parameter(U.sample(ch.Size([1, 1])))

        # create base classifier
        with ch.no_grad():
            # generate data
            X = U_.sample(ch.Size([args.samples, args.dims]))                # 
            y = ground_truth(X)
            # standardize input features
            X_ = (X - X.mean(0)[None,...]) / ch.sqrt(X.var(0))

        # increase variance up to 20
        for var in range(1, args.var_ + 1):
            noise_var = Tensor([var])[...,None]
            # remove synthetic data from the computation graph
            with ch.no_grad():
                # add noise to ground-truth pedictions
                noised = y + ch.sqrt(noise_var) * ch.randn(X.size(0), 1)
                # standardize noised ground truth output features
                noised_ = (y - y.mean(0)[None,...]) / ch.sqrt(y.var(0))
                # truncate based off of the standardized data
                indices = args.phi(noised_).nonzero(as_tuple=False).flatten()
                y_trunc, x_trunc = noised_[indices], X_[indices]
                alpha = Tensor([y_trunc.size(0) / args.samples])
                print("alpha: ", alpha)

            print("x trunc: ", x_trunc)
            print("y trunc: ", y_trunc)

            # standardize ground-truth parameters
            gt_ols = LinearRegression().fit(X_, noised_)
            gt_params = ch.cat([Tensor(gt_ols.coef_).T, Tensor(gt_ols.intercept_)[..., None]])
            print("gt params: ", gt_params)

            # empirical linear regression
            ols = LinearRegression() 
            ols.fit(x_trunc, y_trunc)
            ols_var = ch.var(Tensor(ols.predict(x_trunc)) - y_trunc, dim=0)[..., None]
            ols_params = ch.cat([Tensor(ols.coef_).T, Tensor(ols.intercept_)[..., None]])
            # check r2 for entire dataset
            ols_pred = ols.predict(X_)
            print("ols params: ", ols_params)

            # ols results
            store[TABLE_NAME].update_row({
                'ols_r2': r2_score(y.flatten(), ols_pred.flatten()), 
                'ols_param_mse': mse_loss(Tensor(ols_params), Tensor(gt_params)),
                'ols_var_mse': mse_loss(ols_var, noise_var), 
                })

            # truncated linear regression with known noise variance using empirical noise variance
            trunc_reg = TruncatedLinearRegression(phi=args.phi, alpha=alpha, args=args, bias=args.bias, var=ols_var)
            st = datetime.datetime.now()
            known_emp_results = trunc_reg.fit(x_trunc, y_trunc)
            known_emp_params = ch.cat([known_emp_results.weight.detach().cpu().T, known_emp_results.bias.detach().cpu()[..., None]])
            # check r2 for entire dataset
            known_emp_pred = known_emp_results(X_).detach().cpu()
            print("known emp params: ", known_emp_params)

            # known emp results
            store[TABLE_NAME].update_row({
                'known_emp_r2': r2_score(noised_.flatten(), known_emp_pred.flatten()), 
                'known_emp_param_mse': mse_loss(known_emp_params, gt_params),
                'known_emp_time': int((datetime.datetime.now() - st).total_seconds()), 
                })

            # truncated linear regression with known noise variance using actual noise variance
            trunc_reg = TruncatedLinearRegression(phi=args.phi, alpha=alpha, args=args, bias=args.bias, var=Tensor([var])[...,None])
            st = datetime.datetime.now()
            known_results = trunc_reg.fit(x_trunc, y_trunc)
            known_params = ch.cat([known_results.weight.detach().cpu().T, known_results.bias.detach().cpu()[..., None]])
            known_time = int((datetime.datetime.now() - st).total_seconds())
            # check r2 for entire dataset
            known_pred = known_results(X_).detach().cpu()

            print("known params: ", known_params)

            # known results
            store[TABLE_NAME].update_row({
                'known_r2': r2_score(noised_.flatten(), known_pred.flatten()), 
                'known_param_mse': mse_loss(known_params, gt_params),
                'known_time': int((datetime.datetime.now() - st).total_seconds()), 
                })

            # truncated linear regression with unknown noise variance
            trunc_reg = TruncatedLinearRegression(phi=args.phi, alpha=alpha, args=args, bias=args.bias)
            st = datetime.datetime.now()
            unknown_results = trunc_reg.fit(x_trunc, y_trunc)
            var_ = unknown_results.lambda_.inverse().detach()
            unknown_params = ch.cat([(unknown_results.weight.detach() * var_).cpu().T, (unknown_results.bias.detach() * var_).cpu()])
            # check r2 for entire dataset
            unknown_pred = unknown_results(X_).detach().cpu()

            print("unknown params", unknown_params)
            print("var_: ", var_)

            # unknown results
            store[TABLE_NAME].update_row({
                'unknown_r2': r2_score(noised_.flatten(), unknown_pred.flatten()), 
                'unknown_param_mse': mse_loss(unknown_params, gt_params),
                'unknown_var_mse': mse_loss(var_, noise_var),
                'unknown_time': int((datetime.datetime.now() - st).total_seconds()), 
                })

            # spawn subprocess to run truncreg experiment
            concat = ch.cat([X_, noised_], dim=1).numpy()
            """
            DATA FORMAT:
                -First n-1 columns are independent variables
                -nth column is dependent variable
            """
            concat_df = pd.DataFrame(concat)
            concat_df.to_csv(args.out_dir + '/' + TMP_FILE) # save data to csv
            """
            Arguments
            - c - truncation point (float)
            - dir - left or right -> type of truncation (str)
            """
            cmd = [COMMAND, PATH2SCRIPT] + [str(args.C), str(args.dims), 'left', args.out_dir]

            # check_output will run the command and store the result
            st = datetime.datetime.now()
            result = subprocess.check_output(cmd, universal_newlines=True)
            trunc_res = Tensor(pd.read_csv(args.out_dir + '/' + RESULT_FILE)['x'].to_numpy())
            trunc_reg_params = ch.cat([trunc_res[1:-1].flatten(), trunc_res[0][None,...]])[..., None]

            print("trunc reg params: ", trunc_reg_params)
            print("first term: ", trunc_reg_params[:-1][None,...].mm(X_))
            print("second term: ", trunc_reg_params[-1][None,...])
            trunc_reg_pred = trunc_reg_params[:-1][None,...].mm(X_) + trunc_reg_params[-1][None,...]

            # truncreg results
            store[TABLE_NAME].update_row({
                'trunc_reg_r2': r2_score(noised_.flatten(), trunc_reg_pred.flatten()), 
                'trunc_reg_param_mse': mse_loss(trunc_reg_params, gt_params),
                'trunc_reg_var_mse': mse_loss(trunc_res[-1].pow(2)[None,...], noise_var),
                'trunc_reg_time': int((datetime.datetime.now() - st).total_seconds()), 
                })

            # add additional metadata to store
            store[TABLE_NAME].update_row({ 
                'alpha': float(alpha.flatten()),
                'num_samples': x_trunc.size(0),
                'noise_scale': float(var), 
            })

            # append row to table
            store[TABLE_NAME].flush_row()

        # close current store
        store.close()

if __name__ == '__main__': 
    # set environment variable so that stores can create output files
    os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

    args = Parameters(parser.parse_args().__dict__)
    args.__setattr__('workers', 8)
    # args.__setattr__('custom_lr_multiplier', consts.COSINE)
    args.__setattr__('step_lr', 100000000)
    args.__setattr__('step_lr_gamma', 1.0)
    # independent variable bounds
    args.__setattr__('x_lower', -100)
    args.__setattr__('x_upper', 100)
    # parameter bounds
    args.__setattr__('lower', -1)
    args.__setattr__('upper', 1)
    args.__setattr__('device', 'cuda' if ch.cuda.is_available() else 'cpu')
    # normalize gradient
    args.__setattr__('norm', False)

    # setup store with metadata
    store = Store(args.out_dir)
    setup_store_with_metadata(args, store)
    store.close()

    print('args: ', args)

    args.__setattr__('phi', Left(Tensor([args.C])))

    # run experiment
    main(args)
