"""
Truncated regression with and without known noise variance
"""

import torch as ch
from torch import Tensor
import torch.nn as nn
from torch.nn import Linear
from torch.nn import init
from torch.utils.data import DataLoader
from cox.utils import Parameters
from sklearn.linear_model import LinearRegression

from ..Function import TruncatedMSE, TruncatedUnknownVarianceMSE
from ..projection_set import TruncatedRegressionProjectionSet, TruncatedRegressionUnknownVarianceProjectionSet
from ..defaults import REGRESSION_DEFAULTS, setup_args
from ..train import train_model
from ..utils.helpers import Bounds, setup_store_with_metadata, LinearUnknownVariance


def truncated_regression(args, X, y, known=False, store=None):
    """
    Truncated regression module with known and unknown noise variance.
    """
    # setup model and training procedure
    if known:
        # check all parameters
        args = setup_args(args, REGRESSION_DEFAULTS['known'])
        setattr(args, 'custom_criterion', TruncatedMSE.apply)
        trunc_reg = Linear(in_features=X.size(1), out_features=1, bias=args.bias)
        # assign emprical estimates
        lin_reg = LinearRegression(intercept_=args.bias)
        lin_reg.fit(X, y)
        trunc_reg.weight = lin_reg.coef_
        if args.bias: trunc_reg.bias = lin_reg.intercept_
        params = None
        setattr('projection_set', TruncatedRegressioinProjectionSet)
    else:
        # check all parameters
        args = setup_args(args, REGRESSION_DEFAULTS['unknown'])
        setattr(args, 'custom_criterion', TruncatedUnknownVarianceMSE.apply)
        trunc_reg = LinearUnknownVariance(in_features=X.size(0), bias=args.bias)
        # assign emprical estimates
        lin_reg = LinearRegression(intercept_=args.bias)
        lin_reg.fit(X, y)
        trunc_reg.lambda_ = ch.var(Tensor(lin_reg.predict(X)) - y, dim=0).unsqueeze(0).inverse()
        trunc_reg.v = lin_reg.coef_*trunc_reg.lambda_
        if args.bias: trunc_reg.bias = lin_reg.intercept_*trunc_reg.lambda_
        # can use different learning rate for unknown noise variance
        params = [
            {'params': trunc_reg.v},
            {'params': trunc_reg.bias},
            {'params': trunc_reg.lambda_, 'lr': args.var_lr}]
        setattr('projection_set', TruncatedUnknownVarianceProjectionSet)

    # dataset
    dataset = TensorDataset(X, y)
    S = DataLoader(dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=args.shuffle)

    if store:
        store = setup_store_with_metadata(args)

    # perform truncated regression procedure 
    return train_model(trunc_reg, (S, None), update_params=params, device=args.device)


