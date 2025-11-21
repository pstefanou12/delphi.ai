"""
Test suite for truncated logistic regression.
Includes: 
    -Truncated logistic regression 
    -Truncated multinomial logistic regression
"""

import torch as ch
from torch.distributions import Uniform, Gumbel
from torch.nn import CosineSimilarity
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from statsmodels.discrete.discrete_model import Probit

from delphi.utils.helpers import Parameters, logistic
from delphi import oracle
from delphi.utils.datasets import make_train_and_val
from delphi.stats.truncated_logistic_regression import TruncatedLogisticRegression
from delphi.stats.truncated_probit_regression import TruncatedProbitRegression

cos_sim = CosineSimilarity()
G = Gumbel(0, 1)
seed = 69 

# left truncated binary classification task
def test_truncated_one_dimension_logistic_regression_no_intercept():
    d, k = 1, 1
    C = -1.0
    # ground-truth logistic regression model 
    w = Uniform(-1, 1)
    U = Uniform(-5, 5)
    W = w.sample([k, d])
    
    # input features
    SAMPLES = 10000
    X = U.sample([SAMPLES, d])
    # latent variables
    z = X@W.T + logistic.sample([X.size(0), 1])
    # classification
    y = (z > 0).float()

    # generate ground-truth data
    phi = oracle.Left_Regression(C)
    indices = phi(z).nonzero(as_tuple=True)
    x_trunc = X[indices[0]]
    y_trunc = y[indices][...,None]
    alpha = x_trunc.size(0) / X.size(0)
    print(f'C: {C}')
    print(f'alpha: {alpha}')
                
    sklearn = LogisticRegression(penalty=None, fit_intercept=False)
    sklearn.fit(X, y.flatten())
    sklearn_ = ch.from_numpy(sklearn.coef_.flatten())
    print(f'sklearn: {sklearn_}')
    pred = sklearn.predict(X)
    acc = np.equal(pred, y.flatten()).sum() / len(y)
    print(f'sklearn acc: {acc}')
    sklearn_conf_matrix = confusion_matrix(y, pred)
    print(f'sklearn confusion matrix: \n {sklearn_conf_matrix}')
        
    trunc_sklearn = LogisticRegression(penalty=None, fit_intercept=False)
    trunc_sklearn.fit(x_trunc, y_trunc.flatten())
    trunc_sklearn_ = ch.from_numpy(trunc_sklearn.coef_.flatten())
    print(f'trunc sklearn: {trunc_sklearn_}')
    trunc_sklearn_cos_sim = float(cos_sim(trunc_sklearn_[None,...], sklearn_[None,...]))
    pred = trunc_sklearn.predict(X)
    trunc_sklearn_acc = np.equal(pred, y.flatten()).sum() / len(y)
    print(f'trunc sklearn acc: {trunc_sklearn_acc}')
    print(f'trunc sklearn cos sim: {trunc_sklearn_cos_sim}')
    trunc_sklearn_conf_matrix = confusion_matrix(y, pred)
    print(f'trunc sklearn confusion matrix: \n {trunc_sklearn_conf_matrix}')
                        
    args = Parameters({
                        'normalize': False, 
                        'batch_size': 10,
                        'epochs': 10,
                        'trials': 1,
                        'verbose': True,
                        'early_stopping': True})
    ch.manual_seed(seed)
    trunc_log_reg = TruncatedLogisticRegression(args, 
                                                phi, 
                                                alpha, 
                                                fit_intercept=False)
    trunc_log_reg.fit(x_trunc, y_trunc) 
    
    delphi_log_reg_ = trunc_log_reg.coef_.flatten()
    print(f'delphi log reg: {delphi_log_reg_}')
    delphi_log_reg_pred = trunc_log_reg.predict(X)
    delphi_log_reg_acc = delphi_log_reg_pred.eq(y).sum() / len(y)
    print(f'delphi log reg accuracy: {delphi_log_reg_acc}')
    delphi_log_reg_conf_matrix = confusion_matrix(y, delphi_log_reg_pred)
    print(f'delphi log reg confusion matrix: \n {delphi_log_reg_conf_matrix}')

    delphi_log_reg_cos_sim = float(cos_sim(delphi_log_reg_[None,...], sklearn_[None,...]))
    print(f'delphi cos sim: {delphi_log_reg_cos_sim}')
    assert delphi_log_reg_cos_sim > 9e-1, f"trunc log reg cosine similarity is {delphi_log_reg_cos_sim}"

# left truncated binary classification task
def test_truncated_one_dimension_logistic_regression():
    d, k = 1, 1
    C = -1.0
    # ground-truth logistic regression model 
    w = Uniform(-1, 1)
    U = Uniform(-5, 5)
    W = w.sample([k, d+1])
    
    # input features
    SAMPLES = 10000
    X = U.sample([SAMPLES, d])
    X_ones = ch.cat([X, ch.ones([SAMPLES, 1])], dim=1)
    # latent variables
    z = X_ones@W.T + logistic.sample([X.size(0), 1])
    # classification
    y = (z > 0).float()

    # generate ground-truth data
    phi = oracle.Left_Regression(C)
    indices = phi(z).nonzero(as_tuple=True)
    x_trunc = X[indices[0]]
    y_trunc = y[indices][...,None]
    alpha = x_trunc.size(0) / X.size(0)
    print(f'C: {C}')
    print(f'alpha: {alpha}')
                
    sklearn = LogisticRegression(penalty=None, fit_intercept=True)
    sklearn.fit(X, y.flatten())
    sklearn_ = ch.from_numpy(np.concatenate([sklearn.coef_.flatten(), sklearn.intercept_]))
    print(f'sklearn: {sklearn_}')
    pred = sklearn.predict(X)
    acc = np.equal(pred, y.flatten()).sum() / len(y)
    print(f'sklearn acc: {acc}')
    sklearn_conf_matrix = confusion_matrix(y, pred)
    print(f'sklearn confusion matrix: \n {sklearn_conf_matrix}')
        
    trunc_sklearn = LogisticRegression(penalty=None, fit_intercept=True)
    trunc_sklearn.fit(x_trunc, y_trunc.flatten())
    trunc_sklearn_ = ch.from_numpy(np.concatenate([trunc_sklearn.coef_.flatten(), trunc_sklearn.intercept_]))
    print(f'trunc sklearn: {trunc_sklearn_}')
    trunc_sklearn_cos_sim = float(cos_sim(trunc_sklearn_[None,...], sklearn_[None,...]))
    pred = trunc_sklearn.predict(X)
    trunc_sklearn_acc = np.equal(pred, y.flatten()).sum() / len(y)
    print(f'trunc sklearn acc: {trunc_sklearn_acc}')
    print(f'trunc sklearn cos sim: {trunc_sklearn_cos_sim}')
    trunc_sklearn_conf_matrix = confusion_matrix(y, pred)
    print(f'trunc sklearn confusion matrix: \n {trunc_sklearn_conf_matrix}')
                        
    args = Parameters({
                        'normalize': False, 
                        'batch_size': 10,
                        'epochs': 10,
                        'trials': 1,
                        'verbose': True,
                        'early_stopping': True})
    ch.manual_seed(seed)
    trunc_log_reg = TruncatedLogisticRegression(args, 
                                                phi, 
                                                alpha)
    trunc_log_reg.fit(x_trunc, y_trunc) 
    
    delphi_log_reg_ = ch.cat([trunc_log_reg.coef_.flatten(), trunc_log_reg.intercept_])
    print(f'delphi log reg: {delphi_log_reg_}')
    delphi_log_reg_pred = trunc_log_reg.predict(X)
    delphi_log_reg_acc = delphi_log_reg_pred.eq(y).sum() / len(y)
    print(f'delphi log reg accuracy: {delphi_log_reg_acc}')
    delphi_log_reg_conf_matrix = confusion_matrix(y, delphi_log_reg_pred)
    print(f'delphi log reg confusion matrix: \n {delphi_log_reg_conf_matrix}')

    delphi_log_reg_cos_sim = float(cos_sim(delphi_log_reg_[None,...], sklearn_[None,...]))
    print(f'delphi cos sim: {delphi_log_reg_cos_sim}')
    assert delphi_log_reg_cos_sim > 9e-1, f"trunc log reg cosine similarity is {delphi_log_reg_cos_sim}"

# left truncated binary classification task
def test_truncated_ten_dimension_logistic_regression():
    d, k = 10, 1
    C = -.5
    # ground-truth logistic regression model 
    w = Uniform(-1, 1)
    U = Uniform(-5, 5)
    W = w.sample([k, d+1])
    
    # input features
    SAMPLES = 10000
    X = U.sample([SAMPLES, d])
    X_ones = ch.cat([X, ch.ones([SAMPLES, 1])], dim=1)
    # latent variables
    z = X_ones@W.T + logistic.sample([X.size(0), 1])
    # classification
    y = (z > 0).float()

    # generate ground-truth data
    phi = oracle.Left_Regression(C)
    indices = phi(z).nonzero(as_tuple=True)
    x_trunc = X[indices[0]]
    y_trunc = y[indices][...,None]
    alpha = x_trunc.size(0) / X.size(0)
    print(f'C: {C}')
    print(f'alpha: {alpha}')
                
    sklearn = LogisticRegression(penalty=None, fit_intercept=True)
    sklearn.fit(X, y.flatten())
    sklearn_ = ch.from_numpy(np.concatenate([sklearn.coef_.flatten(), sklearn.intercept_]))
    print(f'sklearn: {sklearn_}')
    pred = sklearn.predict(X)
    acc = np.equal(pred, y.flatten()).sum() / len(y)
    print(f'sklearn acc: {acc}')
    sklearn_conf_matrix = confusion_matrix(y, pred)
    print(f'sklearn confusion matrix: \n {sklearn_conf_matrix}')
        
    trunc_sklearn = LogisticRegression(penalty=None, fit_intercept=True)
    trunc_sklearn.fit(x_trunc, y_trunc.flatten())
    trunc_sklearn_ = ch.from_numpy(np.concatenate([trunc_sklearn.coef_.flatten(), trunc_sklearn.intercept_]))
    print(f'trunc sklearn: {trunc_sklearn_}')
    trunc_sklearn_cos_sim = float(cos_sim(trunc_sklearn_[None,...], sklearn_[None,...]))
    pred = trunc_sklearn.predict(X)
    trunc_sklearn_acc = np.equal(pred, y.flatten()).sum() / len(y)
    print(f'trunc sklearn acc: {trunc_sklearn_acc}')
    print(f'trunc sklearn cos sim: {trunc_sklearn_cos_sim}')
    trunc_sklearn_conf_matrix = confusion_matrix(y, pred)
    print(f'trunc sklearn confusion matrix: \n {trunc_sklearn_conf_matrix}')
                        
    args = Parameters({
                        'normalize': False, 
                        'batch_size': 10,
                        'epochs': 10,
                        'trials': 1,
                        'verbose': True,
                        'early_stopping': True})
    ch.manual_seed(seed)
    trunc_log_reg = TruncatedLogisticRegression(args, 
                                                phi, 
                                                alpha)
    trunc_log_reg.fit(x_trunc, y_trunc) 
    
    delphi_log_reg_ = ch.cat([trunc_log_reg.coef_.flatten(), trunc_log_reg.intercept_])
    print(f'delphi log reg: {delphi_log_reg_}')
    delphi_log_reg_pred = trunc_log_reg.predict(X)
    delphi_log_reg_acc = delphi_log_reg_pred.eq(y).sum() / len(y)
    print(f'delphi log reg accuracy: {delphi_log_reg_acc}')
    delphi_log_reg_conf_matrix = confusion_matrix(y, delphi_log_reg_pred)
    print(f'delphi log reg confusion matrix: \n {delphi_log_reg_conf_matrix}')

    delphi_log_reg_cos_sim = float(cos_sim(delphi_log_reg_[None,...], sklearn_[None,...]))
    print(f'delphi cos sim: {delphi_log_reg_cos_sim}')
    assert delphi_log_reg_cos_sim > 9e-1, f"trunc log reg cosine similarity is {delphi_log_reg_cos_sim}"

# left truncated binary classification task with no intercept
def test_truncated_one_dimension_probit_regression_no_intercept():
    d, k = 1, 1
    C = -.25
    # ground-truth logistic regression model 
    w = Uniform(-1, 1)
    U = Uniform(-5, 5)
    W = w.sample([k, d])
    
    # input features
    SAMPLES = 10000
    X = U.sample([SAMPLES, d])
    # latent variables
    z = X@W.T + ch.randn([X.size(0), 1])
    # classification
    y = (z > 0).float()

    # generate ground-truth data
    phi = oracle.Left_Regression(C)
    indices = phi(z).nonzero(as_tuple=True)
    x_trunc = X[indices[0]]
    y_trunc = y[indices][...,None]
    alpha = x_trunc.size(0) / X.size(0)
    print(f'C: {C}')
    print(f'alpha: {alpha}')
                
    probit = Probit(y.numpy(), X.numpy())
    probit_results = probit.fit()
    probit_ = ch.from_numpy(probit_results.params) 
    print(f'probit: {probit_}')
    pred = probit_results.predict(X) > .5
    acc = np.equal(pred, y.flatten()).sum() / len(y)
    print(f'probit acc: {acc}')
    probit_conf_matrix = confusion_matrix(y, pred)
    print(f'probit confusion matrix: \n {probit_conf_matrix}')
        
    trunc_probit = Probit(y_trunc.numpy(), x_trunc.numpy()) 
    trunc_probit_results = trunc_probit.fit()
    trunc_probit_ = ch.from_numpy(trunc_probit_results.params)
    print(f'trunc probit: {trunc_probit_}')
    trunc_probit_cos_sim = float(cos_sim(trunc_probit_[None,...], probit_[None,...]))
    pred = trunc_probit_results.predict(X) > .5
    trunc_probit_acc = np.equal(pred, y.flatten()).sum() / len(y)
    print(f'trunc probit acc: {trunc_probit_acc}')
    print(f'trunc probit cos sim: {trunc_probit_cos_sim}')
    trunc_probit_conf_matrix = confusion_matrix(y, pred)
    print(f'trunc probit confusion matrix: \n {trunc_probit_conf_matrix}')
                        
    args = Parameters({
                        'normalize': False, 
                        'batch_size': 10,
                        'epochs': 10,
                        'trials': 1,
                        'verbose': True,
                        'early_stopping': True})
    ch.manual_seed(seed)
    trunc_prob_reg = TruncatedProbitRegression(args, 
                                                phi, 
                                                alpha, 
                                                fit_intercept=False)
    trunc_prob_reg.fit(x_trunc, y_trunc) 
    
    delphi_prob_reg_ = trunc_prob_reg.coef_.flatten()
    print(f'delphi prob reg: {delphi_prob_reg_}')
    delphi_prob_reg_pred = trunc_prob_reg.predict(X)
    delphi_prob_reg_acc = delphi_prob_reg_pred.eq(y).sum() / len(y)
    print(f'delphi prob reg accuracy: {delphi_prob_reg_acc}')
    delphi_prob_reg_conf_matrix = confusion_matrix(y, delphi_prob_reg_pred)
    print(f'delphi prob reg confusion matrix: \n {delphi_prob_reg_conf_matrix}')

    delphi_prob_reg_cos_sim = float(cos_sim(delphi_prob_reg_[None,...], probit_[None,...]))
    print(f'delphi cos sim: {delphi_prob_reg_cos_sim}')
    assert delphi_prob_reg_cos_sim > 9e-1, f"trunc prob reg cosine similarity is {delphi_prob_reg_cos_sim}"

# left truncated binary classification task 
def test_truncated_one_dimension_probit_regression():
    d, k = 1, 1
    C = -.25
    # ground-truth logistic regression model 
    w = Uniform(-1, 1)
    U = Uniform(-5, 5)
    W = w.sample([k, d+1])
    
    # input features
    SAMPLES = 10000
    X = U.sample([SAMPLES, d])
    X_ones = ch.cat([X, ch.ones([SAMPLES, 1])], dim=1)
    # latent variables
    z = X_ones@W.T + ch.randn([X.size(0), 1])
    # classification
    y = (z > 0).float()

    # generate ground-truth data
    phi = oracle.Left_Regression(C)
    indices = phi(z).nonzero(as_tuple=True)
    x_trunc = X[indices[0]]
    x_trunc_ones = X_ones[indices[0]]
    y_trunc = y[indices][...,None]
    alpha = x_trunc.size(0) / X.size(0)
    print(f'C: {C}')
    print(f'alpha: {alpha}')
                
    probit = Probit(y.numpy(), X_ones.numpy())
    probit_results = probit.fit()
    probit_ = ch.from_numpy(probit_results.params) 
    print(f'probit: {probit_}')
    pred = probit_results.predict(X_ones) > .5
    acc = np.equal(pred, y.flatten()).sum() / len(y)
    print(f'probit acc: {acc}')
    probit_conf_matrix = confusion_matrix(y, pred)
    print(f'probit confusion matrix: \n {probit_conf_matrix}')
        
    trunc_probit = Probit(y_trunc.numpy(), x_trunc_ones.numpy()) 
    trunc_probit_results = trunc_probit.fit()
    trunc_probit_ = ch.from_numpy(trunc_probit_results.params)
    print(f'trunc probit: {trunc_probit_}')
    trunc_probit_cos_sim = float(cos_sim(trunc_probit_[None,...], probit_[None,...]))
    pred = trunc_probit_results.predict(X_ones) > .5
    trunc_probit_acc = np.equal(pred, y.flatten()).sum() / len(y)
    print(f'trunc probit acc: {trunc_probit_acc}')
    print(f'trunc probit cos sim: {trunc_probit_cos_sim}')
    trunc_probit_conf_matrix = confusion_matrix(y, pred)
    print(f'trunc probit confusion matrix: \n {trunc_probit_conf_matrix}')
                        
    args = Parameters({
                        'normalize': False, 
                        'batch_size': 10,
                        'epochs': 10,
                        'trials': 1,
                        'verbose': True,
                        'early_stopping': True})
    ch.manual_seed(seed)
    trunc_prob_reg = TruncatedProbitRegression(args, 
                                                phi, 
                                                alpha)
    trunc_prob_reg.fit(x_trunc, y_trunc) 
    
    delphi_prob_reg_ = ch.cat([trunc_prob_reg.coef_.flatten(), trunc_prob_reg.intercept_])
    print(f'delphi prob reg: {delphi_prob_reg_}')
    delphi_prob_reg_pred = trunc_prob_reg.predict(X)
    delphi_prob_reg_acc = delphi_prob_reg_pred.eq(y).sum() / len(y)
    print(f'delphi prob reg accuracy: {delphi_prob_reg_acc}')
    delphi_prob_reg_conf_matrix = confusion_matrix(y, delphi_prob_reg_pred)
    print(f'delphi prob reg confusion matrix: \n {delphi_prob_reg_conf_matrix}')
    delphi_prob_reg_cos_sim = float(cos_sim(delphi_prob_reg_[None,...], probit_[None,...]))
    print(f'delphi cos sim: {delphi_prob_reg_cos_sim}')
    assert delphi_prob_reg_cos_sim > 9e-1, f"trunc prob reg cosine similarity is {delphi_prob_reg_cos_sim}"

# left truncated binary classification task 
def test_truncated_ten_dimension_probit_regression():
    d, k = 10, 1
    C = -.25
    # ground-truth logistic regression model 
    w = Uniform(-1, 1)
    U = Uniform(-5, 5)
    W = w.sample([k, d+1])
    
    # input features
    SAMPLES = 10000
    X = U.sample([SAMPLES, d])
    X_ones = ch.cat([X, ch.ones([SAMPLES, 1])], dim=1)
    # latent variables
    z = X_ones@W.T + ch.randn([X.size(0), 1])
    # classification
    y = (z > 0).float()

    # generate ground-truth data
    phi = oracle.Left_Regression(C)
    indices = phi(z).nonzero(as_tuple=True)
    x_trunc = X[indices[0]]
    x_trunc_ones = X_ones[indices[0]]
    y_trunc = y[indices][...,None]
    alpha = x_trunc.size(0) / X.size(0)
    print(f'C: {C}')
    print(f'alpha: {alpha}')
                
    probit = Probit(y.numpy(), X_ones.numpy())
    probit_results = probit.fit()
    probit_ = ch.from_numpy(probit_results.params) 
    print(f'probit: {probit_}')
    pred = probit_results.predict(X_ones) > .5
    acc = np.equal(pred, y.flatten()).sum() / len(y)
    print(f'probit acc: {acc}')
    probit_conf_matrix = confusion_matrix(y, pred)
    print(f'probit confusion matrix: \n {probit_conf_matrix}')
        
    trunc_probit = Probit(y_trunc.numpy(), x_trunc_ones.numpy()) 
    trunc_probit_results = trunc_probit.fit()
    trunc_probit_ = ch.from_numpy(trunc_probit_results.params)
    print(f'trunc probit: {trunc_probit_}')
    trunc_probit_cos_sim = float(cos_sim(trunc_probit_[None,...], probit_[None,...]))
    pred = trunc_probit_results.predict(X_ones) > .5
    trunc_probit_acc = np.equal(pred, y.flatten()).sum() / len(y)
    print(f'trunc probit acc: {trunc_probit_acc}')
    print(f'trunc probit cos sim: {trunc_probit_cos_sim}')
    trunc_probit_conf_matrix = confusion_matrix(y, pred)
    print(f'trunc probit confusion matrix: \n {trunc_probit_conf_matrix}')
                        
    args = Parameters({
                        'normalize': False, 
                        'batch_size': 10,
                        'epochs': 10,
                        'trials': 1,
                        'verbose': True,
                        'early_stopping': True})
    ch.manual_seed(seed)
    trunc_prob_reg = TruncatedProbitRegression(args, 
                                                phi, 
                                                alpha)
    trunc_prob_reg.fit(x_trunc, y_trunc) 
    
    delphi_prob_reg_ = ch.cat([trunc_prob_reg.coef_.flatten(), trunc_prob_reg.intercept_])
    print(f'delphi prob reg: {delphi_prob_reg_}')
    delphi_prob_reg_pred = trunc_prob_reg.predict(X)
    delphi_prob_reg_acc = delphi_prob_reg_pred.eq(y).sum() / len(y)
    print(f'delphi prob reg accuracy: {delphi_prob_reg_acc}')
    delphi_prob_reg_conf_matrix = confusion_matrix(y, delphi_prob_reg_pred)
    print(f'delphi prob reg confusion matrix: \n {delphi_prob_reg_conf_matrix}')
    delphi_prob_reg_cos_sim = float(cos_sim(delphi_prob_reg_[None,...], probit_[None,...]))
    print(f'delphi cos sim: {delphi_prob_reg_cos_sim}')
    assert delphi_prob_reg_cos_sim > 9e-1, f"trunc prob reg cosine similarity is {delphi_prob_reg_cos_sim}"

# gumbel ce classification task
def test_gumbel_ce(self):     
    d, k = 20, 10
    # ground-truth logistic regression model 
    gt = ch.nn.Linear(in_features=d, out_features=k, bias=True)
    W = Uniform(-1, 1)
    U = Uniform(-5, 5)
    gt.weight = ch.nn.Parameter(W.sample([k, d]))
    gt.bias = ch.nn.Parameter(W.sample([1, k]))

    # input features
    X = U.sample([10000, d])
    # latent variables
    stacked = gt(X).repeat(100, 1, 1)
    noised = stacked + G.sample(stacked.size())
    # classification
    y = noised.mean(0).argmax(-1).long()

    sklearn = LogisticRegression(penalty='none', fit_intercept=True)
    sklearn.fit(X, y.flatten())
    sklearn_ = ch.from_numpy(np.concatenate([sklearn.coef_, sklearn.intercept_[...,None]], axis=1))
    print(f'sklearn: {sklearn_}')
    pred = sklearn.predict(X)
    acc = np.equal(pred, y.flatten()).sum() / len(y)
    print(f'sklearn acc: {acc}')
    sklearn_conf_matrix = confusion_matrix(y, pred)
    print(f'sklearn confusion matrix: \n {sklearn_conf_matrix}')

    train_kwargs = Parameters({
                        'batch_size': 100,
                        'momentum': 0.5,
                        'epochs': 30,
                        'trials': 3,
                        'verbose': True,
                        'early_stopping': True,
                        'workers': 0,
                        'num_samples': 100})        
    ch.manual_seed(seed)
    X_ones = ch.cat([X, ch.ones(X.size(0), 1)], axis=1)
    gumbel_model = stats.GumbelCEModel(train_kwargs, X_ones.size(1), len(y.unique()))
    trainer = Trainer(gumbel_model, train_kwargs)
    train_loader, val_loader = make_train_and_val(train_kwargs, X_ones, y)
    trainer.train_model((train_loader, val_loader))
    gumbel_ = gumbel_model.model                
    gumbel_cos_sim = float(cos_sim(gumbel_.flatten()[None,...], sklearn_.flatten()[None,...]))
    gumbel_pred = gumbel_model.predict(X_ones)
    gumbel_acc = gumbel_pred.eq(y).sum() / len(y)
    print(f'gumbel accuracy: {gumbel_acc}')
    print(f'gumbel cos sim: {gumbel_cos_sim}')
    gumbel_conf_matrix = confusion_matrix(y[...,None], gumbel_pred)
    print(f'trunc gumbel confusion matrix: \n {gumbel_conf_matrix}')

# softmax classification task
def test_softmax(self): 
    d, k = 20, 10
    # ground-truth logistic regression model 
    gt = ch.nn.Linear(in_features=d, out_features=k, bias=True)
    W = Uniform(-1, 1)
    U = Uniform(-5, 5)
    gt.weight = ch.nn.Parameter(W.sample([k, d]))
    gt.bias = ch.nn.Parameter(W.sample([1, k]))

    # input features
    X = U.sample([10000, d])
    # latent variables
    stacked = gt(X).repeat(100, 1, 1)
    noised = stacked + G.sample(stacked.size())
    # classification
    y = noised.mean(0).argmax(-1).long()

    sklearn = LogisticRegression(penalty='none', fit_intercept=True)
    sklearn.fit(X, y)
    sklearn_ = ch.from_numpy(np.concatenate([sklearn.coef_, sklearn.intercept_[...,None]], axis=1))
    print(f'sklearn: {sklearn_}')
    pred = sklearn.predict(X)
    acc = np.equal(pred, y).sum() / len(y)
    print(f'sklearn acc: {acc}')
    sklearn_conf_matrix = confusion_matrix(y, pred)
    print(f'sklearn confusion matrix: \n {sklearn_conf_matrix}')

    train_kwargs = Parameters({
                        'batch_size': 100,
                        'epochs': 30,
                        'momentum': 0.5,
                        'trials': 3,
                        'verbose': True,
                        'early_stopping': True,
                        'workers': 0})        
    ch.manual_seed(seed)
    X_ones = ch.cat([X, ch.ones(X.size(0), 1)], axis=1)
    softmax_model = stats.SoftmaxModel(train_kwargs, X_ones.size(1), len(y.unique()))
    trainer = Trainer(softmax_model, train_kwargs) 
    train_loader, val_loader = make_train_and_val(train_kwargs, X_ones, y)
    trainer.train_model((train_loader, val_loader))
    softmax_ = softmax_model.model

    softmax_cos_sim = float(cos_sim(softmax_.flatten()[None,...], sklearn_.flatten()[None,...]))
    softmax_pred = softmax_model.predict(X_ones)
    softmax_acc = softmax_pred.eq(y).sum() / len(y)
    print(f'softmax accuracy: {softmax_acc}')
    print(f'softmax cos sim: {softmax_cos_sim}')
    softmax_conf_matrix = confusion_matrix(y, softmax_pred)
    print(f'softmax confusion matrix: \n {softmax_conf_matrix}')
        