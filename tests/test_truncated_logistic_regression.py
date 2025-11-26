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
from delphi.stats.softmax import SoftmaxRegression

cos_sim = CosineSimilarity()
gumbel = Gumbel(0, 1)
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

# 1 dimensional untruncated multinomial logistic regression 
def test_untruncated_multinomial_logistic_regression():     
    D, K = 5, 2

    W = ch.rand(K, D)
    W_eff = W[1] - W[0]
    print(f'ground truth: \n {W}')
    print(f'effective ground truth: \n {W_eff}')
    
    # input features
    NUM_SAMPLES = 5000
    X = ch.rand(NUM_SAMPLES, D) 
    # latent variables
    z = X@W.T + gumbel.sample([X.size(0), K])
    # classification
    y = z.argmax(-1)

    sklearn = LogisticRegression(penalty=None, fit_intercept=False)
    sklearn.fit(X, y.flatten())
    sklearn_ = ch.from_numpy(sklearn.coef_)
    print(f'sklearn: \n {sklearn_}')
    pred = sklearn.predict(X)
    acc = np.equal(pred, y.flatten()).sum() / len(y)
    print(f'sklearn acc: \n {acc}')
    sklearn_conf_matrix = confusion_matrix(y, pred)
    print(f'sklearn confusion matrix: \n {sklearn_conf_matrix}')

    def phi(z): 
        z_diff = (z[...,1] - z[...,0])[...,None]
        return ch.ones_like(z_diff)
    
    trunc_indices = phi(z).flatten().nonzero().flatten() 
    X_trunc, Y_trunc = X[trunc_indices], y[trunc_indices][...,None]
    alpha = X_trunc.size(0) / X.size(0)
    print(f'alpha: {alpha}')

    trunc_sklearn = LogisticRegression(penalty=None, fit_intercept=False)
    trunc_sklearn.fit(X_trunc, Y_trunc.flatten())
    trunc_sklearn_ = ch.from_numpy(trunc_sklearn.coef_)
    print(f'trunc sklearn: {trunc_sklearn_}')
    pred = trunc_sklearn.predict(X)
    acc = np.equal(pred, Y_trunc.flatten()).sum() / len(Y_trunc)
    print(f'trunc sklearn acc: {acc}')
    trunc_sklearn_conf_matrix = confusion_matrix(Y_trunc, pred)
    print(f'trunc sklearn confusion matrix: \n {trunc_sklearn_conf_matrix}')

    # emp_weight = ch.randn(W.size())
    # print(f'emp weight: {emp_weight}')
    args = Parameters({
                        'batch_size': 50,
                        'epochs': 10,
                        'trials': 1,
                        'verbose': True,
                        'early_stopping': True,
                        'num_samples': 10000,
                        'gradient_steps': 1500,
                        'grad_tol': 1e-3,
                        'step_lr_gamma': 1.0, 
                        'lr': 1e-1,
                    })        
    ch.manual_seed(seed)
    delphi_log_reg = TruncatedLogisticRegression(args,
                                                phi, 
                                                alpha, 
                                                fit_intercept=False,
                                                multi_class="multinomial")
                                                # emp_weight=emp_weight)
    delphi_log_reg.fit(X_trunc, Y_trunc)
    delphi_log_reg_ = delphi_log_reg.coef_                
    print(f'delphi log reg: {delphi_log_reg_}')
    delphi_diff_ = delphi_log_reg_[1] - delphi_log_reg_[0]
    print(f'delphi diff: {delphi_diff_}')
    delphi_cos_sim = float(cos_sim(delphi_diff_[None,...], W_eff))
    delphi_pred = delphi_log_reg.predict(X_trunc)
    delphi_acc = delphi_pred.eq(Y_trunc.flatten()).sum() / len(Y_trunc)
    print(f'delphi accuracy: {delphi_acc}')
    print(f'delphi cos sim: {delphi_cos_sim}')
    delphi_conf_matrix = confusion_matrix(Y_trunc, delphi_pred)
    print(f'delphi confusion matrix: \n {delphi_conf_matrix}')
    # assert delphi_cos_sim > 9e-1, f"trunc multinomial log reg cosine similarity is {delphi_cos_sim}"

    args = Parameters({
                        'batch_size': 50,
                        'epochs': 10,
                        'trials': 1,
                        'verbose': True,
                        'early_stopping': True,
                        'gradient_steps': 1500,
                        'grad_tol': 1e-3,
                        'step_lr_gamma': 1.0,
                    })        
    ch.manual_seed(seed)
    delphi_soft_reg = SoftmaxRegression(args, 
                                       fit_intercept=False)
    delphi_soft_reg.fit(X_trunc, Y_trunc)
    delphi_soft_reg_ = delphi_soft_reg.coef_                
    print(f'delphi soft reg: {delphi_soft_reg_}')
    delphi_soft_diff_ = delphi_soft_reg_[1] - delphi_soft_reg_[0]
    print(f'delphi soft diff: {delphi_soft_diff_}')
    delphi_soft_cos_sim = float(cos_sim(delphi_soft_diff_[None,...], W_eff))
    delphi_soft_pred = delphi_soft_reg.predict(X_trunc)
    delphi_soft_acc = delphi_soft_pred.eq(Y_trunc.flatten()).sum() / len(Y_trunc)
    print(f'delphi soft accuracy: {delphi_soft_acc}')
    print(f'delphi soft cos sim: {delphi_soft_cos_sim}')
    delphi_soft_conf_matrix = confusion_matrix(Y_trunc, delphi_soft_pred)
    print(f'delphi soft confusion matrix: \n {delphi_soft_conf_matrix}')
    assert delphi_soft_cos_sim > 9e-1, f"trunc multinomial log reg cosine similarity is {delphi_soft_cos_sim}"

# 1 dimensional truncated multinomial logistic regression 
def test_truncated_multinomial_logistic_regression():   
    ch.manual_seed(69)  
    d, k = 2, 2
    C = .1
    # ground-truth logistic regression model 
    w = Uniform(-1, 1)
    U = Uniform(-5, 5)
    W = w.sample([k, d])
    print(f'ground truth: {W}')
    print(f'effective ground truth: {W[1] - W[0]}')
    
    # input features
    SAMPLES = 5000
    X = U.sample([SAMPLES, d])
    # latent variables
    z = X@W.T + gumbel.sample([X.size(0), k])
    # classification
    y = z.argmax(-1)

    sklearn = LogisticRegression(penalty=None, fit_intercept=False)
    sklearn.fit(X, y.flatten())
    sklearn_ = ch.from_numpy(sklearn.coef_)

    print(f'sklearn: {sklearn_}')
    pred = sklearn.predict(X)
    acc = np.equal(pred, y.flatten()).sum() / len(y)
    print(f'sklearn acc: {acc}')
    sklearn_conf_matrix = confusion_matrix(y, pred)
    print(f'sklearn confusion matrix: \n {sklearn_conf_matrix}')

    sklearn_noised = X@sklearn_.T.float() + logistic.sample([X.size(0), 1])

    def phi_logistic(z_diff): 
        z_diff = (z_diff - .1).abs()
        return (z_diff < C)[...,None]

    def phi(z): 
        z_diff = ((z[...,1] - z[...,0]) - .1).abs()
        return (z_diff < C)[...,None]
    
    def phi(z): 
        return ch.ones(z.size(-2), 1)
    
    trunc_indices = phi_logistic(sklearn_noised).flatten().nonzero().flatten()  
    X_trunc, Y_trunc = X[trunc_indices], y[trunc_indices][...,None]
    alpha = X_trunc.size(0) / X.size(0)
    print(f'alpha: {alpha}')
    print(f'number of samples in truncation set: {X_trunc.size(0)}')

    trunc_sklearn = LogisticRegression(penalty=None, fit_intercept=False)
    trunc_sklearn.fit(X_trunc, Y_trunc.flatten())
    trunc_sklearn_ = ch.from_numpy(trunc_sklearn.coef_)

    print(f'trunc sklearn: {trunc_sklearn_}')
    pred = trunc_sklearn.predict(X)
    acc = np.equal(pred, y.flatten()).sum() / len(y)
    print(f'trunc sklearn acc: {acc}')
    trunc_sklearn_conf_matrix = confusion_matrix(y, pred)
    print(f'trunc sklearn confusion matrix: \n {trunc_sklearn_conf_matrix}')
    trunc_sklearn_cos_sim = cos_sim(trunc_sklearn_, sklearn_)
    print(f'trunc sklearn cos sim: {trunc_sklearn_cos_sim}')

    args = Parameters({
                        'batch_size': 50,
                        'epochs': 50,
                        'trials': 1,
                        'verbose': True,
                        'early_stopping': True,
                        'num_samples': 10000,
                        'grad_tol': -1, 
                        'lr': 1e-2, 
                        'step_lr_gamma': 1.0
                    })        
    ch.manual_seed(seed)
    delphi_log_reg = TruncatedLogisticRegression(args,
                                                phi, 
                                                alpha, 
                                                fit_intercept=False,
                                                multi_class="multinomial")
    delphi_log_reg.fit(X_trunc, Y_trunc)
    delphi_log_reg_ = delphi_log_reg.coef_                
    print(f'delphi log reg: {delphi_log_reg_.tolist()}')
    delphi_diff_ = delphi_log_reg_[1] - delphi_log_reg_[0]
    print(f'delphi diff: {delphi_diff_.tolist()}')
    delphi_cos_sim = float(cos_sim(delphi_diff_[None,...], sklearn_))
    delphi_pred = delphi_log_reg.predict(X)
    delphi_acc = delphi_pred.eq(y).sum() / len(y)
    print(f'delphi accuracy: {delphi_acc}')
    print(f'delphi cos sim: {delphi_cos_sim}')
    delphi_conf_matrix = confusion_matrix(y[...,None], delphi_pred)
    print(f'delphi confusion matrix: \n {delphi_conf_matrix}')

    args = Parameters({
                        'batch_size': 50,
                        'epochs': 50,
                        'trials': 1,
                        'verbose': True,
                        'early_stopping': True,
                        'num_samples': 10000,
                        'grad_tol': -1, 
                        'lr': 1e-2, 
                        'step_lr_gamma': 1.0
                    })        
    ch.manual_seed(seed)
    delphi_softmax = SoftmaxRegression(args,
                                        fit_intercept=False)
    delphi_softmax.fit(X_trunc, Y_trunc)
    delphi_softmax_ = delphi_softmax.coef_                
    print(f'delphi softmax: {delphi_softmax_.tolist()}')
    delphi_soft_diff_ = delphi_softmax_[1] - delphi_softmax_[0]
    print(f'delphi sofmax diff: {delphi_soft_diff_.tolist()}')
    delphi_soft_cos_sim = float(cos_sim(delphi_soft_diff_[None,...], sklearn_))
    delphi_soft_pred = delphi_softmax.predict(X)
    delphi_soft_acc = delphi_soft_pred.eq(y).sum() / len(y)
    print(f'delphi soft accuracy: {delphi_soft_acc}')
    print(f'delphi soft cos sim: {delphi_soft_cos_sim}')
    delphi_soft_conf_matrix = confusion_matrix(y[...,None], delphi_soft_pred)
    print(f'delphi soft confusion matrix: \n {delphi_soft_conf_matrix}')


    import pdb; pdb.set_trace()
    assert delphi_cos_sim > 9e-1, f"trunc multinomial log reg cosine similarity is {delphi_cos_sim}"

# 3 dimensional truncated multinomial logistic regression 
def test_3_dim_truncated_multinomial_logistic_regression():     
    d, k = 3, 2
    C = -.25
    # ground-truth logistic regression model 
    w = Uniform(-1, 1)
    U = Uniform(-5, 5)
    W = w.sample([k, d])
    print(f'ground truth: {W}')
    print(f'effective ground truth: {W[1] - W[0]}')
    
    # input features
    SAMPLES = 10000
    X = U.sample([SAMPLES, d])
    # latent variables
    z = X@W.T + gumbel.sample([X.size(0), k])
    # classification
    y = z.argmax(-1)

    sklearn = LogisticRegression(penalty=None, fit_intercept=False)
    sklearn.fit(X, y.flatten())
    sklearn_ = ch.from_numpy(sklearn.coef_)

    print(f'sklearn: {sklearn_}')
    pred = sklearn.predict(X)
    acc = np.equal(pred, y.flatten()).sum() / len(y)
    print(f'sklearn acc: {acc}')
    sklearn_conf_matrix = confusion_matrix(y, pred)
    print(f'sklearn confusion matrix: \n {sklearn_conf_matrix}')

    def phi(z): 
        z_diff = z[...,1] - z[...,0]
        return (z_diff > C)[...,None]
    
    trunc_indices = phi(z).flatten().nonzero().flatten() 
    X_trunc, Y_trunc = X[trunc_indices], y[trunc_indices][...,None]
    alpha = X_trunc.size(0) / X.size(0)
    print(f'alpha: {alpha}')

    trunc_sklearn = LogisticRegression(penalty=None, fit_intercept=False)
    trunc_sklearn.fit(X_trunc, Y_trunc.flatten())
    trunc_sklearn_ = ch.from_numpy(trunc_sklearn.coef_)

    print(f'trunc sklearn: {trunc_sklearn_}')
    pred = trunc_sklearn.predict(X)
    acc = np.equal(pred, y.flatten()).sum() / len(y)
    print(f'trunc sklearn acc: {acc}')
    trunc_sklearn_conf_matrix = confusion_matrix(y, pred)
    print(f'trunc sklearn confusion matrix: \n {trunc_sklearn_conf_matrix}')
    trunc_sklearn_cos_sim = cos_sim(trunc_sklearn_, sklearn_)
    print(f'trunc sklearn cos sim: {trunc_sklearn_cos_sim}')

    args = Parameters({
                        'batch_size': 50,
                        'epochs': 20,
                        'trials': 1,
                        'verbose': True,
                        'early_stopping': True,
                        'num_samples': 5000,
                    })        
    ch.manual_seed(seed)
    delphi_log_reg = TruncatedLogisticRegression(args,
                                                 phi, 
                                                 alpha, 
                                                 fit_intercept=False,
                                                 multi_class="multinomial")
    delphi_log_reg.fit(X_trunc, Y_trunc)
    delphi_log_reg_ = delphi_log_reg.coef_                
    print(f'delphi log reg:\n {delphi_log_reg_.tolist()}')
    delphi_diff_ = delphi_log_reg_[1] - delphi_log_reg_[0]
    print(f'delphi diff:\n {delphi_diff_.tolist()}')
    delphi_cos_sim = float(cos_sim(delphi_diff_[None,...], sklearn_))
    delphi_pred = delphi_log_reg.predict(X)
    delphi_acc = delphi_pred.eq(y).sum() / len(y)
    print(f'delphi accuracy: {delphi_acc}')
    print(f'delphi cos sim: {delphi_cos_sim}')
    delphi_conf_matrix = confusion_matrix(y[...,None], delphi_pred)
    print(f'delphi confusion matrix: \n {delphi_conf_matrix}')
    assert delphi_cos_sim > 9e-1, f"trunc multinomial log reg cosine similarity is {delphi_cos_sim}"

# 10 dimensional truncated multinomial logistic regression 
def test_10_dim_truncated_multinomial_logistic_regression():     
    d, k = 10, 2
    C = -.25
    # ground-truth logistic regression model 
    w = Uniform(-1, 1)
    U = Uniform(-5, 5)
    W = w.sample([k, d])
    print(f'ground truth: {W}')
    print(f'effective ground truth: {W[1] - W[0]}')
    
    # input features
    SAMPLES = 10000
    X = U.sample([SAMPLES, d])
    # latent variables
    z = X@W.T + gumbel.sample([X.size(0), k])
    # classification
    y = z.argmax(-1)

    sklearn = LogisticRegression(penalty=None, fit_intercept=False)
    sklearn.fit(X, y.flatten())
    sklearn_ = ch.from_numpy(sklearn.coef_)

    print(f'sklearn: {sklearn_}')
    pred = sklearn.predict(X)
    acc = np.equal(pred, y.flatten()).sum() / len(y)
    print(f'sklearn acc: {acc}')
    sklearn_conf_matrix = confusion_matrix(y, pred)
    print(f'sklearn confusion matrix: \n {sklearn_conf_matrix}')

    def phi(z): 
        z_diff = z[...,1] - z[...,0]
        return (z_diff > C)[...,None]
    
    trunc_indices = phi(z).flatten().nonzero().flatten() 
    X_trunc, Y_trunc = X[trunc_indices], y[trunc_indices][...,None]
    alpha = X_trunc.size(0) / X.size(0)
    print(f'alpha: {alpha}')

    trunc_sklearn = LogisticRegression(penalty=None, fit_intercept=False)
    trunc_sklearn.fit(X_trunc, Y_trunc.flatten())
    trunc_sklearn_ = ch.from_numpy(trunc_sklearn.coef_)

    print(f'trunc sklearn: {trunc_sklearn_}')
    pred = trunc_sklearn.predict(X)
    acc = np.equal(pred, y.flatten()).sum() / len(y)
    print(f'trunc sklearn acc: {acc}')
    trunc_sklearn_conf_matrix = confusion_matrix(y, pred)
    print(f'trunc sklearn confusion matrix: \n {trunc_sklearn_conf_matrix}')
    trunc_sklearn_cos_sim = cos_sim(trunc_sklearn_, sklearn_)
    print(f'trunc sklearn cos sim: {trunc_sklearn_cos_sim}')

    args = Parameters({
                        'batch_size': 50,
                        'epochs': 20,
                        'trials': 1,
                        'verbose': True,
                        'early_stopping': True,
                        'num_samples': 5000,
                    })        
    ch.manual_seed(seed)
    delphi_log_reg = TruncatedLogisticRegression(args,
                                                 phi, 
                                                 alpha, 
                                                 fit_intercept=False,
                                                 multi_class="multinomial")
    delphi_log_reg.fit(X_trunc, Y_trunc)
    delphi_log_reg_ = delphi_log_reg.coef_                
    print(f'delphi log reg:\n {delphi_log_reg_.tolist()}')
    delphi_diff_ = delphi_log_reg_[1] - delphi_log_reg_[0]
    print(f'delphi diff:\n {delphi_diff_.tolist()}')
    delphi_cos_sim = float(cos_sim(delphi_diff_[None,...], sklearn_))
    delphi_pred = delphi_log_reg.predict(X)
    delphi_acc = delphi_pred.eq(y).sum() / len(y)
    print(f'delphi accuracy: {delphi_acc}')
    print(f'delphi cos sim: {delphi_cos_sim}')
    delphi_conf_matrix = confusion_matrix(y[...,None], delphi_pred)
    print(f'delphi confusion matrix: \n {delphi_conf_matrix}')
    assert delphi_cos_sim > 9e-1, f"trunc multinomial log reg cosine similarity is {delphi_cos_sim}"

# 1 dimensional truncated multinomial logistic regression - truncate on labels
def test_truncated_multinomial_logistic_regression_labels():     
    d, k = 10, 2
    C = .5
    # ground-truth logistic regression model 
    w = Uniform(-1, 1)
    U = Uniform(-5, 5)
    W = w.sample([k, d])
    print(f'ground truth: {W}')
    print(f'effective ground truth: {W[1] - W[0]}')
    
    # input features
    SAMPLES = 10000
    X = U.sample([SAMPLES, d])
    # latent variables
    z = X@W.T + gumbel.sample([X.size(0), k])
    # classification
    y = z.argmax(-1)

    sklearn = LogisticRegression(penalty=None, fit_intercept=False)
    sklearn.fit(X, y.flatten())
    sklearn_ = ch.from_numpy(sklearn.coef_)

    print(f'sklearn: {sklearn_}')
    pred = sklearn.predict(X)
    acc = np.equal(pred, y.flatten()).sum() / len(y)
    print(f'sklearn acc: {acc}')
    sklearn_conf_matrix = confusion_matrix(y, pred)
    print(f'sklearn confusion matrix: \n {sklearn_conf_matrix}')

    sklearn_noised = X@sklearn_.T.float() + logistic.sample([X.size(0), 1])

    def phi_logistic(z_diff, y):
        return ((y == 1) & (z_diff < C)) | ((y == 0) & (z_diff > C))

    def phi(z, y):
        z_diff = (z[..., 1] - z[..., 0])[...,None]
        return ((y == 1) & (z_diff < C)) | ((y == 0) & (z_diff > C))
    
    trunc_indices = phi_logistic(sklearn_noised, y[...,None]).flatten().nonzero().flatten() 
    X_trunc, Y_trunc = X[trunc_indices], y[trunc_indices][...,None]
    alpha = X_trunc.size(0) / X.size(0)
    print(f'alpha: {alpha}')

    trunc_sklearn = LogisticRegression(penalty=None, fit_intercept=False)
    trunc_sklearn.fit(X_trunc, Y_trunc.flatten())
    trunc_sklearn_ = ch.from_numpy(trunc_sklearn.coef_)

    print(f'trunc sklearn: {trunc_sklearn_}')
    pred = trunc_sklearn.predict(X)
    acc = np.equal(pred, y.flatten()).sum() / len(y)
    print(f'trunc sklearn acc: {acc}')
    trunc_sklearn_conf_matrix = confusion_matrix(y, pred)
    print(f'trunc sklearn confusion matrix: \n {trunc_sklearn_conf_matrix}')
    trunc_sklearn_cos_sim = cos_sim(trunc_sklearn_, sklearn_)
    print(f'trunc sklearn cos sim: {trunc_sklearn_cos_sim}')

    args = Parameters({
                        'batch_size': 50,
                        'epochs': 20,
                        'trials': 1,
                        'verbose': True,
                        'early_stopping': True,
                        'num_samples': 10000,
                        'grad_tol': 1e-3
                    })        
    ch.manual_seed(seed)
    delphi_log_reg = TruncatedLogisticRegression(args,
                                                phi, 
                                                alpha, 
                                                fit_intercept=False,
                                                multi_class="multinomial")
    delphi_log_reg.fit(X_trunc, Y_trunc)
    delphi_log_reg_ = delphi_log_reg.coef_                
    print(f'delphi log reg: {delphi_log_reg_.tolist()}')
    delphi_diff_ = delphi_log_reg_[1] - delphi_log_reg_[0]
    print(f'delphi diff: {delphi_diff_.tolist()}')
    delphi_cos_sim = float(cos_sim(delphi_diff_[None,...], sklearn_))
    delphi_pred = delphi_log_reg.predict(X)
    delphi_acc = delphi_pred.eq(y).sum() / len(y)
    print(f'delphi accuracy: {delphi_acc}')
    print(f'delphi cos sim: {delphi_cos_sim}')
    delphi_conf_matrix = confusion_matrix(y[...,None], delphi_pred)
    print(f'delphi confusion matrix: \n {delphi_conf_matrix}')
    assert delphi_cos_sim > 9e-1, f"trunc multinomial log reg cosine similarity is {delphi_cos_sim}"
