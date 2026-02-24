"""
Test suite for various statistical models.
Includes:
    -Softmax regression
"""

import torch as ch
from torch.distributions import Gumbel
from torch.nn import CosineSimilarity
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

from delphi.utils.helpers import Parameters
from delphi.stats.softmax import SoftmaxRegression

cos_sim = CosineSimilarity()
gumbel = Gumbel(0, 1)
seed = 69


# softmax regression test
def test_softmax_regression():
    D, K = 1, 2

    W = ch.rand(K, D)
    W_eff = W[1] - W[0]
    print(f"ground truth: {W}")
    print(f"effective ground truth: {W_eff}")

    # input features
    NUM_SAMPLES = 5000
    X = ch.rand(NUM_SAMPLES, D)
    # latent variables
    z = X @ W.T + gumbel.sample([X.size(0), K])
    # classification
    y = z.argmax(-1, keepdim=True)

    sklearn = LogisticRegression(penalty=None, fit_intercept=False)
    sklearn.fit(X, y.flatten())
    sklearn_ = ch.from_numpy(sklearn.coef_)

    print(f"sklearn: {sklearn_}")
    pred = sklearn.predict(X)
    acc = np.equal(pred, y.flatten()).sum() / len(y)
    print(f"sklearn acc: {acc}")
    sklearn_conf_matrix = confusion_matrix(y, pred)
    print(f"sklearn confusion matrix: \n {sklearn_conf_matrix}")

    args = Parameters(
        {
            "batch_size": 32,
            "epochs": 10,
            "lr": 1e-1,
            "trials": 1,
            "verbose": True,
            "early_stopping": True,
            "grad_tol": 0,
            "patience": float("inf"),
        }
    )
    ch.manual_seed(seed)
    delphi_soft_reg = SoftmaxRegression(args, fit_intercept=False)
    delphi_soft_reg.fit(X, y)
    delphi_soft_reg_ = delphi_soft_reg.coef_
    print(f"delphi log reg: {delphi_soft_reg_}")
    delphi_diff_ = delphi_soft_reg_[1] - delphi_soft_reg_[0]
    print(f"delphi diff: {delphi_diff_}")
    delphi_cos_sim = float(cos_sim(delphi_diff_[None, ...], W_eff))
    delphi_pred = delphi_soft_reg.predict(X)
    delphi_acc = delphi_pred.eq(y.flatten()).sum() / len(y)
    print(f"delphi accuracy: {delphi_acc}")
    print(f"delphi cos sim: {delphi_cos_sim}")
    delphi_conf_matrix = confusion_matrix(y, delphi_pred)
    print(f"delphi confusion matrix: \n {delphi_conf_matrix}")
    assert delphi_cos_sim > 9e-1, (
        f"delphi softmax reg cosine similarity is {delphi_cos_sim}"
    )
