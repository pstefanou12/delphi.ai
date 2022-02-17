# distribution tests 

from re import A, I
import unittest
import numpy as np
import torch as ch
from torch import Tensor
from torch.distributions import MultivariateNormal, Uniform, Gumbel
from torch.distributions.kl import kl_divergence
from torch.distributions.multivariate_normal import _batch_mahalanobis
from torch.distributions.transforms import SigmoidTransform
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.nn import CosineSimilarity, Softmax, CrossEntropyLoss
from torch.nn import MSELoss
import torch.linalg as LA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import confusion_matrix
import random
from cox.store import Store

from delphi import delphi
from delphi.trainer import Trainer
from delphi import stats 
from delphi import oracle
from delphi.utils.helpers import Parameters, cov, accuracy
from delphi.utils.datasets import make_train_and_val
from delphi.grad import GumbelCE

# CONSTANT
mse_loss =  MSELoss()
base_distribution = Uniform(0, 1)
transforms_ = [SigmoidTransform().inv]
logistic = TransformedDistribution(base_distribution, transforms_)
cos_sim = CosineSimilarity()
softmax = Softmax(dim=0)
ce = CrossEntropyLoss()
G = Gumbel(0, 1)


class GumbelLogisticLeftTruncation:
    """
    The difference beween two samples from a Gumbel distribution is equivalent to 
    a sample from a logistic distribution. Thus, this oracle takes noised logits 
    from a multinomial logistic regression, calculates the difference and uses the
    left truncation mechanism for truncated logistic regression.
    """
    def __init__(self, left): 
        """
        Args: 
           left (float): left truncation threshold
        """
        self.left = left 
        
    def __call__(self, x): 
        """
        Args: 
            x (torch.Tensor): n by 2 array of gumbel noised logits
        """
        return ((x[:,:,1] - x[:,:,0]) > self.left)[...,None]

    def __str__(self): 
      return 'Gumbel Logistic Left Truncation Set'


class GumbelCEModel(stats.LinearModel):
    '''
    Truncated logistic regression model to pass into trainer framework.
    '''
    def __init__(self, args, d, k):
        '''
        Args:
            args (cox.utils.Parameters) : parameter object holding hyperparameters
        '''
        super().__init__(args, d=d, k=k)

    def calc_emp_model(self): 
        pass    

    def pretrain_hook(self):
        self.model.data = self.weight
        self.model.requires_grad = True
        self.params = [self.model]

    def predict(self, x): 
        with ch.no_grad():
            stacked = (x@self.model.T).repeat(self.args.num_samples, 1, 1)
            noised = stacked + G.sample(stacked.size())
            return noised.mean(0).argmax(-1)

    def __call__(self, batch):
        '''
        Training step for defined model.
        Args:
            batch (Iterable) : iterable of inputs that
        '''
        inp, targ = batch
        z = inp@self.model.T
        loss = GumbelCE.apply(z, targ)
        pred = z.argmax(-1)
        
        # calculate precision accuracies
        if z.size(1) >= 5:
            prec1, prec5 = accuracy(pred.reshape(pred.size(0), 1), targ.reshape(targ.size(0), 1).float(), topk=(1, 5))
        else:
            prec1, prec5 = accuracy(pred.reshape(pred.size(0), 1), targ.reshape(targ.size(0), 1).float(), topk=(1,))
        return loss, prec1, prec5
    
    def calc_logits(self, inp): 
        return self.model(inp)


class SoftmaxModel(stats.LinearModel):
    '''
    Truncated logistic regression model to pass into trainer framework.
    '''
    def __init__(self, args, d, k):
        '''
        Args:
            args (cox.utils.Parameters) : parameter object holding hyperparameters
        '''
        super().__init__(args, d=d, k=k)

    def calc_emp_model(self): 
        pass    

    def pretrain_hook(self):
        self.model.data = self.weight
        self.model.requires_grad = True
        self.params = [self.model]
        
    def predict(self, x): 
        with ch.no_grad():
            return softmax(x@self.model.T).argmax(dim=-1)

    def __call__(self, batch):
        '''
        Training step for defined model.
        Args:
            batch (Iterable) : iterable of inputs that
        '''
        inp, targ = batch
        z = inp@self.model.T
        loss = ce(z, targ)
        pred = z.argmax(-1)
        
        # calculate precision accuracies
        if z.size(1) >= 5:
            prec1, prec5 = accuracy(pred.reshape(pred.size(0), 1), targ.reshape(targ.size(0), 1).float(), topk=(1, 5))
        else:
            prec1, prec5 = accuracy(pred.reshape(pred.size(0), 1), targ.reshape(targ.size(0), 1).float(), topk=(1,))
        return loss, prec1, prec5
    
    def calc_logits(self, inp): 
        return self.model(inp)

class TestStats(unittest.TestCase): 
    """
    Test suite for the stats module.
    """
    ## right truncated normal distribution with known truncation
    #def test_truncated_regression(self):
    #    W = Uniform(-1, 1)
    #    M = Uniform(-10, 10)
    #    X = M.rsample([10000, 10])
    #    # generate ground truth
    #    gt = ch.nn.Linear(in_features=1, out_features=1)
    #    gt.weight = ch.nn.Parameter(W.sample(ch.Size([1, 10])))
    #    gt.bias = ch.nn.Parameter(W.sample(ch.Size([1, 1])))
    #    noise_var = Tensor([10.0])[...,None]
    #    with ch.no_grad():
    #        # generate data
    #        y = gt(X) + ch.sqrt(noise_var) * ch.randn(X.size(0), 1) 
    #    # generate ground-truth data
    #    phi = oracle.Left_Regression(Tensor([0.0]))
    #    # truncate
    #    indices = phi(y).nonzero()[:,0]
    #    x_trunc, y_trunc = X[indices], y[indices]
    #    alpha = x_trunc.size(0) / X.size(0)
    #    # normalize input features
    #    l_inf = LA.norm(x_trunc, dim=-1, ord=float('inf')).max()
    #    beta = l_inf * (10 ** .5)
    #    x_trunc /= beta
    #    X /= beta 

    #    gt_norm = LinearRegression()
    #    gt_norm.fit(X, y)
    #    gt_ = ch.from_numpy(np.concatenate([gt_norm.coef_.flatten(), gt_norm.intercept_]))

    #    # calculate empirical noise variance for regression 
    #    ols_trunc = LinearRegression()
    #    ols_trunc.fit(x_trunc, y_trunc)
    #    emp_noise_var = ch.from_numpy(ols_trunc.predict(X) - y.numpy()).var(0)

    #    # scale y features
    #    y_trunc_scale = y_trunc / ch.sqrt(noise_var)
    #    phi_scale = oracle.Left_Regression(phi.left / ch.sqrt(noise_var))
    #    # train algorithm
    #    train_kwargs = Parameters({'phi': phi_scale, 
    #                            'alpha': alpha,
    #                            'epochs': 10, 
    #                            'batch_size': 100,
    #                            'normalize': False,
    #                            'noise_var': 1.0}) 
    #    trunc_reg = stats.TruncatedLinearRegression(train_kwargs)
    #    trunc_reg.fit(x_trunc, y_trunc_scale)
    #    w_ = ch.cat([(trunc_reg.coef_).flatten(), trunc_reg.intercept_]) * ch.sqrt(noise_var)
    #    known_mse_loss = mse_loss(gt_, w_.flatten())
    #    self.assertTrue(known_mse_loss <= 3e-1, f'known mse loss: {known_mse_loss}')
    #    
    #   # scale y features by empirical noise variance
    #    y_trunc_emp_scale = y_trunc / ch.sqrt(emp_noise_var)
    #    phi_emp_scale = oracle.Left_Regression(phi.left / ch.sqrt(emp_noise_var))
    #    # train algorithm
    #    train_kwargs = Parameters({'phi': phi_emp_scale, 
    #                            'alpha': alpha,
    #                            'epochs': 10, 
    #                            'batch_size': 10,
    #                            'normalize': False,})
    #    unknown_trunc_reg = stats.TruncatedLinearRegression(train_kwargs)
    #    unknown_trunc_reg.fit(x_trunc, y_trunc_emp_scale)
    #    w_ = ch.cat([(unknown_trunc_reg.coef_).flatten(), unknown_trunc_reg.intercept_]) * ch.sqrt(emp_noise_var)
    #    noise_var_ = unknown_trunc_reg.variance_ * emp_noise_var
    #    unknown_mse_loss = mse_loss(gt_, w_.flatten())
    #    unknown_var_l1 = ch.abs(noise_var - noise_var_)
    #    self.assertTrue(unknown_mse_loss <= 3e-1, f'unknown mse loss: {unknown_mse_loss}')
#   #     self.assertTrue(unknown_var_l1 <= 3e-1)

    def test_truncated_logistic_regression(self):
        OUT_DIR = '/Users/patroklos/Desktop/exp/truncated/'
        result_store = Store(OUT_DIR + 'debug')
        result_store.add_table('models', {
            'sklearn': '__object__', 'sklearn_acc': float, 'sklearn_conf_matrix': '__object__',
            'trunc_sklearn': '__object__', 'trunc_sklearn_acc': float, 'trunc_sklearn_conf_matrix': '__object__', 'trunc_sklearn_cos_sim': float,
            'softmax_diff': '__object__', 'softmax_acc': float, 'softmax_conf_matrix': '__object__', 'softmax_cos_sim': float,
            'softmax': '__object__',
            'gumbel_diff': '__object__', 'gumbel_acc': float, 'gumbel_conf_matrix': '__object__', 'gumbel_cos_sim': float,
            'gumbel': '__object__', 
            'trunc_log_reg': '__object__', 'trunc_log_reg_acc': float, 'trunc_log_reg_conf_matrix': '__object__', 'trunc_log_reg_cos_sim': float,
            'trunc_multi_log_reg_diff': '__object__',  'trunc_multi_log_reg_acc': float, 'trunc_multi_log_reg_conf_matrix': '__object__', 'trunc_multi_log_reg_cos_sim': float,
            'trunc_multi_log_reg': '__object__',            
            'id_trunc_log_reg': '__object__', 'id_trunc_log_reg_acc': float, 'id_trunc_log_reg_conf_matrix': '__object__', 'id_trunc_log_reg_cos_sim': float,
            'id_trunc_multi_log_reg_diff': '__object__',  'id_trunc_multi_log_reg_acc': float, 'id_trunc_multi_log_reg_conf_matrix': '__object__', 'id_trunc_multi_log_reg_cos_sim': float,
            'id_trunc_multi_log_reg': '__object__', 
#            'X': '__object__', 'y': '__object__',
            'alpha': float, 
            'c': float })

        d, k = 5, 1
        trials = 3
        C = [-2.0, -1.5, -1.0, -.75, -.5, -.25]
            # ground-truth logistic regression model 
        gt = ch.nn.Linear(in_features=d, out_features=k, bias=True)
        W = Uniform(-1, 1)
        U = Uniform(-5, 5)
        gt.weight = ch.nn.Parameter(W.sample([k, d]))

        gt.bias = ch.nn.Parameter(W.sample([1, k]))
        gt_ = ch.cat([gt.weight.flatten(), gt.bias.flatten()])

        for i in range(trials):
            print(f'gt: {gt_}')
#             gt.weight = ch.nn.Parameter(ch.ones(k, d))
 #            gt.bias = ch.nn.Parameter(ch.ones(1, k))

            # input features
            X = U.sample([10000, d])
            # latent variables
            z = gt(X) + logistic.sample([X.size(0), 1])
            # classification
            y = (z > 0).float()

            for c in C:
                seed = random.randint(0, 100)
                # generate ground-truth data
                phi = oracle.Left_Regression(Tensor([c]))
                phi_gumbel = GumbelLogisticLeftTruncation(Tensor([c]))
                # phi = oracle.Identity()
                indices = phi(z).nonzero(as_tuple=True)
                x_trunc = X[indices[0]]
                y_trunc = y[indices][...,None]
                alpha = x_trunc.size(0) / X.size(0)
                norm = LA.norm(x_trunc, dim=-1, ord=float('inf')).max()
                print(f'max sample norm: {norm}')
                beta = norm * (d ** .5)
                print(f'beta: {beta}')
#                x_trunc  /= beta
                X_ = X 

                print(f'C: {c}')
                print(f'alpha: {alpha}')
                result_store['models'].update_row({ 
#                    'X': x_trunc, 
#                    'y': y_trunc,
                    'alpha': alpha, 
                    'c': c
                })
        
                sklearn = LogisticRegression(penalty='none', fit_intercept=True)
                sklearn.fit(X_, y.flatten())
                sklearn_ = ch.from_numpy(np.concatenate([sklearn.coef_.flatten(), sklearn.intercept_]))
                print(f'sklearn: {sklearn_}')
                pred = sklearn.predict(X_)
                acc = np.equal(pred, y.flatten()).sum() / len(y)
                print(f'sklearn acc: {acc}')
                sklearn_conf_matrix = confusion_matrix(y, pred)
                print(f'sklearn confusion matrix: \n {sklearn_conf_matrix}')
                result_store['models'].update_row({ 
                    'sklearn': sklearn_, 
                    'sklearn_acc': acc, 
                    'sklearn_conf_matrix': sklearn_conf_matrix,
                })

                trunc_sklearn = LogisticRegression(penalty='none', fit_intercept=True)
                trunc_sklearn.fit(x_trunc, y_trunc.flatten())
                trunc_sklearn_ = ch.from_numpy(np.concatenate([trunc_sklearn.coef_.flatten(), trunc_sklearn.intercept_]))
                print(f'trunc sklearn: {trunc_sklearn_}')
                trunc_sklearn_cos_sim = float(cos_sim(trunc_sklearn_[None,...], sklearn_[None,...]))
                pred = trunc_sklearn.predict(X_)
                trunc_sklearn_acc = np.equal(pred, y.flatten()).sum() / len(y)
                print(f'trunc sklearn acc: {trunc_sklearn_acc}')
                print(f'trunc sklearn cos sim: {trunc_sklearn_cos_sim}')
                trunc_sklearn_conf_matrix = confusion_matrix(y, pred)
                print(f'trunc sklearn confusion matrix: \n {trunc_sklearn_conf_matrix}')
                result_store['models'].update_row({ 
                    'trunc_sklearn': trunc_sklearn_,
                    'trunc_sklearn_acc': trunc_sklearn_acc,
                    'trunc_sklearn_conf_matrix': trunc_sklearn_conf_matrix, 
                    'trunc_sklearn_cos_sim': trunc_sklearn_cos_sim,
                })
                
                id_ = oracle.Identity()
                train_kwargs = Parameters({'phi': id_,
                                    'alpha': alpha,
                                    'fit_intercept': True, 
                                    'momentum': 0.0,
                                    'normalize': False, 
                                    'batch_size': 100,
                                    'epochs': 30,
                                    'trials': 3,
                                    'verbose': True,
                                    'early_stopping': True, 
                                    'num_samples': 100})
                ch.manual_seed(seed)
                trunc_log_reg = stats.TruncatedLogisticRegression(train_kwargs)
                trunc_log_reg.fit(x_trunc, y_trunc) 
                trunc_log_reg_ = ch.cat([trunc_log_reg.coef_.flatten(), trunc_log_reg.intercept_])
                print(f'id trunc log reg: {trunc_log_reg_}')
                trunc_log_reg_cos_sim = float(cos_sim(trunc_log_reg_[None,...], sklearn_[None,...]))
                trunc_log_reg_pred = trunc_log_reg.predict(X)
                trunc_log_reg_acc = trunc_log_reg_pred.eq(y).sum() / len(y)
                print(f'id trunc log reg accuracy: {trunc_log_reg_acc}')
                print(f'id trunc cos sim: {trunc_log_reg_cos_sim}')
                # self.assertTrue(trunc_cos_sim >= .8, f'trunc cos sim: {trunc_cos_sim}')
                trunc_log_reg_conf_matrix = confusion_matrix(y, trunc_log_reg_pred)
                print(f'id trunc log reg confusion matrix: \n {trunc_sklearn_conf_matrix}')
                result_store['models'].update_row({ 
                    'id_trunc_log_reg': trunc_log_reg_, 
                    'id_trunc_log_reg_acc': trunc_log_reg_acc,
                    'id_trunc_log_reg_conf_matrix': trunc_log_reg_conf_matrix, 
                    'id_trunc_log_reg_cos_sim': trunc_log_reg_cos_sim,
                })

                train_kwargs = Parameters({'phi': id_,
                                    'alpha': alpha,
                                    'fit_intercept': True, 
                                    'normalize': False, 
                                    'batch_size': 100,
                                    'epochs': 30,
                                    'momentum': 0.0,
                                    'trials': 3,
                                    'multi_class': 'multinomial', 
                                    'verbose': True,
                                    'early_stopping': True,
                                    'num_samples': 100})
                ch.manual_seed(seed)
                trunc_multi_log_reg = stats.TruncatedLogisticRegression(train_kwargs)
                trunc_multi_log_reg.fit(x_trunc, y_trunc.flatten().long()) 
                trunc_multi_log_reg_diff = ch.cat([trunc_multi_log_reg.coef_[1] - trunc_multi_log_reg.coef_[0], 
                (trunc_multi_log_reg.intercept_[1] - trunc_multi_log_reg.intercept_[0])[...,None]])
                trunc_multi_log_reg_ = ch.cat([trunc_multi_log_reg.coef_, trunc_multi_log_reg.intercept_[...,None]], axis=1)
                trunc_multi_cos_sim = float(cos_sim(trunc_multi_log_reg_diff[None,...], sklearn_[None,...]))
                trunc_multi_log_reg_pred = trunc_multi_log_reg.predict(X_)
                trunc_multi_log_reg_acc = trunc_multi_log_reg_pred.eq(y.flatten()).sum() / len(y)
                print(f'id trunc multi log reg accuracy: {trunc_multi_log_reg_acc}')
                print(f'id trunc multi cos sim: {trunc_multi_cos_sim}')
                # self.assertTrue(trunc_cos_sim >= .8, f'trunc multi cos sim: {trunc_multi_cos_sim}')
                trunc_multi_log_reg_conf_matrix = confusion_matrix(y, trunc_multi_log_reg_pred)
                print(f'id trunc multi log reg confusion matrix: \n {trunc_multi_log_reg_conf_matrix}')
                result_store['models'].update_row({ 
                    'id_trunc_multi_log_reg': trunc_multi_log_reg_,
                    'id_trunc_multi_log_reg_diff': trunc_multi_log_reg_diff,   
                    'id_trunc_multi_log_reg_acc': trunc_multi_log_reg_acc, 
                    'id_trunc_multi_log_reg_cos_sim': trunc_multi_cos_sim, 
                    'id_trunc_multi_log_reg_conf_matrix': trunc_multi_log_reg_conf_matrix,
                })

                train_kwargs = Parameters({'phi': phi,
                                    'alpha': alpha,
                                    'fit_intercept': True, 
                                    'momentum': 0.0,
                                    'normalize': False, 
                                    'batch_size': 100,
                                    'epochs': 30,
                                    'trials': 3,
                                    'verbose': True,
                                    'early_stopping': True, 
                                    'num_samples': 100})
                ch.manual_seed(seed)
                size = trunc_log_reg_[None,...].size()
                print(f'trunc log reg size: {size}')
                trunc_log_reg = stats.TruncatedLogisticRegression(train_kwargs, weight=trunc_log_reg_[None,...])
                trunc_log_reg.fit(x_trunc, y_trunc) 
                trunc_log_reg_ = ch.cat([trunc_log_reg.coef_.flatten(), trunc_log_reg.intercept_])
                print(f'trunc log reg: {trunc_log_reg_}')
                trunc_log_reg_cos_sim = float(cos_sim(trunc_log_reg_[None,...], sklearn_[None,...]))
                trunc_log_reg_pred = trunc_log_reg.predict(X)
                trunc_log_reg_acc = trunc_log_reg_pred.eq(y).sum() / len(y)
                print(f'trunc log reg accuracy: {trunc_log_reg_acc}')
                print(f'trunc log reg cos sim: {trunc_log_reg_cos_sim}')
                # self.assertTrue(trunc_cos_sim >= .8, f'trunc cos sim: {trunc_cos_sim}')
                trunc_log_reg_conf_matrix = confusion_matrix(y, trunc_log_reg_pred)
                print(f'trunc log reg confusion matrix: \n {trunc_sklearn_conf_matrix}')
                result_store['models'].update_row({ 
                    'trunc_log_reg': trunc_sklearn_, 
                    'trunc_log_reg_acc': trunc_log_reg_acc,
                    'trunc_log_reg_conf_matrix': trunc_log_reg_conf_matrix, 
                    'trunc_log_reg_cos_sim': trunc_log_reg_cos_sim,
                })

                train_kwargs = Parameters({'phi': phi_gumbel,
                                    'alpha': alpha,
                                    'fit_intercept': True, 
                                    'normalize': False, 
                                    'batch_size': 100,
                                    'epochs': 30,
                                    'momentum': 0.0,
                                    'trials': 3,
                                    'multi_class': 'multinomial', 
                                    'verbose': True,
                                    'early_stopping': True,
                                    'num_samples': 100})
                ch.manual_seed(seed)
                trunc_multi_log_reg = stats.TruncatedLogisticRegression(train_kwargs, weight=trunc_multi_log_reg_.T)
                trunc_multi_log_reg.fit(x_trunc, y_trunc.flatten().long()) 
                trunc_multi_log_reg_diff = ch.cat([trunc_multi_log_reg.coef_[1] - trunc_multi_log_reg.coef_[0], 
                (trunc_multi_log_reg.intercept_[1] - trunc_multi_log_reg.intercept_[0])[...,None]])
                trunc_multi_log_reg_ = ch.cat([trunc_multi_log_reg.coef_, trunc_multi_log_reg.intercept_[...,None]], axis=1)
                trunc_multi_cos_sim = float(cos_sim(trunc_multi_log_reg_diff[None,...], sklearn_[None,...]))
                trunc_multi_log_reg_pred = trunc_multi_log_reg.predict(X_)
                trunc_multi_log_reg_acc = trunc_multi_log_reg_pred.eq(y.flatten()).sum() / len(y)
                print(f'trunc multi log reg accuracy: {trunc_multi_log_reg_acc}')
                print(f'trunc multi cos sim: {trunc_multi_cos_sim}')
                # self.assertTrue(trunc_cos_sim >= .8, f'trunc multi cos sim: {trunc_multi_cos_sim}')
                trunc_multi_log_reg_conf_matrix = confusion_matrix(y, trunc_multi_log_reg_pred)
                print(f'trunc multi log reg confusion matrix: \n {trunc_multi_log_reg_conf_matrix}')
                result_store['models'].update_row({ 
                    'trunc_multi_log_reg': trunc_multi_log_reg_,
                    'trunc_multi_log_reg_diff': trunc_multi_log_reg_diff,   
                    'trunc_multi_log_reg_acc': trunc_multi_log_reg_acc, 
                    'trunc_multi_log_reg_cos_sim': trunc_multi_cos_sim, 
                    'trunc_multi_log_reg_conf_matrix': trunc_multi_log_reg_conf_matrix,
                })
                
                train_kwargs = Parameters({
                                    'alpha': alpha,
                                    'batch_size': 100,
                                    'momentum': 0.0,
                                    'epochs': 30,
                                    'trials': 3,
                                    'verbose': True,
                                    'early_stopping': True,
                                    'workers': 0,
                                    'num_samples': 100})        
                ch.manual_seed(seed)
                gumbel_model = GumbelCEModel(train_kwargs, X.size(1) + 1, len(y.unique()))
                trainer = Trainer(gumbel_model, train_kwargs)
                x_trunc_ones = ch.cat([x_trunc, ch.ones(x_trunc.size(0), 1)], axis=1)
                X_ones = ch.cat([X_, ch.ones(X_.size(0), 1)], axis=1)
                train_loader, val_loader = make_train_and_val(train_kwargs, x_trunc_ones, y_trunc.flatten().long())
                trainer.train_model((train_loader, val_loader))
                gumbel_diff = gumbel_model.model[1] - gumbel_model.model[0]
                gumbel_ = gumbel_model.model                
                gumbel_cos_sim = float(cos_sim(gumbel_diff[None,...], sklearn_[None,...]))
                gumbel_pred = gumbel_model.predict(X_ones)
                gumbel_acc = gumbel_pred.eq(y.flatten()).sum() / len(y)
                print(f'gumbel accuracy: {gumbel_acc}')
                print(f'gumbel cos sim: {gumbel_cos_sim}')
                gumbel_conf_matrix = confusion_matrix(y, gumbel_pred)
                print(f'trunc gumbel confusion matrix: \n {gumbel_conf_matrix}')
                result_store['models'].update_row({ 
                    'gumbel_diff': gumbel_diff,
                    'gumbel': gumbel_, 
                    'gumbel_acc': gumbel_acc, 
                    'gumbel_cos_sim': gumbel_cos_sim, 
                    'gumbel_conf_matrix': gumbel_conf_matrix, 
                })

                train_kwargs = Parameters({
                                    'alpha': alpha,
                                    'batch_size': 100,
                                    'epochs': 30,
                                    'momentum': 0.0,
                                    'trials': 3,
                                    'verbose': True,
                                    'early_stopping': True,
                                    'workers': 0})        
                ch.manual_seed(seed)
                softmax_model = SoftmaxModel(train_kwargs, X.size(1) + 1, len(y.unique()))
                trainer = Trainer(softmax_model, train_kwargs) 
                train_loader, val_loader = make_train_and_val(train_kwargs, x_trunc_ones, y_trunc.flatten().long())
                trainer.train_model((train_loader, val_loader))
                softmax_diff = softmax_model.model[1] - softmax_model.model[0]
                softmax_ = softmax_model.model            
                
                softmax_cos_sim = float(cos_sim(softmax_diff[None,...], sklearn_[None,...]))
                softmax_pred = softmax_model.predict(X_ones)
                softmax_acc = softmax_pred.eq(y.flatten()).sum() / len(y)
                print(f'softmax accuracy: {softmax_acc}')
                print(f'softmax cos sim: {softmax_cos_sim}')
                softmax_conf_matrix = confusion_matrix(y, softmax_pred)
                print(f'softmax confusion matrix: \n {softmax_conf_matrix}')
                result_store['models'].update_row({ 
                    'softmax_diff': softmax_diff, 
                    'softmax': softmax_,
                    'softmax_acc': softmax_acc, 
                    'softmax_cos_sim': softmax_cos_sim, 
                    'softmax_conf_matrix': softmax_conf_matrix,
                })

                result_store['models'].flush_row()
                print('===================================================\n')
        result_store.close()

if __name__ == '__main__':
    unittest.main()
