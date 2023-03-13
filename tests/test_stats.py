# stats tests 
import unittest
import numpy as np
import torch as ch
from torch import Tensor
from torch.distributions import Uniform, Gumbel
from torch.nn import CosineSimilarity
from torch.nn import MSELoss
import torch.linalg as LA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import confusion_matrix
from typing import Callable

from delphi import stats 
from delphi import oracle
from delphi.utils.helpers import Parameters, logistic, calc_spectral_norm
from delphi.utils.datasets import make_train_and_val

# CONSTANTS
mse_loss =  MSELoss()
cos_sim = CosineSimilarity()
G = Gumbel(0, 1)
seed = 69 

class TestStats(unittest.TestCase): 
    """
    Test suite for the stats module.
    """
    # left truncated linear regression
    def test_known_truncated_regression(self):
        D, K = 100, 1
        SAMPLES = 10000
        w_ = Uniform(-1, 1)
        M = Uniform(-10, 10)
        # generate ground truth
        NOISE_VAR = 20*ch.ones(1, 1)
        W = w_.sample([K, D])
        W0 = w_.sample([1, 1])

        print(f"gt weight: {W}")
        print(f"gt intercept: {W0}")
        print(f"gt noise var: {NOISE_VAR}")
        # generate data
        X = M.sample(ch.Size([SAMPLES, D])) if isinstance(M, Uniform) else M.sample(ch.Size([SAMPLES]))
        y = X@W.T + W0 
        noised = y + ch.sqrt(NOISE_VAR) * ch.randn(y.size(0), 1)
        # generate ground-truth data
        phi = oracle.Left_Regression(ch.zeros(1))
        # truncate
        indices = phi(noised).nonzero()[:,0]
        x_trunc, y_trunc = X[indices], noised[indices]
        alpha = x_trunc.size(0) / X.size(0)
        print(f'alpha: {alpha}')

        gt_norm = LinearRegression()
        gt_norm.fit(X, y)
        gt_ = ch.from_numpy(np.concatenate([gt_norm.coef_.flatten(), gt_norm.intercept_]))

        # calculate empirical noise variance for regression 
        ols_trunc = LinearRegression()
        ols_trunc.fit(x_trunc, y_trunc)
        emp_ = ch.from_numpy(np.concatenate([ols_trunc.coef_.flatten(), ols_trunc.intercept_]))
        print(f'empirical weights: {emp_}')
        emp_mse_loss = mse_loss(emp_, gt_)
        print(f'emp mse loss: {emp_mse_loss}')

        # scale y features
        y_trunc_scale = y_trunc / ch.sqrt(NOISE_VAR)
        phi_scale = oracle.Left_Regression(phi.left / ch.sqrt(NOISE_VAR))
        # train algorithm
        train_kwargs = Parameters({'phi': phi_scale, 
                                'alpha': alpha,
                                'epochs': 10,
                                'lr': 5e-1,
                                'num_samples': 10,
                                # 'momentum': .9,
                                'batch_size': 1,
                                'trials': 1,
                                'constant': True,
                                'noise_var': ch.ones(1, 1)}) 
        trunc_reg = stats.TruncatedLinearRegression(train_kwargs)
        trunc_reg.fit(x_trunc, y_trunc_scale)
        w_ = ch.cat([(trunc_reg.best_coef_).flatten(), trunc_reg.best_intercept_]) * ch.sqrt(NOISE_VAR)
        print(f'estimated weights: {w_}')
        known_mse_loss = mse_loss(gt_, w_.flatten())
        print(f'known mse loss: {known_mse_loss}')
        msg = f'known mse loss is larger than empirical mse loss. known mse loss is {known_mse_loss}, and empirical mse loss is: {emp_mse_loss}'
        self.assertTrue(known_mse_loss <= emp_mse_loss, msg)
        
        avg_w_ = ch.cat([(trunc_reg.avg_coef_).flatten(), trunc_reg.avg_intercept_]) * ch.sqrt(NOISE_VAR)
        avg_known_mse_loss = mse_loss(gt_, avg_w_.flatten())
        print(f'avg known mse loss: {avg_known_mse_loss}')
        msg = f'avg known mse loss is larger than empirical mse loss. avg known mse loss is {avg_known_mse_loss}, and empirical mse loss is: {emp_mse_loss}'
        self.assertTrue(avg_known_mse_loss <= emp_mse_loss, msg)
    
    def test_unknown_truncated_regression(self):
        D, K = 10, 1
        SAMPLES = 1000
        w_ = Uniform(-1, 1)
        M = Uniform(-10, 10)
        # generate ground truth
        noise_var = 10*ch.ones(1, 1)
        W = w_.sample([K, D])
        W0 = w_.sample([1, 1])

        print(f"gt weight: {W}")
        print(f"gt intercept: {W0}")
        print(f"gt noise var: {noise_var}")
        # generate data
        X = M.sample(ch.Size([SAMPLES, D])) if isinstance(M, Uniform) else M.sample(ch.Size([SAMPLES]))
        y = X@W.T + W0 
        noised = y + ch.sqrt(noise_var) * ch.randn(y.size(0), 1)
        # generate ground-truth data
        phi = oracle.Left_Regression(ch.zeros(1))
        # truncate
        indices = phi(noised).nonzero()[:,0]
        x_trunc, y_trunc = X[indices], noised[indices]
        alpha = x_trunc.size(0) / X.size(0)
        print(f'alpha: {alpha}')

        gt_norm = LinearRegression()
        gt_norm.fit(X, noised)
        gt_ = ch.from_numpy(np.concatenate([gt_norm.coef_.flatten(), gt_norm.intercept_]))

        # calculate empirical noise variance for regression 
        ols_trunc = LinearRegression()
        ols_trunc.fit(x_trunc, y_trunc)
        emp_noise_var = ch.from_numpy(ols_trunc.predict(X) - noised.numpy()).var(0)
        emp_ = ch.from_numpy(np.concatenate([ols_trunc.coef_.flatten(), ols_trunc.intercept_]))
        emp_mse_loss = mse_loss(emp_, gt_)
        emp_var_l1 = float(ch.abs(emp_noise_var - noise_var))
        print(f'emp mse loss: {emp_mse_loss}')
        print(f'emp noise var l1: {emp_var_l1}')

        # scale y features by empirical noise variance
        y_trunc_emp_scale = y_trunc / ch.sqrt(emp_noise_var)
        phi_emp_scale = oracle.Left_Regression(phi.left / ch.sqrt(emp_noise_var))
        # train algorithm
        train_kwargs = Parameters({'phi': phi_emp_scale, 
                                'alpha': alpha,
                                'trials': 1,
                                'momentum': .9,
                                'batch_size': 10,
                                'var_lr': 1e-2,})
        unknown_trunc_reg = stats.TruncatedLinearRegression(train_kwargs)
        unknown_trunc_reg.fit(x_trunc.repeat(100, 1), y_trunc_emp_scale.repeat(100, 1))
        w_ = ch.cat([(unknown_trunc_reg.best_coef_).flatten(), unknown_trunc_reg.best_intercept_]) * ch.sqrt(emp_noise_var)
        noise_var_ = unknown_trunc_reg.variance_ * emp_noise_var
        unknown_mse_loss = mse_loss(gt_, w_.flatten())
        print(f'unknown mse loss: {unknown_mse_loss}')
        unknown_var_l1 = float(ch.abs(noise_var - noise_var_))
        print(f'unknown var l1: {unknown_var_l1}')
        self.assertTrue(unknown_mse_loss <= emp_mse_loss, f'unknown mse loss: {unknown_mse_loss}')
        self.assertTrue(unknown_var_l1 <= emp_var_l1, f'unknown var l1: {unknown_var_l1}')

    def test_truncated_dependent_regression(self): 
        D = 3 # number of dimensions for A_{*} matrix
        T = 10000 # uncensored system trajectory length 

        spectral_norm = float('inf')
        while spectral_norm > 1.0: 
            A = .25 * ch.randn((D, D))
            spectral_norm = calc_spectral_norm(A)


        A = .25 * ch.eye(D)
        spectral_norm = calc_spectral_norm(A)
        print(f'A spectral norm: {calc_spectral_norm(A)}')

        phi = oracle.LogitBall(1.5)

        X, Y = ch.Tensor([]), ch.Tensor([])
        NOISE_VAR = ch.eye(D)
        M = ch.distributions.MultivariateNormal(ch.zeros(D), NOISE_VAR) 
        x_t = ch.zeros((1, D))
        for i in range(T): 
            noise = M.sample()
            y_t = (A@x_t.T).T + noise
            if phi(y_t): # returns a boolean 
                X = ch.cat([X, x_t])
                Y = ch.cat([Y, y_t])
            x_t = y_t

        alpha = X.size(0) / T

        train_kwargs = Parameters({
            'phi': phi, 
            'c_eta': 1.0,
            'epochs': 1, 
            'trials': 1, 
            'batch_size': 1,
            'num_samples': 100,
            'T': X.size(0),
            'trials': 1,
            'c_s': 10.0,
            'alpha': alpha,
            'tol': 1e-1,
            'noise_var': NOISE_VAR, 
        })
        trunc_lds = stats.TruncatedLinearRegression(train_kwargs, 
                                                    dependent=True, 
                                                    rand_seed=seed)
        trunc_lds.fit(X, Y)
        
        A_ = trunc_lds.best_coef_.T
        A0_ = trunc_lds.ols_coef_.T
        A_avg = trunc_lds.avg_coef_.T
        trunc_spec_norm = calc_spectral_norm(A - A_)
        emp_spec_norm = calc_spectral_norm(A - A0_)
        avg_trunc_spec_norm = calc_spectral_norm(A - A_avg)

        print(f'alpha: {alpha}')
        print(f'A spectral norm: {spectral_norm}')
        print(f'truncated spectral norm: {trunc_spec_norm}')
        print(f'average truncated spectral norm: {avg_trunc_spec_norm}')
        print(f'ols spectral norm: {emp_spec_norm}')

        self.assertTrue(trunc_spec_norm <= emp_spec_norm, f"truncated spectral norm {trunc_spec_norm}, while OLS spectral norm is: {emp_spec_norm}")
        self.assertTrue(avg_trunc_spec_norm <= emp_spec_norm, f"average truncated spectral norm {avg_trunc_spec_norm}, while OLS spectral norm is: {emp_spec_norm}")

    def test_truncated_lqr(self): 

        class GenerateTruncatedLQRData:
            def __init__(self, phi, A, B, noise_var = None): 
                self.phi = phi
                self.A = A 
                self.B = B
                self.noise_var = noise_var
                if noise_var is None: 
                    self.noise_var = ch.eye(self.A.size(0))

                self.M = ch.distributions.MultivariateNormal(ch.zeros(A.size(0)), self.noise_var)

            def __call__(self, x_t, u_t = None):
                if u_t is None: u_t = ch.randn((1, self.B.size(1)))
                y_t = (self.A@x_t.T + self.B@u_t.T).T + self.M.sample((x_t.size(0),))
                if self.phi(y_t): 
                    return y_t, u_t
                else: 
                    return None


        def calc_sarah_dean(train_kwargs: Parameters, 
                            gen_data: Callable,
                            num_traj: int,
                            D: int, 
                            M: int):
            '''
            Sarah dean LQR used for testing.
            '''
            number_trajectories = 1
            num_samples = 0
            X, Y, U = ch.Tensor([]), ch.Tensor([]), ch.Tensor([])

            xt = ch.zeros((1, D)) 
            total_samples = 0

            while number_trajectories < num_traj:
                sample = gen_data(xt)

                if sample is not None: 
                    yt, ut = sample 
                    X, Y, U = ch.cat((X,xt)), ch.cat((Y,yt)), ch.cat((U,ut))

                    xt = yt 
                    num_samples += 1
                else: 
                    number_trajectories += 1
                    xt = ch.zeros((1, A.size(0)))
                total_samples += 1

            feat_concat = ch.cat([X, U], axis=1)

            alpha = X.size(0) / total_samples

            train_kwargs.__setattr__('alpha', alpha)
            train_kwargs.__setattr__('noise_var', gen_data.noise_var)
            train_kwargs.__setattr__('b', True)

            lr = (1/train_kwargs.alpha) ** train_kwargs.c_gamma
            train_kwargs.__setattr__('lr', lr)

            trunc_lds = stats.TruncatedLinearRegression(train_kwargs,
                                                dependent=True)
            trunc_lds.fit(feat_concat.detach(), Y.detach())


            AB = trunc_lds.best_coef_
            A_, B_ = AB[:D], AB[D:]

            ols_ = trunc_lds.emp_weight
            A_ols, B_ols = ols_[:D], ols_[D:]

            return A_, B_, A_ols, B_ols 

        # This is for the exploration radius for first 2.
        gamma = 2.0 

        # Here, m >= d must hold
        D = 2
        M = 3
        assert M >= D, f'M is currently: {M}, but it needs to be greater than or equal to D: {D}'

        NOISE_VAR = ch.eye(D)

        # I design A as a random matrix, which the eigen values at range [0.5,2]
        fake_A = ch.randn((D, D))
        u, s, v = LA.svd(fake_A)
        s2 = ch.diag(ch.rand(D)*2.5+0.5)
        A = u@s2@v
        # I design B as a random matrix, which the eigen values also at range [0.5,2]
        fake_B = ch.randn((D, M))
        u, s, v = LA.svd(fake_B)
        s2 = ch.zeros((D, M))
        s2[:D, :D] = ch.diag(ch.rand(D)*2.5+0.5)
        B = u@s2@v
        U_A = 3.0
        U_B = 3.0
        L_B = 0.5
        R = 5.0

        TRAIN_KWARGS = Parameters({
            'c_gamma': 2.0,
            'fit_intercept': False,
            'epochs': 10, 
            'trials': 1, 
            'batch_size': 10,
            'num_samples': 10,
            'tol': 1e-2,
            'R': R, 
            'U_A': U_A, 
            'U_B': U_B,
            'delta': .9, 
            'gamma': gamma, 
            'repeat': 1,
            'T_gen_samples_A': 1000,
            'T_gen_samples_B': 1000,
            'target_thickness': 2.0*U_A*U_A*max(U_A,U_B)/L_B
        })

        # membership oracle
        phi = oracle.LogitBall(R)
        gen_data = GenerateTruncatedLQRData(phi, A, B, noise_var=NOISE_VAR)

        TRAIN_KWARGS.__setattr__('phi', phi)

        trunc_lqr = stats.truncated_lqr.TruncatedLQR(TRAIN_KWARGS, gen_data, D, M)
        trunc_lqr.fit()

        A_yao, B_yao = trunc_lqr.best_A_.T, trunc_lqr.best_B_.T
        A_sarah_dean_plev, B_sarah_dean_plev, A_sarah_dean_ols, B_sarah_dean_ols = calc_sarah_dean(TRAIN_KWARGS, gen_data, 1000, D, M)

        A_yao_spec_norm = stats.truncated_lqr.calc_spectral_norm(A_yao - A)
        B_yao_spec_norm = stats.truncated_lqr.calc_spectral_norm(B_yao - B)

        A_sd_plevr_spec_norm = stats.truncated_lqr.calc_spectral_norm(A_sarah_dean_plev.T - A)
        B_sd_plevr_spec_norm = stats.truncated_lqr.calc_spectral_norm(B_sarah_dean_plev.T - B)

        A_sd_ols_spec_norm = stats.truncated_lqr.calc_spectral_norm(A_sarah_dean_ols.T - A)
        B_sd_ols_spec_norm = stats.truncated_lqr.calc_spectral_norm(B_sarah_dean_ols.T - B)
        self.assertTrue(A_yao_spec_norm < A_sd_ols_spec_norm, f"A yao spectral norm is: {A_yao_spec_norm}, and A sarah dean ols spectral norm is: {A_sd_ols_spec_norm}")
        self.assertTrue(B_yao_spec_norm < B_sd_ols_spec_norm, f"B yao spectral norm is: {B_yao_spec_norm}, and B sarah dean ols spectral norm is: {B_sd_ols_spec_norm}")
       
        self.assertTrue(A_yao_spec_norm < A_sd_plevr_spec_norm, f"A yao spectral norm is: {A_yao_spec_norm}, and A sarah dean plevrakis spectral norm is: {A_sd_plevr_spec_norm}")
        self.assertTrue(B_yao_spec_norm < B_sd_plevr_spec_norm, f"B yao spectral norm is: {B_yao_spec_norm}, and B sarah dean plevrakis spectral norm is: {B_sd_plevr_spec_norm}")

    # left truncated binary classification task
    def test_truncated_logistic_regression(self):
        d, k = 5, 1
        C = -1.0
        # ground-truth logistic regression model 
        gt = ch.nn.Linear(in_features=d, out_features=k, bias=True)
        W = Uniform(-1, 1)
        U = Uniform(-5, 5)
        gt.weight = ch.nn.Parameter(W.sample([k, d]))
        gt.bias = ch.nn.Parameter(W.sample([1, k]))
        gt_ = ch.cat([gt.weight.flatten(), gt.bias.flatten()])

        # input features
        X = U.sample([10000, d])
        # latent variables
        z = gt(X) + logistic.sample([X.size(0), 1])
        # classification
        y = (z > 0).float()

        # generate ground-truth data
        phi = oracle.Left_Regression(Tensor([C]))
        phi_gumbel = oracle.GumbelLogisticLeftTruncation(Tensor([C]))
        indices = phi(z).nonzero(as_tuple=True)
        x_trunc = X[indices[0]]
        y_trunc = y[indices][...,None]
        alpha = x_trunc.size(0) / X.size(0)
        print(f'C: {C}')
        print(f'alpha: {alpha}')
                
        sklearn = LogisticRegression(penalty='none', fit_intercept=True)
        sklearn.fit(X, y.flatten())
        sklearn_ = ch.from_numpy(np.concatenate([sklearn.coef_.flatten(), sklearn.intercept_]))
        print(f'sklearn: {sklearn_}')
        pred = sklearn.predict(X)
        acc = np.equal(pred, y.flatten()).sum() / len(y)
        print(f'sklearn acc: {acc}')
        sklearn_conf_matrix = confusion_matrix(y, pred)
        print(f'sklearn confusion matrix: \n {sklearn_conf_matrix}')
        
        trunc_sklearn = LogisticRegression(penalty='none', fit_intercept=True)
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
        trunc_log_reg_conf_matrix = confusion_matrix(y, trunc_log_reg_pred)
        print(f'id trunc log reg confusion matrix: \n {trunc_log_reg_conf_matrix}')
        
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
        trunc_multi_log_reg_pred = trunc_multi_log_reg.predict(X)
        trunc_multi_log_reg_acc = trunc_multi_log_reg_pred.eq(y.flatten()).sum() / len(y)
        print(f'id trunc multi log reg accuracy: {trunc_multi_log_reg_acc}')
        print(f'id trunc multi cos sim: {trunc_multi_cos_sim}')
        trunc_multi_log_reg_conf_matrix = confusion_matrix(y, trunc_multi_log_reg_pred)
        print(f'id trunc multi log reg confusion matrix: \n {trunc_multi_log_reg_conf_matrix}')
        
        train_kwargs = Parameters({'phi': phi,
                            'alpha': alpha,
                            'fit_intercept': True, 
                            'momentum': 0.5,
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
        self.assertTrue(trunc_log_reg_cos_sim >= .9, f'trunc log reg cos sim: {trunc_log_reg_cos_sim}')
        trunc_log_reg_conf_matrix = confusion_matrix(y, trunc_log_reg_pred)
        print(f'trunc log reg confusion matrix: \n {trunc_sklearn_conf_matrix}')
        
        train_kwargs = Parameters({'phi': phi_gumbel,
                            'alpha': alpha,
                            'fit_intercept': True, 
                            'normalize': False, 
                            'batch_size': 100,
                            'epochs': 30,
                            'momentum': 0.5,
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
        self.assertTrue(trunc_multi_cos_sim >= .9, f'trunc multi cos sim: {trunc_multi_cos_sim}')
        trunc_multi_log_reg_pred = trunc_multi_log_reg.predict(X)
        trunc_multi_log_reg_acc = trunc_multi_log_reg_pred.eq(y.flatten()).sum() / len(y)
        print(f'trunc multi log reg accuracy: {trunc_multi_log_reg_acc}')
        print(f'trunc multi cos sim: {trunc_multi_cos_sim}')
        trunc_multi_log_reg_conf_matrix = confusion_matrix(y, trunc_multi_log_reg_pred)
        print(f'trunc multi log reg confusion matrix: \n {trunc_multi_log_reg_conf_matrix}')

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
        

if __name__ == '__main__':
    unittest.main()
