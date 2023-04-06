import torch as ch
from typing import Callable

from delphi.stats import TruncatedLinearRegression
from delphi.utils.helpers import Parameters


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
        if u_t is None: u_t = ch.randn((1, self.B.size(0)))
        y_t = x_t@self.A + u_t@self.B + self.M.sample((x_t.size(0),))
        if self.phi(y_t): 
            return y_t, u_t
        else: 
            return None



def calc_sarah_dean(train_kwargs: Parameters, 
                    gen_data: Callable,
                    num_traj: int,
                    D: int, 
                    M: int,
                    rand_seed: int=0):
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
            xt = ch.zeros((1, D))
        total_samples += 1

    feat_concat = ch.cat([X, U], axis=1)

    train_kwargs.__setattr__('noise_var', gen_data.noise_var)

    trunc_lds = TruncatedLinearRegression(train_kwargs,
                                            dependent=True, 
                                            rand_seed=rand_seed)
    trunc_lds.fit(feat_concat.detach(), Y.detach())


    AB = trunc_lds.best_coef_
    A_, B_ = AB[:D], AB[D:]

    ols_ = trunc_lds.emp_weight
    A_ols, B_ols = ols_[:D], ols_[D:]

    return A_, B_, A_ols, B_ols 