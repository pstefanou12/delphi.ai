import torch.linalg as LA
import torch as ch
import numpy as np
from typing import Callable

from .truncated_linear_regression import TruncatedLinearRegression
from ..utils.helpers import Parameters

# HELPER FUNCTIONS
def calc_spectral_norm(A):
    u, s, v = LA.svd(A)
    return s.max()

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

# Return the thickness of a positive semi-definite matrix
def calc_thickness(X):
  return LA.eig(X).eigenvalues.real.min()


def phase_one(train_kwargs: Parameters,
              gen_data: Callable, 
              D: int, 
              M: int,
              target_thickness: float=float('inf'),
              num_traj: int = float('inf'), 
              T: int=float('inf')):
    '''
    Cold start phase 1. Initial estimation for A.
    Args: 
      num_traj: number of trajectories allowed
    Returns: ols estimate, initial estimation with plevrakis, and 
      number of trajectories taken
    '''
    assert target_thickness != float('inf') or num_traj != float('inf') or T != float('inf'), f"all stopping conditions are {float('inf')}, need to provide at least one stopping variable: (T, num_traj, target_thickness), that isn't infinity"
    num_trajectories = 1
    total_samples = 0
    x_t = ch.zeros((1, D))
    X, Y, U = ch.Tensor([]), ch.Tensor([]), ch.Tensor([])
    covariate_matrix = ch.zeros([D,D])

    while num_trajectories < num_traj and X.size(0) < T and calc_thickness(covariate_matrix) < target_thickness:
        sample = gen_data(x_t, u_t=ch.zeros((1, M)))
        total_samples += 1
        if sample is not None:
            y_t, u_t = sample
            X, Y, U = ch.cat((X,x_t)), ch.cat((Y,y_t)), ch.cat((U, u_t))
            covariate_matrix += x_t.T@x_t
            x_t = y_t
        else:
            x_t = ch.zeros((1, D))
            num_trajectories += 1

        if total_samples % 100 == 0: 
            print(f'total number of samples: {total_samples}')

    alpha = X.size(0) / total_samples

    train_kwargs.__setattr__('alpha', alpha)
    train_kwargs.__setattr__('noise_var', gen_data.noise_var)
    train_kwargs.__setattr__('b', False)


    lr = (1/train_kwargs.alpha) ** train_kwargs.c_gamma
    train_kwargs.__setattr__('lr', lr)

    trunc_lds = TruncatedLinearRegression(train_kwargs, 
                                         dependent=True)
    trunc_lds.fit(X, Y)

    return trunc_lds.emp_weight, trunc_lds.avg_coef_, trunc_lds.best_coef_, num_trajectories, X.size(0)


def phase_two(train_kwargs: Parameters, 
              gen_data: Callable,
              D: int, 
              M: int,
              target_thickness: float=float('inf'), 
              num_traj: int=float('inf'), 
              T: int=float('inf')): 
    '''
    Cold start phase 2. Initial estimation for B.
    '''
    assert target_thickness != float('inf') or num_traj != float('inf') or T != float('inf'), f"all stopping conditions are {float('inf')}, need to provide at least one stopping variable: (T, num_traj, target_thickness), that isn't infinity"

    total_samples = 0
    index = 0
    xt = ch.zeros((1, D))
    id_ = ch.eye(M)
    U, Y = ch.Tensor([]), ch.Tensor([])
    covariate_matrix = ch.zeros([M,M])
    # gamma = R/U_B
    gamma = 2.5

    while total_samples < num_traj and U.size(0) < T and calc_thickness(covariate_matrix) < target_thickness:
        u = (gamma*id_[index])[None,...]
        sample = gen_data(xt, u_t=u)
        
        total_samples += 1
        if sample is not None: 
            y, u = sample
            index = (index+1)%M
            U, Y = ch.cat([U, u]), ch.cat([Y, y])
            covariate_matrix += u.T@u

    alpha = U.size(0) / total_samples

    train_kwargs.__setattr__('alpha', alpha)
    train_kwargs.__setattr__('noise_var', gen_data.noise_var)
    train_kwargs.__setattr__('b', True)

    lr = (1/train_kwargs.alpha) ** train_kwargs.c_gamma
    train_kwargs.__setattr__('lr', lr)

    trunc_lds = TruncatedLinearRegression(train_kwargs, 
                                         dependent=True)
    trunc_lds.fit(U, Y)
    return trunc_lds.ols_coef_, trunc_lds.avg_coef_, trunc_lds.best_coef_, total_samples, U.size(0)


def find_max(L, eps):
    # L is a list of equal-size numpy 2d arrays
    # We find one that is most eps-close to the others
    # If each of them has 2/3 prob to be eps/2 close to true value
    # Then that one has high prob to get eps close
    n = len(L)
    Max = -1
    output = None
    for mat in L:
        counter = 0
        for mat2 in L:
            if calc_spectral_norm(mat-mat2)<eps:
                counter += 1
            if counter > Max:
                output = mat
                Max = counter
    return output


# internal function
def generate_samples_B(train_kwargs: Parameters,
                        gen_data: Callable, 
                        eps1: float, 
                        eps2: float, 
                        a_hat: ch.Tensor, 
                        b_hat: ch.Tensor,
                        target_thickness: float, 
                        num_traj: int,
                        T: int, 
                        gamma: float=None):
    '''
    Args: 
      train_kwargs: algorithm's hyperparameters
      gen_data: a callable that creates a sample from the truncated dynamical system
      eps_1: is the initial precision
      eps_2: is the final precision
      a_hat: empirical A estimation
      b_hat: empirical B estimation
      num_traj: number of trajectories that method can use
    return X, U, Y, number of samples, and number of trajectories
    '''
    assert target_thickness != float('inf') or num_traj != float('inf') or T != float('inf'), f"all stopping conditions are {float('inf')}, need to provide at least one stopping variable: (T, num_traj, target_thickness), that isn't infinity"

    calculate_u_t_one = lambda a, b, x: (-b.T@LA.inv(b@b.T)@a@x.T).T
    D, M = a_hat.size(0), b_hat.size(0)

    traj, total_samples = 0, 0
    X, Y, U = ch.zeros((1, D)), ch.zeros((1, D)), ch.zeros((1, M))
    index = 0
    id_ = ch.eye(M)
    covariate_matrix = ch.zeros([D+M,D+M])

    xt = ch.zeros((1, D))
    target = (1/(eps2*eps2)-1/(eps1*eps1))*4
    target_mat = ch.zeros([D+M,D+M])
    np.fill_diagonal(target_mat.numpy(), [0]*D+[target]*M)
    
    while traj < num_traj and X.size(0) < T and calc_thickness(covariate_matrix) < target_thickness:
        traj += 1
        xt = ch.zeros((1, D))
        responsive = True
        while responsive:
          ut = (gamma*id_[index]) [None,...]
          # import pdb; pdb.set_trace()
          sample = gen_data(xt, u_t=ut)
          total_samples += 1
          if sample is not None:
            yt, ut = sample
            X, Y, U = ch.cat((X,xt)), ch.cat((Y,yt)), ch.cat((U,ut))
            xt = yt
            index = (index+1)%M
          else: 
            break
          while True:
            # import pdb; pdb.set_trace()
            ut = calculate_u_t_one(a_hat, b_hat.T, xt)
            sample = gen_data(xt, u_t=ut)
            total_samples += 1
            if sample is not None:
              yt, ut = sample
              X, Y, U = ch.cat((X,xt)), ch.cat((Y,yt)), ch.cat((U,ut))
              xt = yt
              if sample[0].norm() <= 2*np.sqrt(D):
                break
            else: 
              responsive = False
              break
    return X[1:], U[1:], Y[1:], X.size(0), num_traj, X.size(0) / total_samples


# internal function
def generate_samples_A(train_kwargs: Parameters, 
                        gen_data: Callable,
                        eps1: float, 
                        eps2: float, 
                        a_hat: float, 
                        b_hat: ch.Tensor, 
                        target_thickness: float,
                        num_traj: int, 
                        T: int,
                        gamma = None):
    assert target_thickness != float('inf') or num_traj != float('inf') or T != float('inf'), f"all stopping conditions are {float('inf')}, need to provide at least one stopping variable: (T, num_traj, target_thickness), that isn't infinity"

    calculate_u_t_two = lambda a, b, gamma_e_i: (b.T@LA.inv(b@b.T)@gamma_e_i)[None,...]

    D, M = a_hat.size(0), b_hat.size(0)
    
    covariate_matrix = ch.zeros([D+M,D+M])

    traj, total_samples = 0, 0
    X, Y, U = ch.zeros((1, D)), ch.zeros((1, D)), ch.zeros((1, M))
    index = 0
    id_ = ch.eye(D)
    xt = ch.zeros((1, D))

    # break based off of the number of samples collected or number of trajectories
    while traj < num_traj and X.size(0) < T and calc_thickness(covariate_matrix) < target_thickness:
        xt = ch.zeros((1, D))
        traj += 1
        # while the system is responsive 
        responsive = True
        while responsive: 
          ut = calculate_u_t_two(a_hat, b_hat.T, gamma*id_[index]) 
          sample = gen_data(xt, u_t=ut)
          if sample is not None:
            yt, ut = sample
            xt = yt
            sample = gen_data(xt, u_t=ch.zeros(ut.size()))
            total_samples += 1
            if sample is not None:
                yt, ut = sample
                X, Y, U = ch.cat((X,xt)), ch.cat((Y, yt)), ch.cat((U, ut))
                xu = ch.cat((xt,ut), dim=1)
                xu_2 = xu.reshape(M+D,1)@xu.reshape(1,M+D)
                xt = yt
                index = (index+1)%D
            else: 
              break
          else: 
            break
          while True: 
              ut = -(b_hat@LA.inv((b_hat.T@b_hat))@a_hat@xt.T).T              
              sample = gen_data(xt, u_t=ut)
              total_samples += 1
              if sample is not None:
                yt, ut = sample
                X, Y, U = ch.cat((X,xt)), ch.cat((Y,yt)), ch.cat((U,ut))
                xt = yt
                if sample[0].norm() <= 2*np.sqrt(D):
                  break
              else:
                responsive = False
                break
    return X[1:], U[1:], Y[1:], X.size(0), num_traj, X.size(0) / total_samples

def find_estimate(train_kwargs: Parameters, 
                  gen_data: Callable, 
                  eps1: float, 
                  eps2: float, 
                  hat_A: ch.Tensor, 
                  hat_B: ch.Tensor, 
                  delta: float, 
                  gamma_B: float, 
                  gamma_A: float,
                  target_thickness_B: float=float('inf'),
                  target_thickness_A: float=float('inf'),
                  num_traj_part_one: int=float('inf'), 
                  num_traj_part_two: int=float('inf'),
                  T_part_one: int=float('inf'),
                  T_part_two: int=float('inf'), 
                  repeat: int=None):
    '''
    eps1 is the initial precision
    eps2 is the final precision
    hat_A is the estimation of A, same as hat_B
    delta is failing rate
    return new estimators hat_A and hat_B
    '''
    repeat = int(-2*np.log2(delta)) if repeat is None else repeat
    D, M = hat_A.size(0), hat_B.size(0)

    A_buf = []
    B_buf = []

    hat_A, hat_B = hat_A.T, hat_B.T
    
    for _ in range(repeat):
        Xu, Uu, Yu, Nsu, Ntu, alpha_u = generate_samples_B(train_kwargs, gen_data, eps1, eps2/2, hat_A, 
                                                                hat_B, gamma = gamma_B,
                                                                target_thickness=target_thickness_B, 
                                                                num_traj=num_traj_part_one, 
                                                                T=T_part_one)
        Xx, Ux, Yx, Nsx, Ntx, alpha_x = generate_samples_A(train_kwargs, gen_data, eps1, eps2/2, hat_A, 
                                                                hat_B, gamma = gamma_A, 
                                                                target_thickness=target_thickness_A,
                                                                num_traj=num_traj_part_two, 
                                                                T=T_part_two)

        # import pdb; pdb.set_trace()
        # coef_concat = ch.cat([hat_A, hat_B.T], axis=1)
        coef_concat = ch.vstack([hat_A, hat_B])

        
        XU_concat, XX_concat = ch.cat([Xu, Uu], axis=1), ch.cat([Xx, Ux], axis=1)
        eigs = LA.eig(XX_concat.T@XX_concat).eigenvalues.real
        feat_concat = ch.cat([XU_concat, XX_concat])

        y_concat = ch.cat([Yu, Yx])

        train_kwargs.__setattr__('alpha', alpha_x)
        train_kwargs.__setattr__('noise_var', gen_data.noise_var)
        train_kwargs.__setattr__('b', True)
        lr = (1/train_kwargs.alpha) ** train_kwargs.c_gamma
        train_kwargs.__setattr__('lr', lr)

        trunc_lds = TruncatedLinearRegression(train_kwargs, 
                                              emp_weight=coef_concat,
                                              dependent=True)
        trunc_lds.fit(feat_concat.detach(), y_concat.detach())
        
        AB = trunc_lds.best_coef_
        A_, B_ = AB[:,:D], AB[:,D:]

        AB_avg = trunc_lds.avg_coef_
        A_avg, B_avg = AB_avg[:,:D], AB_avg[:,D:]

        A_buf.append(A_)
        B_buf.append(B_)
    return find_max(A_buf,eps2), find_max(B_buf,eps2), A_avg, B_avg, XX_concat, y_concat, XU_concat, feat_concat
