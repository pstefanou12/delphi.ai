import torch.linalg as LA
import torch as ch
import numpy as np

from .. import oracle
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


def phase_one(train_kwargs, 
              gen_data,
              D: int, M: int, 
              target_thickness=float('inf'), 
              num_traj=float('inf'), 
              store=None):
    '''
    Cold start phase 1. Initial estimation for A.
    Args: 
      num_traj: number of trajectories allowed
    Returns: ols estimate, initial estimation with plevrakis, and 
      number of trajectories taken
    '''
    num_samples = 0
    num_trajectories = 1
    total_samples = 0
    x_t = ch.zeros((1, D))
    X, Y, U = ch.Tensor([]), ch.Tensor([]), ch.Tensor([])
    covariate_matrix = ch.zeros([D,D])

    while num_trajectories < num_traj:
        sample = gen_data(x_t, u_t=ch.zeros((1, M)))
        total_samples += 1
        if sample is not None:
            y_t, u_t = sample
            X, Y, U = ch.cat((X,x_t)), ch.cat((Y,y_t)), ch.cat((U, u_t))
            covariate_matrix += x_t.T@x_t
            x_t = y_t
            num_samples += 1

        else:
            x_t = ch.zeros((1, D))
            num_trajectories += 1

        thickness = calc_thickness(covariate_matrix)
        if thickness > target_thickness: 
          break

    print(f'number of allowed trajectories: {num_traj}')
    print(f'number of trajectories used: {num_trajectories}')

    alpha = num_samples / total_samples

    X_1, Y_1 = X.clone(), Y.clone()

    train_kwargs.__setattr__('alpha', alpha)
    train_kwargs.__setattr__('noise_var', gen_data.noise_var)
    train_kwargs.__setattr__('b', False)
    train_kwargs.__setattr__('verbose', True)


    lr = (1/train_kwargs.alpha) ** train_kwargs.c_gamma
    train_kwargs.__setattr__('lr', lr)

    trunc_lds = TruncatedLinearRegression(train_kwargs, 
                                         dependent=True, 
                                          store=store)
    trunc_lds.fit(X_1, Y_1)

    return trunc_lds.emp_weight, trunc_lds.best_coef_, trunc_lds.avg_coef_, num_trajectories


def phase_two(train_kwargs, 
              gen_data, 
              D: int, M: int,
              R: float, U_B: float,
              target_thickness=float('inf'),
              store=None,
              num_traj=float('inf')): 
    '''
    Cold start phase 2. Initial estimation for B.
    '''
    total_samples = 0
    index = 0
    xt = ch.zeros((1, D))
    id_ = ch.eye(M)
    U, Y = ch.Tensor([]), ch.Tensor([])
    curr_mat = ch.zeros([M,M])
    gamma = R/U_B

    while total_samples < num_traj:
        u = (gamma*id_[index])[None,...]
        sample = gen_data(xt)
        
        total_samples += 1
        if sample is not None: 
            y, u = sample
            index = (index+1)%M
            U, Y = ch.cat([U, u]), ch.cat([Y, y])
            curr_mat += u.T@u

        thickness = calc_thickness(curr_mat)
        if thickness > target_thickness and U.size(0) > U.size(1):
            break

    print(f'number of allowed trajectories: {num_traj}')
    print(f'number of trajectories used: {total_samples}')
        
    alpha = U.size(0) / total_samples

    train_kwargs.__setattr__('alpha', alpha)
    train_kwargs.__setattr__('noise_var', gen_data.noise_var)
    train_kwargs.__setattr__('b', True)
    train_kwargs.__setattr__('batch_size', 10)
    train_kwargs.__setattr__('shuffle', False)

    lr = (1/train_kwargs.alpha) ** train_kwargs.c_gamma
    train_kwargs.__setattr__('lr', lr)
        
    trunc_lds = TruncatedLinearRegression(train_kwargs, 
                                         dependent=True, 
                                          store=store)
    trunc_lds.fit(U, Y)
    return trunc_lds.emp_weight, trunc_lds.best_coef_, trunc_lds.avg_coef_, total_samples


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


def part_one(train_kwargs,
              gen_data, 
              eps1, 
              eps2, 
              a_hat, b_hat, 
              D: int, M: int,
              target_thickness=float('inf'),
              traj=float('inf'), 
              gamma = None):
    '''
    Args: 
      train_kwargs: algorithm's hyperparameters
      gen_data: a callable that creates a sample from the truncated dynamical system
      eps_1: is the initial precision
      eps_2: is the final precision
      a_hat: empirical A estimation
      b_hat: empirical B estimation
      target_thickness: lower bound thickness for covariate matrix (used for stopping)
      traj: number of trajectories that method can use
    return X, U, Y, number of samples, and number of trajectories
    '''
    calculate_u_t_one = lambda a, b, x: (-b.T@LA.inv(b@b.T)@a@x.T).T

    num_traj, total_samples = 0, 0
    X, Y, U = ch.zeros((1, D)), ch.zeros((1, D)), ch.zeros((1, M))
    index = 0
    id_ = ch.eye(M)
    curr_mat = ch.zeros([D+M,D+M])

    xt = ch.zeros((1, D))
    target = (1/(eps2*eps2)-1/(eps1*eps1))*4
    target_mat = ch.zeros([D+M,D+M])
    np.fill_diagonal(target_mat.numpy(), [0]*D+[target]*M)
    
    while num_traj < traj:
        num_traj += 1
        xt = ch.zeros((1, D))
        responsive = True

        while responsive:
          ut = (gamma*id_[index])[None,...]
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
            ut = calculate_u_t_one(a_hat, b_hat, xt)
            sample = gen_data(xt, u_t=ut)
            total_samples += 1
            if sample is not None:
              yt, ut = sample
              if yt.norm() <= 2*(D**.5):
                break
              X, Y, U = ch.cat((X,xt)), ch.cat((Y,yt)), ch.cat((U,ut))
              xt = yt
            else: 
              responsive = False
              break
    return X[1:], U[1:], Y[1:], X.size(0), num_traj, X.size(0) / total_samples


def part_two(train_kwargs, 
              gen_data,
              eps1, 
              eps2, 
              a_hat, 
              b_hat, 
              D: int, M: int, 
              target_thickness=float('inf'),
              traj=float('inf'), 
              gamma = None):
    calculate_u_t_two = lambda a, b, gamma_e_i: (b.T@LA.inv(b@b.T)@gamma_e_i)[None,...]

    num_sample, num_traj, total_samples = 0, 0, 0
    X, Y, U = ch.zeros((1, D)), ch.zeros((1, D)), ch.zeros((1, M))
    index = 0
    id_ = ch.eye(D)
    xt = ch.zeros((1, D))

    while num_traj < traj:
        xt = ch.zeros((1, D))
        num_traj += 1

        responsive = True
        while responsive: 
          ut = calculate_u_t_two(a_hat, b_hat, gamma*id_[index]) 
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
                # xu_2 = xu.reshape(M+D,1)@xu.reshape(1,M+D)

                xt = yt
                index = (index+1)%D
            else: 
              break
          else: 
            break
          while True: 
              ut = -(b_hat.T@LA.inv((b_hat@b_hat.T))@a_hat@xt.T).T
              sample = gen_data(xt, u_t=ut)

              total_samples += 1
              if sample is not None:
                yt, ut = sample
                if yt.norm() <= 2*(D**.5):
                  break
                X, Y, U = ch.cat((X,xt)), ch.cat((Y,yt)), ch.cat((U,ut))
                xt = yt
              else:
                responsive = False
                break
    return X[1:], U[1:], Y[1:], X.size(0), num_traj, X.size(0) / total_samples


REPEAT = 1
def find_estimate(train_kwargs, 
                  gen_data, 
                  eps1, 
                  eps2, 
                  hat_A, 
                  hat_B, 
                  delta, 
                  gamma_A, 
                  gamma_B, 
                  D: int, M: int,
                  num_traj=float('inf'),
                  store=None):
    '''
    eps1 is the initial precision
    eps2 is the final precision
    hat_A is the estimation of A, same as hat_B
    delta is failing rate
    return new estimators hat_A and hat_B
    '''
    repeat = int(-2*np.log2(delta))

    A_buf = []
    B_buf = []
    
    for _ in range(REPEAT):
        Xu, Uu, Yu, Nsu, Ntu, alpha_u = part_one(train_kwargs, gen_data, eps1, eps2/2, hat_A, 
                                                      hat_B, D, M, gamma = gamma_B, 
                                                      traj=int(num_traj/2))
        Xx, Ux, Yx, Nsx, Ntx, alpha_x = part_two(train_kwargs, gen_data, eps1, eps2/2, hat_A, 
                                                      hat_B, D, M, gamma = gamma_A, 
                                                      traj=int(num_traj/2))
        coef_concat = ch.cat([hat_A, hat_B], axis=1)

        # import pdb; pdb.set_trace()
        XU_concat, XX_concat = ch.cat([Xu, Uu], axis=1), ch.cat([Xx, Ux], axis=1)
        eigs = LA.eig(XX_concat.T@XX_concat).eigenvalues.real
        feat_concat = ch.cat([XU_concat, XX_concat])


        y_concat = ch.cat([Yu, Yx])

        train_kwargs.__setattr__('alpha', alpha_x)
        train_kwargs.__setattr__('noise_var', gen_data.noise_var)
        train_kwargs.__setattr__('b', False)
        train_kwargs.__setattr__('verbose', True)
        train_kwargs.__setattr__('batch_size', 10)
        train_kwargs.__setattr__('shuffle', False)

        lr = (1/train_kwargs.alpha) ** train_kwargs.c_gamma
        train_kwargs.__setattr__('lr', lr)
        trunc_lds = TruncatedLinearRegression(train_kwargs, 
                                              emp_weight=coef_concat,
                                              store=store,
                                             dependent=True)
        trunc_lds.fit(feat_concat.detach(), y_concat.detach())
        
        AB = trunc_lds.best_coef_
        A_, B_ = AB[:,:D], AB[:,D:]
        AB_avg = trunc_lds.history[-100:].mean(0)
        A_avg, B_avg = AB_avg[:,:D], AB_avg[:,D:]

        A_buf.append(A_)
        B_buf.append(B_)
    return find_max(A_buf,eps2), find_max(B_buf,eps2), A_avg, B_avg, XX_concat, y_concat, XU_concat, num_traj, feat_concat

