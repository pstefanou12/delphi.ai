import torch.linalg as LA
import torch as ch
import numpy as np
from typing import Callable
import logging

from .truncated_linear_regression import TruncatedLinearRegression
from ..utils.helpers import Parameters, calc_spectral_norm, calc_thickness
from ..utils.defaults import TRUNCATED_LQR_DEFAULTS, check_and_fill_args

logger = logging.getLogger('truncated-lqr')
logger.setLevel(logging.INFO)


class TruncatedLQR:
  def __init__(self, 
                args: Parameters,
                gen_data: Callable, 
                d: int, 
                m: int): 

    # super().__init__(args, defaults=TRUNCATED_LQR_DEFAULTS)
    self.args = check_and_fill_args(args, TRUNCATED_LQR_DEFAULTS)
    assert m >= d, f"m must be greater than or equal to d; d: {d} and m: {m}"
    assert self.args.target_thickness != float('inf') or self.args.num_traj_phase_one != float('inf') or self.args.T_phase_one != float('inf'), f"all stopping conditions are {float('inf')}, need to provide at least one stopping variable: (T, num_traj, target_thickness), that isn't infinity"
    assert self.args.target_thickness != float('inf') or self.args.num_traj_phase_two != float('inf') or self.args.T_phase_two != float('inf'), f"all stopping conditions are {float('inf')}, need to provide at least one stopping variable: (T, num_traj, target_thickness), that isn't infinity"
    assert self.args.target_thickness != float('inf') or self.args.num_traj_gen_samples_B != float('inf') or self.args.T_gen_samples_B != float('inf'), f"all stopping conditions are {float('inf')}, need to provide at least one stopping variable: (T, num_traj, target_thickness), that isn't infinity"
    assert self.args.target_thickness != float('inf') or self.args.num_traj_gen_samples_A != float('inf') or self.args.T_gen_samples_A != float('inf'), f"all stopping conditions are {float('inf')}, need to provide at least one stopping variable: (T, num_traj, target_thickness), that isn't infinity"

    self.gen_data = gen_data
    self.d, self.m = d, m
    self.gamma_A, self.gamma_B = self.args.R / self.args.U_A, self.args.R / self.args.U_B

  def fit(self): 
    self.run_phase_one()
    self.run_phase_two()
    self.run_warm_phase()

  def run_phase_one(self):
      '''
      Cold start phase 1. Initial estimation for A.
      Args: 
        num_traj: number of trajectories allowed
      Returns: ols estimate, initial estimation with plevrakis, and 
        number of trajectories taken
      '''
      # assert self.args.target_thickness != float('inf') or self.args.num_traj_phase_one != float('inf') or self.T_phase_one != float('inf'), f"all stopping conditions are {float('inf')}, need to provide at least one stopping variable: (T, num_traj, target_thickness), that isn't infinity"
      logger.info(f'begin cold start phase one...')
      num_trajectories = 1
      total_samples = 0
      x_t = ch.zeros((1, self.d))
      X, Y, U = ch.Tensor([]), ch.Tensor([]), ch.Tensor([])
      covariate_matrix = ch.zeros([self.d,self.d])

      while num_trajectories < self.args.num_traj_phase_one and X.size(0) < self.args.T_phase_one and calc_thickness(covariate_matrix) < self.args.target_thickness:
          sample = self.gen_data(x_t, u_t=ch.zeros((1, self.m)))
          total_samples += 1
          if sample is not None:
              y_t, u_t = sample
              X, Y, U = ch.cat((X,x_t)), ch.cat((Y,y_t)), ch.cat((U, u_t))
              covariate_matrix += x_t.T@x_t
              x_t = y_t
          else:
              x_t = ch.zeros((1, self.d))
              num_trajectories += 1

          if X.size(0) % 100 == 0:
              logger.info(f'total number of samples: {X.size(0)}')

      self.args.__setattr__('noise_var', self.gen_data.noise_var)

      trunc_lds = TruncatedLinearRegression(self.args, 
                                            dependent=True)
      trunc_lds.fit(X, Y)
      self.A_hat_ = trunc_lds.best_coef_

  def run_phase_two(self): 
      '''
      Cold start phase 2. Initial estimation for B.
      '''
      logger.info(f'begin cold start phase two...')
      total_samples = 0
      index = 0
      xt = ch.zeros([1, self.d])
      id_ = ch.eye(self.m)
      U, Y = ch.Tensor([]), ch.Tensor([])
      covariate_matrix = ch.zeros([self.m, self.m])
      gamma = 2.5

      while total_samples < self.args.num_traj_phase_two and U.size(0) < self.args.T_phase_two and calc_thickness(covariate_matrix) < self.args.target_thickness:
          u = (gamma*id_[index])[None,...]
          sample = self.gen_data(xt, u_t=u)
        
          total_samples += 1
          if sample is not None: 
              y, u = sample
              index = (index+1)%self.m
              U, Y = ch.cat([U, u]), ch.cat([Y, y])
              covariate_matrix += u.T@u

          if U.size(0) % 100 == 0: 
              logger.info(f'total number of samples: {U.size(0)}')

      self.args.__setattr__('noise_var', self.gen_data.noise_var)

      trunc_lds = TruncatedLinearRegression(self.args, 
                                            dependent=True)
      trunc_lds.fit(U, Y)
      self.B_hat_ = trunc_lds.best_coef_

  def find_max(self, 
                L, 
                eps):
    # L is a list of equal-size numpy 2d arrays
    # We find one that is most eps-close to the others
    # If each of them has 2/3 prob to be eps/2 close to true value
    # Then that one has high prob to get eps close
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

  @staticmethod
  def calculate_u_t_one(a, b, x): 
    return (-b@LA.inv(b.T@b)@a.T@x.T).T

  def generate_samples_B(self):
      traj, total_samples = 0, 0
      X, Y, U = ch.zeros([1, self.d]), ch.zeros([1, self.d]), ch.zeros([1, self.m])
      index = 0
      id_ = ch.eye(self.m)
      covariate_matrix = ch.zeros([self.d+self.m,self.d+self.m])

      xt = ch.zeros((1, self.d))
      target = (1/(self.args.eps2**2)-1/(self.args.eps1**2))*4
      target_mat = ch.zeros([self.d+self.m,self.d+self.m])
      np.fill_diagonal(target_mat.numpy(), [0]*self.d+[target]*self.m)

      '''
      TODO: figure out a way to do better stopping criteria
      ''' 
      while (traj < self.args.num_traj_gen_samples_B and X.size(0) < self.args.T_gen_samples_B 
      and calc_thickness(covariate_matrix) < self.args.target_thickness):
          traj += 1
          xt = ch.zeros((1, self.d))
          responsive = True
          while responsive:
            ut = (self.args.gamma*id_[index]) [None,...]
            sample = self.gen_data(xt, u_t=ut)
            total_samples += 1
            if sample is not None:
              yt, ut = sample
              X, Y, U = ch.cat((X,xt)), ch.cat((Y,yt)), ch.cat((U,ut))
              xu = ch.cat([xt, ut], dim=1) 
              covariate_matrix += xu.T@xu
              xt = yt
              index = (index+1)%self.m
            else: 
              break
            while True:
              ut = self.calculate_u_t_one(self.A_hat_, self.B_hat_, xt)
              sample = self.gen_data(xt, u_t=ut)
              total_samples += 1
              if sample is not None:
                yt, ut = sample
                X, Y, U = ch.cat((X,xt)), ch.cat((Y,yt)), ch.cat((U,ut))
                xu = ch.cat([xt, ut], dim=1) 
                covariate_matrix += xu.T@xu
                xt = yt
                if sample[0].norm() <= 2*np.sqrt(self.d):
                  break
              else: 
                responsive = False
                break
      return X[1:], U[1:], Y[1:]

  @staticmethod
  def calculate_u_t_two(a, b, gamma_e_i): 
    return (b@LA.inv(b.T@b)@gamma_e_i)[None,...]

  @staticmethod
  def calculate_u_t_three(a, b, x): 
    return (-b@LA.inv(b.T@b)@a.T@x.T).T 

  def generate_samples_A(self):
      covariate_matrix = ch.zeros([self.d+self.m,self.d+self.m])

      traj, total_samples = 0, 0
      X, Y, U = ch.zeros([1, self.d]), ch.zeros([1, self.d]), ch.zeros([1, self.m])
      index = 0
      id_ = ch.eye(self.d)

      # break based off of the number of samples collected or number of trajectories
      while traj < self.args.num_traj_gen_samples_A and X.size(0) < self.args.T_gen_samples_A and calc_thickness(covariate_matrix) < self.args.target_thickness:
          xt = ch.zeros(1, self.d)
          traj += 1
          # while the system is responsive 
          responsive = True
          while responsive: 
            ut = self.calculate_u_t_two(self.A_hat_, self.B_hat_, self.args.gamma*id_[index]) 
            sample = self.gen_data(xt, u_t=ut)
            if sample is not None:
              yt, ut = sample
              xt = yt
              sample = self.gen_data(xt, u_t=ch.zeros(ut.size()))
              total_samples += 1
              if sample is not None:
                  yt, ut = sample
                  X, Y, U = ch.cat((X,xt)), ch.cat((Y, yt)), ch.cat((U, ut))
                  xu = ch.cat([xt, ut], dim=1)
                  covariate_matrix += xu.T@xu
                  xt = yt
                  index = (index+1)%self.d
              else: 
                break
            else: 
              break
            while True: 
                ut = self.calculate_u_t_three(self.A_hat_, self.B_hat_, xt)
                sample = self.gen_data(xt, u_t=ut)
                total_samples += 1
                if sample is not None:
                  yt, ut = sample
                  X, Y, U = ch.cat((X,xt)), ch.cat((Y,yt)), ch.cat((U,ut))
                  xu = ch.cat([xt, ut], dim=1) 
                  covariate_matrix += xu.T@xu 
                  xt = yt
                  if sample[0].norm() <= self.args.R + 3*np.sqrt(self.d):
                    break
                else:
                  responsive = False
                  break
      return X[1:], U[1:], Y[1:]

  def run_warm_phase(self) -> None:
      logger.info(f'begin warm start...')
      repeat = int(-2*np.log2(self.args.delta)) if self.args.repeat is None else self.args.repeat

      assert repeat >= 1, f"repeat must be greater than or equal to 1; repeat: {repeat}"

      A_results, B_results = ch.Tensor([]), ch.Tensor([])
      A_avg_results, B_avg_results = ch.Tensor([]), ch.Tensor([])

      coef_concat = ch.cat([self.A_hat_, self.B_hat_])

      for _ in range(repeat):
        Xu, Uu, Yu = self.generate_samples_B()
        Xx, Ux, Yx = self.generate_samples_A()

        XU_concat, XX_concat = ch.cat([Xu, Uu], axis=1), ch.cat([Xx, Ux], axis=1)
        feat_concat = ch.cat([XU_concat, XX_concat])
        y_concat = ch.cat([Yu, Yx])

        self.args.__setattr__('noise_var', self.gen_data.noise_var)

        trunc_lds = TruncatedLinearRegression(self.args, 
                                              emp_weight=coef_concat,
                                              dependent=True)
        trunc_lds.fit(feat_concat.detach(), y_concat.detach())
        
        AB = trunc_lds.best_coef_
        A_, B_ = AB[:self.d], AB[self.d:]

        AB_avg = trunc_lds.avg_coef_
        A_avg, B_avg = AB_avg[:self.d], AB_avg[self.d:]

        A_results = ch.cat([A_results, A_[None,...]])
        B_results = ch.cat([B_results, B_[None,...]])
         
        A_avg_results = ch.cat([A_avg_results, A_avg[None,...]])
        B_avg_results = ch.cat([B_avg_results, B_avg[None,...]])


      self.A_ = self.find_max(A_results, self.args.eps2)
      self.B_ = self.find_max(B_results, self.args.eps2)

  @property
  def A_hat_(self): 
    return self._A_hat_

  @A_hat_.setter
  def A_hat_(self, value): 
    self._A_hat_ = value

  @property
  def B_hat_(self): 
    return self._B_hat_

  @B_hat_.setter
  def B_hat_(self, value): 
    self._B_hat_ = value

  @property
  def A_(self): 
    return self._A_

  @A_.setter
  def A_(self, value): 
    self._A_ = value

  @property
  def B_(self): 
    return self._B_

  @B_.setter
  def B_(self, value): 
    self._B_ = value
