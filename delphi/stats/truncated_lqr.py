# Author: pstefanou12@
"""
Truncated Linear Quadratic Regulator (LQR) algorithm.

Three phase algorithm that removes bias from the case when there is
truncation bias in a LQR dynamical system.
"""

import logging
from typing import Callable

import numpy as np
import torch as ch
from torch import linalg as LA
from cox.store import Store

from delphi.utils.defaults import TRUNC_LQR_DEFAULTS, check_and_fill_args
from delphi.utils.helpers import Parameters, calc_spectral_norm, calc_thickness
from delphi.stats.truncated_linear_regression import TruncatedLinearRegression

logger = logging.getLogger("truncated-lqr")
logger.setLevel(logging.INFO)


class TruncatedLQR:  # pylint: disable=too-many-instance-attributes
    """Three-phase algorithm that removes truncation bias in a LQR dynamical system."""

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        args: Parameters,
        gen_data: Callable,
        d: int,  # pylint: disable=invalid-name
        m: int,  # pylint: disable=invalid-name
        rand_seed: int = 0,
    ):
        """Initialize TruncatedLQR.

        Args:
            args (Parameters): hyperparameter object
            gen_data (Callable): data-generating function for the dynamical system
            d (int): state dimension of matrix A
            m (int): input dimension of matrix B; must satisfy m >= d
            rand_seed (int): random seed for reproducibility
        """
        self.args = check_and_fill_args(args, TRUNC_LQR_DEFAULTS)
        assert m >= d, f"m must be greater than or equal to d; d: {d} and m: {m}"
        assert (
            self.args.target_thickness != float("inf")
            or self.args.num_traj_phase_one != float("inf")
            or self.args.T_phase_one != float("inf")
        ), (
            "all stopping conditions are inf, need to provide at least one stopping "
            "variable: (T, num_traj, target_thickness), that isn't infinity"
        )
        assert (
            self.args.target_thickness != float("inf")
            or self.args.num_traj_phase_two != float("inf")
            or self.args.T_phase_two != float("inf")
        ), (
            "all stopping conditions are inf, need to provide at least one stopping "
            "variable: (T, num_traj, target_thickness), that isn't infinity"
        )
        assert (
            self.args.target_thickness != float("inf")
            or self.args.num_traj_gen_samples_B != float("inf")
            or self.args.T_gen_samples_B != float("inf")
        ), (
            "all stopping conditions are inf, need to provide at least one stopping "
            "variable: (T, num_traj, target_thickness), that isn't infinity"
        )
        assert (
            self.args.target_thickness != float("inf")
            or self.args.num_traj_gen_samples_A != float("inf")
            or self.args.T_gen_samples_A != float("inf")
        ), (
            "all stopping conditions are inf, need to provide at least one stopping "
            "variable: (T, num_traj, target_thickness), that isn't infinity"
        )

        self.gen_data = gen_data
        self.d, self.m = d, m  # pylint: disable=invalid-name
        self.rand_seed = rand_seed

        self.c = (self.args.R - 3 * (self.m**0.5)) / self.args.U_B  # pylint: disable=invalid-name
        logger.info("c: %s", self.c)

    def fit(self):
        """Run all three phases of the truncated LQR algorithm."""
        self.run_phase_one()
        self.run_phase_two()
        self.run_warm_phase()

    def run_phase_one(self, store: Store = None):
        """
        Cold start phase 1. Initial estimation for A.

        Args:
          store: cox.store.Store for logging
        Returns:
          ols estimate, initial estimation with plevrakis, and
          number of trajectories taken
        """
        logger.info("begin cold start phase one...")
        num_trajectories = 1
        total_samples = 0
        x_t = ch.zeros((1, self.d))
        X, Y, U = ch.Tensor([]), ch.Tensor([]), ch.Tensor([])  # pylint: disable=invalid-name
        covariate_matrix = ch.zeros([self.d, self.d])

        while (
            num_trajectories < self.args.num_traj_phase_one
            and X.size(0) < self.args.T_phase_one
            and calc_thickness(covariate_matrix) < self.args.target_thickness
        ):
            sample = self.gen_data(x_t, u_t=ch.zeros((1, self.m)))
            total_samples += 1
            if sample is not None:
                y_t, u_t = sample
                X, Y, U = ch.cat((X, x_t)), ch.cat((Y, y_t)), ch.cat((U, u_t))
                covariate_matrix += x_t.T @ x_t
                x_t = y_t
            else:
                x_t = ch.zeros((1, self.d))
                num_trajectories += 1

            if X.size(0) % 100 == 0:
                logger.info("total number of samples: %s", X.size(0))

        self.trunc_lds_phase_one = TruncatedLinearRegression(  # pylint: disable=attribute-defined-outside-init
            self.args.phi,
            self.args,
            self.gen_data.noise_var,
            dependent=True,
            rand_seed=self.rand_seed,
        )
        self.trunc_lds_phase_one.fit(X, Y)
        self.A_hat_ = self.trunc_lds_phase_one.coef_

    def run_phase_two(self, store: Store = None):
        """
        Cold start phase 2. Initial estimation for B.

        Args:
          store: cox.store.Store for logging
        """
        logger.info("begin cold start phase two...")
        total_samples, index = 0, 0
        xt, id_ = ch.zeros([1, self.d]), ch.eye(self.m)
        U, Y = ch.Tensor([]), ch.Tensor([])  # pylint: disable=invalid-name
        covariate_matrix = ch.zeros([self.m, self.m])

        while (
            total_samples < self.args.num_traj_phase_two
            and U.size(0) < self.args.T_phase_two
            and calc_thickness(covariate_matrix) < self.args.target_thickness
        ):
            u = (self.c * id_[index])[None, ...]
            sample = self.gen_data(xt, u_t=u)

            total_samples += 1
            if sample is not None:
                y, u = sample
                index = (index + 1) % self.m
                U, Y = ch.cat([U, u]), ch.cat([Y, y])
                covariate_matrix += u.T @ u

            if U.size(0) % 100 == 0:
                logger.info("total number of samples: %s", U.size(0))

        self.trunc_lds_phase_two = TruncatedLinearRegression(  # pylint: disable=attribute-defined-outside-init
            self.args.phi,
            self.args,
            self.gen_data.noise_var,
            dependent=True,
            rand_seed=self.rand_seed,
        )
        self.trunc_lds_phase_two.fit(U, Y)
        self.B_hat_ = self.trunc_lds_phase_two.coef_

    def find_max(self, mat_list, eps):
        """
        Find the matrix most eps-close to the others.

        L is a list of equal-size numpy 2d arrays.
        We find one that is most eps-close to the others.
        If each of them has 2/3 prob to be eps/2 close to true value,
        then that one has high prob to get eps close.

        Args:
          mat_list: list of equal-size numpy 2d arrays
          eps: epsilon threshold for spectral norm closeness
        Returns:
          the matrix with the most neighbors within eps distance
        """
        max_count = -1
        output = None
        for mat in mat_list:
            counter = 0
            for mat2 in mat_list:
                if calc_spectral_norm(mat - mat2) < eps:
                    counter += 1
                if counter > max_count:
                    output = mat
                    max_count = counter
        return output

    @staticmethod
    def calculate_u_t_one(a, b, x):  # pylint: disable=invalid-name
        """Calculate control input for phase one stabilization."""
        return (-b @ LA.inv(b.T @ b) @ a.T @ x.T).T  # pylint: disable=not-callable

    def generate_samples_B(self):  # pylint: disable=invalid-name,too-many-locals
        """Generate samples focused on estimating matrix B."""
        logger.info("begin b focused part...")
        traj, total_samples = 0, 0
        X, Y, U = ch.zeros([1, self.d]), ch.zeros([1, self.d]), ch.zeros([1, self.m])  # pylint: disable=invalid-name
        index = 0
        id_ = ch.eye(self.m)
        covariate_matrix = ch.zeros([self.d + self.m, self.d + self.m])

        xt = ch.zeros((1, self.d))
        target = (1 / (self.args.eps2**2) - 1 / (self.args.eps1**2)) * 4
        target_mat = ch.zeros([self.d + self.m, self.d + self.m])
        np.fill_diagonal(target_mat.numpy(), [0] * self.d + [target] * self.m)

        while (
            traj < self.args.num_traj_gen_samples_B
            and X.size(0) < self.args.T_gen_samples_B
            and calc_thickness(covariate_matrix) < self.args.target_thickness
        ):
            traj += 1
            xt = ch.zeros((1, self.d))
            responsive = True
            while responsive:
                ut = (self.args.gamma * id_[index])[None, ...]
                sample = self.gen_data(xt, u_t=ut)
                total_samples += 1
                if sample is not None:
                    yt, ut = sample
                    X, Y, U = ch.cat((X, xt)), ch.cat((Y, yt)), ch.cat((U, ut))
                    xu = ch.cat([xt, ut], dim=1)
                    covariate_matrix += xu.T @ xu
                    xt = yt
                    index = (index + 1) % self.m
                else:
                    break
                while True:
                    ut = self.calculate_u_t_one(self.A_hat_, self.B_hat_, xt)
                    sample = self.gen_data(xt, u_t=ut)
                    total_samples += 1
                    if sample is not None:
                        yt, ut = sample
                        X, Y, U = ch.cat((X, xt)), ch.cat((Y, yt)), ch.cat((U, ut))
                        xu = ch.cat([xt, ut], dim=1)
                        covariate_matrix += xu.T @ xu
                        xt = yt
                        if sample[0].norm() <= 2 * np.sqrt(self.d):
                            break
                    else:
                        responsive = False
                        break
            logger.info(
                "number of trajectories: %s; number of samples collected: %s",
                traj,
                X.size(0),
            )
        return X[1:], U[1:], Y[1:]

    @staticmethod
    def calculate_u_t_two(_a, b, gamma_e_i):  # pylint: disable=invalid-name
        """Calculate control input for phase two B estimation."""
        return (b @ LA.inv(b.T @ b) @ gamma_e_i)[None, ...]  # pylint: disable=not-callable

    @staticmethod
    def calculate_u_t_three(a, b, x):  # pylint: disable=invalid-name
        """Calculate control input for phase three stabilization."""
        return (-b @ LA.inv(b.T @ b) @ a.T @ x.T).T  # pylint: disable=not-callable

    def generate_samples_A(self):  # pylint: disable=invalid-name,too-many-locals
        """Generate samples focused on estimating matrix A."""
        logger.info("begin a focused part...")
        covariate_matrix = ch.zeros([self.d + self.m, self.d + self.m])

        traj, total_samples = 0, 0
        X, Y, U = ch.zeros([1, self.d]), ch.zeros([1, self.d]), ch.zeros([1, self.m])  # pylint: disable=invalid-name
        index = 0
        id_ = ch.eye(self.d)

        # Break based on the number of samples collected or number of trajectories.
        while (
            traj < self.args.num_traj_gen_samples_A
            and X.size(0) < self.args.T_gen_samples_A
            and calc_thickness(covariate_matrix) < self.args.target_thickness
        ):
            xt = ch.zeros(1, self.d)
            traj += 1
            # While the system is responsive.
            responsive = True
            while responsive:
                ut = self.calculate_u_t_two(
                    self.A_hat_, self.B_hat_, self.args.gamma * id_[index]
                )
                sample = self.gen_data(xt, u_t=ut)
                if sample is not None:
                    yt, ut = sample
                    xt = yt
                    sample = self.gen_data(xt, u_t=ch.zeros(ut.size()))
                    total_samples += 1
                    if sample is not None:
                        yt, ut = sample
                        X, Y, U = ch.cat((X, xt)), ch.cat((Y, yt)), ch.cat((U, ut))
                        xu = ch.cat([xt, ut], dim=1)
                        covariate_matrix += xu.T @ xu
                        xt = yt
                        index = (index + 1) % self.d
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
                        X, Y, U = ch.cat((X, xt)), ch.cat((Y, yt)), ch.cat((U, ut))
                        xu = ch.cat([xt, ut], dim=1)
                        covariate_matrix += xu.T @ xu
                        xt = yt
                        if sample[0].norm() <= self.args.R + 3 * np.sqrt(self.d):
                            break
                    else:
                        responsive = False
                        break
            logger.info(
                "number of trajectories: %s; number of samples collected: %s",
                traj,
                X.size(0),
            )
        return X[1:], U[1:], Y[1:]

    def run_warm_phase(self, store: Store = None) -> None:
        """Run the warm start phase of the algorithm."""
        logger.info("begin warm start...")
        repeat = (
            int(-2 * np.log2(self.args.delta))
            if self.args.repeat is None
            else self.args.repeat
        )

        assert repeat >= 1, (
            f"repeat must be greater than or equal to 1; repeat: {repeat}"
        )

        A_results, B_results = ch.Tensor([]), ch.Tensor([])  # pylint: disable=invalid-name

        coef_concat = ch.cat([self.A_hat_, self.B_hat_])
        for _ in range(repeat):
            Xu, Uu, Yu = self.generate_samples_B()  # pylint: disable=invalid-name
            Xx, Ux, Yx = self.generate_samples_A()  # pylint: disable=invalid-name

            XU_concat = ch.cat([Xu, Uu], axis=1)  # pylint: disable=invalid-name
            XX_concat = ch.cat([Xx, Ux], axis=1)  # pylint: disable=invalid-name
            feat_concat = ch.cat([XU_concat, XX_concat])
            y_concat = ch.cat([Yu, Yx])

            self.trunc_lds_phase_three = TruncatedLinearRegression(  # pylint: disable=attribute-defined-outside-init
                self.args.phi,
                self.args,
                self.gen_data.noise_var,
                emp_weight=coef_concat,
                dependent=True,
                rand_seed=self.rand_seed,
            )
            self.trunc_lds_phase_three.fit(feat_concat.detach(), y_concat.detach())

            AB = self.trunc_lds_phase_three.coef_  # pylint: disable=invalid-name
            A_, B_ = AB[: self.d], AB[self.d :]  # pylint: disable=invalid-name

            A_results = ch.cat([A_results, A_[None, ...]])
            B_results = ch.cat([B_results, B_[None, ...]])

        self.A_ = self.find_max(A_results, self.args.eps2)
        self.B_ = self.find_max(B_results, self.args.eps2)

    @property
    def A_hat_(self):  # pylint: disable=invalid-name
        """Property for A_hat_ attribute."""
        return self._A_hat_

    @A_hat_.setter
    def A_hat_(self, value):  # pylint: disable=invalid-name
        self._A_hat_ = value

    @property
    def B_hat_(self):  # pylint: disable=invalid-name
        """Property for B_hat_ attribute."""
        return self._B_hat_

    @B_hat_.setter
    def B_hat_(self, value):  # pylint: disable=invalid-name
        self._B_hat_ = value

    @property
    def A_(self):  # pylint: disable=invalid-name
        """Property for A_ attribute."""
        return self._A_

    @A_.setter
    def A_(self, value):  # pylint: disable=invalid-name
        self._A_ = value

    @property
    def B_(self):  # pylint: disable=invalid-name
        """Property for B_ attribute."""
        return self._B_

    @B_.setter
    def B_(self, value):  # pylint: disable=invalid-name
        self._B_ = value
