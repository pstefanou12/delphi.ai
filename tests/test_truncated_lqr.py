"""
Test suite for evaluating truncated lqr algorithm.
"""
import torch as ch
import torch.linalg as LA

from delphi import oracle
from delphi.stats.truncated_lqr import TruncatedLQR
from delphi.utils.helpers import Parameters, calc_spectral_norm
from .test_utils import GenerateTruncatedLQRData, calc_sarah_dean

def test_truncated_lqr():
    RAND_SEED = 69
    ch.manual_seed(RAND_SEED)
    gamma = 2.0
    R = 3.0
    D = 3
    M = 3
    NUM_TRAJ = 100
    assert M >= D, f'M is currently: {M}, but it needs to be greater than or equal to D: {D}'

    NOISE_VAR = ch.eye(D)
    assert M >= D, f'M is currently: {M}, but it needs to be larger than D: {D}'

    A = ch.Tensor([[1.01, .01, 0], 
                [.01, 1.01, .01], 
                [0, .01, 1.01]])
    B = ch.eye(M)

    # membership oracle
    phi = oracle.LogitBall(R)
    gen_data = GenerateTruncatedLQRData(phi, A, B, noise_var=NOISE_VAR)

    U_A = float(calc_spectral_norm(A))
    U_B = float(calc_spectral_norm(B))

    TRAIN_KWARGS = Parameters({
        'phi': phi,
        'c_gamma': 2.0,
        'fit_intercept': False,
        'epochs': 2, 
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
        'num_traj_phase_one': NUM_TRAJ // 4, 
        'num_traj_phase_two': NUM_TRAJ // 4,
        'num_traj_gen_samples_A': NUM_TRAJ // 4,
        'num_traj_gen_samples_B': NUM_TRAJ // 4,
        'shuffle': False,
        'c_eta': .1,
    })

    trunc_lqr = TruncatedLQR(TRAIN_KWARGS, 
                                gen_data, 
                                D, 
                                M, 
                                rand_seed=RAND_SEED)
    trunc_lqr.fit()

    A_yao, B_yao = trunc_lqr.A_, trunc_lqr.B_
    A_sarah_dean_plev, B_sarah_dean_plev, A_sarah_dean_ols, B_sarah_dean_ols = calc_sarah_dean(TRAIN_KWARGS, 
                                                                                                gen_data, 
                                                                                                NUM_TRAJ, 
                                                                                                D, 
                                                                                                M, 
                                                                                                rand_seed=RAND_SEED)

    A_yao_spec_norm = calc_spectral_norm(A_yao - A)
    B_yao_spec_norm = calc_spectral_norm(B_yao - B)

    print(f'A yao spectral norm: {A_yao_spec_norm}')
    print(f'B yao spectral norm: {B_yao_spec_norm}')

    A_sd_plevr_spec_norm = calc_spectral_norm(A_sarah_dean_plev - A)
    B_sd_plevr_spec_norm = calc_spectral_norm(B_sarah_dean_plev - B)

    print(f'A sd plevr spectral norm: {A_sd_plevr_spec_norm}')
    print(f'B sd plevr spectral norm: {B_sd_plevr_spec_norm}')

    A_sd_ols_spec_norm = calc_spectral_norm(A_sarah_dean_ols - A)
    B_sd_ols_spec_norm = calc_spectral_norm(B_sarah_dean_ols - B)

    print(f'A sd ols spectral norm: {A_sd_ols_spec_norm}')
    print(f'B sd ols spectral norm: {B_sd_ols_spec_norm}')

    assert A_yao_spec_norm < A_sd_ols_spec_norm, f"A yao spectral norm is: {A_yao_spec_norm}, and A sarah dean ols spectral norm is: {A_sd_ols_spec_norm}"
    assert B_yao_spec_norm < B_sd_ols_spec_norm, f"B yao spectral norm is: {B_yao_spec_norm}, and B sarah dean ols spectral norm is: {B_sd_ols_spec_norm}"
       
    assert A_yao_spec_norm < A_sd_plevr_spec_norm, f"A yao spectral norm is: {A_yao_spec_norm}, and A sarah dean plevrakis spectral norm is: {A_sd_plevr_spec_norm}"
    assert B_yao_spec_norm < B_sd_plevr_spec_norm, f"B yao spectral norm is: {B_yao_spec_norm}, and B sarah dean plevrakis spectral norm is: {B_sd_plevr_spec_norm}"