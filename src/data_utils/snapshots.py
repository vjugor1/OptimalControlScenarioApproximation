import numpy as np
from scipy import stats


def sampling_polytope_fixed_alpha(Gamma, Beta, T, sigmas_sq, eta, alpha_tilde):
    Phi_inv = stats.norm.ppf(1-eta)
    # Gamma_snapshots = np.tile(Gamma, (T, 1))
    # Beta_snapshots = np.tile(Beta, (T, 1)).flatten()
    Pi_tau = np.tile(np.eye(T), (Gamma.shape[0], 1))
    Pi_tau = np.array(sorted(Pi_tau, key=lambda x: np.argmax(x)))
    kappa_t = get_vars_brownian(Gamma.shape[1], sigmas_sq)
    t_factors = np.sqrt(kappa_t)
    Delta_poly = (
        (t_factors.reshape(t_factors.shape[0], -1) * (Gamma @ alpha_tilde)) * Phi_inv
    ).flatten()
    # Pi_tau @ xi <= delta_poly - useless polytope
    return Pi_tau, Delta_poly


def get_vars_brownian(dim_x0: int, sigmas_sq: np.ndarray):
    kappa_t = sigmas_sq.cumsum()[dim_x0 - 1 :][
        ::dim_x0
    ]  # equivalent to np.array([sigmas[:i, :] for i in range(T)])
    # assert np.allclose(chi_t, np.array([sigmas_sq[:i, :] for i in range(1,T)]))
    return kappa_t


def get_snapshots_planes(
    Gamma: np.ndarray,
    Beta: np.ndarray,
    sigmas_sq: np.ndarray,
    alpha_tilde: np.ndarray,
    T: int,
):
    r"""This function generated planes for snapshots via the following law:
        A (x_0 / sqrt(kappa_t) + alpha_tilde * \xi) <= b / sqrt(kappa_t), t = 1, ..., T. Here \xi \sim \mathcal{N}(0, 1)
        kappa_t = \sum_{\tau=1}^t \sum_{i=1}^{|Pg|} \sigma^2_{\tau, k} - variance of sum of normal centered r.v.s -- needs to obtain \xi \sim \mathcal{N}(0, 1).
        Without normalization with kappa_t we would have non-standard r.v. 
        - Can be used for sample out of polytope generation, not for optimization with participation factors 

        Args:
            Gamma (: np.ndarray): matrix `A` from the inequalities above
            Beta (: np.ndarray): vector `b` from the ineqs above
            alpha_tilde (: np.ndarray): AGC participation factors to compensate the uncertainty
            sigmas_sq (: np.ndarray): shape = (T, |Pg|). It contains the variances of flutuations at each generator at each snapshot step
            T (int): Number of snapshots to generate for
        
        Returns:
            (np.ndarray, np.ndarray, np.ndarray): snapshot planes and stochastic prefactor (A / sqrt(kappa), b / sqrt(kappa), A alpha_tilde)
    """
    dim_x0 = Gamma.shape[1]  # number of controllabe generators (without slack)
    Gamma_tiled = np.tile(Gamma, (T, 1))
    Beta_tiled = np.tile(Beta, (T, 1)).flatten()
    kappa_t = get_vars_brownian(dim_x0, sigmas_sq)
    t_factors = np.sqrt(kappa_t)
    t_factors = np.tile(
        t_factors, (Gamma.shape[0], 1)
    ).T.flatten()  # tile for snapshot scaling

    Gamma_snapshots = Gamma_tiled / t_factors.reshape(t_factors.shape[0], -1)
    Beta_snapshots = Beta_tiled / t_factors
    if alpha_tilde is not None:
        A_alpha = Gamma @ alpha_tilde  # np.tile(Gamma @ alpha_tilde, (T, 1)).flatten()
    else:
        A_alpha = None

    return Gamma_snapshots, Beta_snapshots, A_alpha
    # Gamma_snapshots = np.vstack((Gamma, Gamma_snapshots))
    # Beta_snapshots = np.vstack((Beta, Beta_snapshots))
