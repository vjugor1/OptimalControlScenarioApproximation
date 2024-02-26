from scipy import stats
import numpy as np
from src.samplers.importance_sampler import *


def get_sampling_poly(Gamma, Beta, alpha_0, T, eta, sigmas_sq):
    kappa_t = sigmas_sq.cumsum()[Gamma.shape[1] - 1 :][
        :: Gamma.shape[1]
    ]  # equivalent to np.array([sigmas[:i, :] for i in range(T)])
    # assert np.allclose(chi_t, np.array([sigmas_sq[:i, :] for i in range(1,T)]))
    Phi_inv = stats.norm.ppf(1 - eta)

    t_factors = np.sqrt(kappa_t)

    Gamma_snapshots = np.tile(Gamma, (T, 1))
    Beta_snapshots = np.tile(Beta, (T, 1)).flatten()
    Pi_tau = np.tile(np.eye(T), (Gamma.shape[0], 1))
    # Pi_tau = (Gamma_snapshots @ alpha_0).reshape(-1, 1) * np.array(sorted(Pi_tau, key=lambda x: np.argmax(x))).shape
    Pi_tau = np.array(sorted(Pi_tau, key=lambda x: np.argmax(x)))
    Pi_tau_sample = (
        (Gamma_snapshots @ alpha_0).reshape(-1, 1)
        * Pi_tau
        * np.sign(np.tile(Gamma @ alpha_0, (T)).reshape(-1, 1))
        * Pi_tau
    )
    # Delta_poly = (
    #     (t_factors.reshape(t_factors.shape[0], -1) * np.abs(Gamma @ alpha_0)) * Phi_inv
    # ).flatten()
    Delta_poly = (
        t_factors.reshape(-1, 1) * np.tile(np.abs(Gamma @ alpha_0) * Phi_inv, (T, 1))
    ).flatten()
    Pi_tau_sample /= np.linalg.norm(Pi_tau_sample, axis=1).reshape(-1, 1)
    Delta_poly /= np.linalg.norm(Pi_tau_sample, axis=1)
    # Pi_tau_sample @ xi <= delta_poly - useless polytope
    return np.vstack((Pi_tau_sample,-Pi_tau_sample)), np.hstack((Delta_poly, Delta_poly))


def get_samples_SAIMIN(
    N: int, eta: float, J: int, A: np.ndarray, Delta_poly: np.ndarray
):
    """Yield samples outside of "useless samples boundary"
        One has inequalities
        A chi + Gamma x <= Beta
        to be satisfied with probability 1 - eta
        "useless samples boundary" is defined as
        A chi <= - Phi^-1 (eta) * Delta_poly
        Delta = np.vstack([- Phi^-1 (eta) * Gamma @ alpha_0 * sigma_i^2, i = 1, ..., T])

    Args:
        N (int): Number of samples
        eta (float): reliability level
        J (int): number of planes in feasibility set
        A (np.ndarray): A from docs
        Delta_poly (np.ndarray): RHS of the ''useless samples'' polytope

    Returns:
        np.ndarray: SAIMIN samples - outside of "useless samples boundary"
    """
    # Phi_inv = stats.norm.ppf(eta)
    # Beta_P = Delta_poly * (-Phi_inv)
    sampler = ConditionedPolytopeGaussianSampler(A, Delta_poly)
    generator = sampler.sample()
    samples_SAIMIN = np.array([next(generator) for s in range(N)])
    return samples_SAIMIN


# def prepare_planes_OOS(x, Gamma, Beta, ramp_up_down, T):
#     alpha = x[Gamma.shape[1] :]
#     Gamma_snapshots_stacked = np.tile(Gamma, (T, 1))
#     Beta_snapshots = np.tile(Beta, (T, 1)).flatten()
#     Pi_tau = np.tile(np.eye(T), (Gamma.shape[0], 1))
#     Pi_tau = np.array(sorted(Pi_tau, key=lambda x: np.argmax(x)))
#     Pi_tau_sample = (
#         (Gamma_snapshots_stacked @ alpha).reshape(-1, 1)
#         * Pi_tau
#         * np.sign(np.tile(Gamma @ alpha, (T)).reshape(-1, 1))
#         * Pi_tau
#     )
#     Pi_tau_ramp = np.tile(np.eye(T), (Gamma.shape[1], 1))
#     Pi_tau_ramp = np.array(sorted(Pi_tau_ramp, key=lambda x: np.argmax(x)))
#     I_alpha = np.hstack(
#         (np.zeros((Gamma.shape[1], Gamma.shape[1])), np.eye(Gamma.shape[1]))
#     )
#     # only those inequalities that contain uncertainty
#     # grid feasibility
#     Gamma_snap_feas = np.hstack((Gamma_snapshots_stacked, Gamma_snapshots_stacked))
#     # Beta_snap_feas  = np.hstack(Beta_P_snapshots)
#     Beta_snap_feas = np.hstack(Beta_snapshots)
#     # ramp up, ramp down
#     rampup_feas = np.tile(I_alpha, (T, 1))
#     rampdown_feas = -np.tile(I_alpha, (T, 1))
#     ramp_rhs = np.tile(ramp_up_down, (T, 1)).flatten()
#     ramp_rhs = np.hstack((ramp_rhs, ramp_rhs))
#     # ramp_up_matrix_scenarios = np.vstack([np.tile(I_alpha, (T, 1)) * (Pi_tau_ramp @ samples_SAIMIN[sc_idx]).reshape(-1, 1) for sc_idx in range(samples_SAIMIN.shape[0])])
#     # ramp_down_matrix_scenarios = np.vstack([np.tile(-I_alpha, (T, 1)) * (Pi_tau_ramp @ samples_SAIMIN[sc_idx]).reshape(-1, 1) for sc_idx in range(samples_SAIMIN.shape[0])])
#     # ramp_rhs_scenarios = np.tile(np.tile(ramp_up_down, (T, 1)).flatten(), (N, 1)).flatten()
#     Gamma_OOS = np.vstack((Gamma_snap_feas, rampup_feas, rampdown_feas))
#     rhs_OOS = np.hstack((Beta_snap_feas, ramp_rhs))

#     # write comment here on human language - snapshot projection
#     Pi_OOS = np.vstack(
#         (
#             (Gamma_snapshots_stacked @ x[len(x) // 2 :]).reshape(-1, 1) * Pi_tau,
#             Pi_tau_ramp,
#             Pi_tau_ramp,
#         )
#     )
#     return Gamma_OOS, rhs_OOS, Pi_OOS


def check_feasibility_out_of_sample(
    x: np.ndarray,
    Gamma: np.ndarray,
    ramp_up_down: np.ndarray,
    Beta: np.ndarray,
    T: int,
    # A: np.ndarray,
    t_factors: np.ndarray,
    N: int = 1000,
) -> float:
    """AI is creating summary for check_feasibility_out_of_sample

    Args:
        x (np.ndarray): Current control variable value
        Gamma (np.ndarray): Plane normals for inequalities
        Beta (np.ndarray): Plane constants for inequalities
        A (np.ndarray): Plane normals for inequalities - standard gaussian component
        N (int, optional): Number of samples. Defaults to 1000.

    Returns:
        float: Probability of `x` being feasible estimate
    """
    # Sample from nominal
    # samples = np.random.multivariate_normal(
    #     np.zeros(A.shape[1]), np.eye(A.shape[1]), size=N
    # )
    samples = np.random.multivariate_normal(
        np.zeros(T), np.diag(t_factors) ** 2, size=N
    )
    alpha_to = x[len(x) // 2 :]
    Gamma_alpha = Gamma @ alpha_to
    Gamma_0 = np.vstack((Gamma, np.zeros((2 * len(alpha_to), len(alpha_to)))))
    Gamma_OOS = np.tile(Gamma_0, (T, 1))
    Gamma_alpha_alpha = np.hstack((Gamma_alpha, alpha_to, -alpha_to))
    Gamma_alpha_alpha_OOS = np.tile(Gamma_alpha_alpha, (T, 1))

    rhs_OOS = np.hstack((Beta, ramp_up_down, ramp_up_down))
    rhs_OOS = np.tile(rhs_OOS, T)
    res = np.tile(
        samples.flatten(), (Gamma_alpha_alpha_OOS.shape[1], 1)
    ).T  # .reshape(samples.shape[0], T, Gamma_alpha_alpha_OOS.shape[1])#.reshape(samples.shape[0], -1)
    # Gamma_alpha_alpha_OOS.T.flatten()*res
    res = (
        res[:].reshape(samples.shape[0], res.shape[0] // samples.shape[0], res.shape[1])
        * Gamma_alpha_alpha_OOS
    ).reshape(
        samples.shape[0], -1
    )  # .reshape().shape#.flatten()
    prob_estimate = (
        ((Gamma_OOS @ x[: len(alpha_to)] + res) - rhs_OOS <= 0).all(axis=1).mean()
    )
    # # Assess deterministic feasibility of x
    # feas_det = Gamma[:, : len(x) // 2].dot(x[: len(x) // 2]) - Beta  # (J,)
    # # Multiply each sample by A matrix
    # A_dot_samples = A.dot(samples.T)  # (J, N)
    # # Tile `feas_det` to add to A_dot_sample in one operation
    # feas_det_tiled = np.tile((feas_det), N).reshape(-1, N)  # (J, N)
    # # Add up, compare to zero and check if all satisfied to samples
    # sample_res = ((feas_det_tiled + A_dot_samples) <= 0.0).all(axis=0)
    # prob_estimate = sample_res.sum() / N

    return prob_estimate
