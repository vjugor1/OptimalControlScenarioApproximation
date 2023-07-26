import numpy as np
from scipy import stats


# def SA_constr_control(Gamma, Beta, ramp_up_down, T, alpha_0, delta_alpha, samples):
#     # time indepedent
#     N = len(samples)
#     ## initial gen pos
#     Gamma_snapshots = np.tile(Gamma, (T, 1))
#     Beta_snapshots = np.tile(Beta, (T, 1)).flatten()
#     Pi_tau = np.tile(np.eye(T), (Gamma.shape[0], 1))
#     Pi_tau = np.array(sorted(Pi_tau, key=lambda x: np.argmax(x)))
#     Gamma_xalpha = np.hstack((Gamma, np.zeros(Gamma.shape)))
#     ineq_init_gen = (Gamma_xalpha, Beta)
#     ## participation factors: S^1_\Delta
#     One_alpha = np.hstack((np.zeros(Gamma.shape[1]), np.ones(Gamma.shape[1])))
#     eq_pfs = (One_alpha, 1)
#     I_alpha = np.hstack(
#         (np.zeros((Gamma.shape[1], Gamma.shape[1])), np.eye(Gamma.shape[1]))
#     )
#     ineq_pfs = (-I_alpha, np.zeros(I_alpha.shape[0]))
#     ## proximity to initial alpha_0: |alpha_i - alpha_0i| leq delta_alpha_i
#     up_down_prox_matrix = np.vstack((I_alpha, -I_alpha))
#     up_down_prox_rhs = np.hstack((delta_alpha + alpha_0, delta_alpha - alpha_0))
#     ineq_prox_alpha = (up_down_prox_matrix, up_down_prox_rhs)
#     # time dependent
#     ## feasibility - scenarios
#     Gamma_snapshots_stacked = np.vstack(Gamma_snapshots)
#     Gamma_xalpha_scenarios = np.vstack(
#         [
#             np.hstack(
#                 (
#                     Gamma_snapshots_stacked,
#                     (
#                         Gamma_snapshots_stacked
#                         * (Pi_tau @ samples[sc_idx]).reshape(-1, 1)
#                     ),
#                 )
#             )
#             for sc_idx in range(samples.shape[0])
#         ]
#     )
#     # Beta_scenarios = np.tile(np.hstack(Beta_P_snapshots), (N, 1)).flatten()#np.vstack(Beta_P_snapshots)
#     Beta_scenarios = np.tile(
#         np.hstack(Beta_snapshots), (N, 1)
#     ).flatten()  # np.vstack(Beta_P_snapshots)
#     ineqs_feas_scen = (Gamma_xalpha_scenarios, Beta_scenarios)
#     ## ramp-up, ramp-down / \sqrt(sigma)

#     # Pi_tau_ramp = np.tile(np.eye(T), (Gamma.shape[1], 1))
#     # Pi_tau_ramp = np.array(sorted(Pi_tau_ramp, key=lambda x: np.argmax(x)))
#     # ramp_up_matrix_scenarios = np.vstack(
#     #     [
#     #         np.tile(I_alpha, (T, 1)) * (Pi_tau_ramp @ samples[sc_idx]).reshape(-1, 1)
#     #         for sc_idx in range(samples.shape[0])
#     #     ]
#     # )
#     # ramp_down_matrix_scenarios = np.vstack(
#     #     [
#     #         np.tile(-I_alpha, (T, 1)) * (Pi_tau_ramp @ samples[sc_idx]).reshape(-1, 1)
#     #         for sc_idx in range(samples.shape[0])
#     #     ]
#     # )
#     # ramp_rhs_scenarios = np.tile(
#     #     np.tile(ramp_up_down, (T, 1)).flatten(), (N, 1)
#     # ).flatten()
#     ramp_up_matrix_scenarios = I_alpha
#     ramp_down_matrix_scenarios = -I_alpha
#     ramp_rhs_scenarios = np.hstack((ramp_up_down, ramp_up_down)) / np.max(
#         np.abs(samples)
#     )
#     ineqs_ramp_up = (ramp_up_matrix_scenarios, ramp_rhs_scenarios)
#     ineqs_ramp_down = (ramp_down_matrix_scenarios, ramp_rhs_scenarios)

#     ineqs = [
#         ineq_init_gen,
#         ineq_pfs,
#         ineq_prox_alpha,
#         ineqs_ramp_up,
#         ineqs_ramp_down,
#         ineqs_feas_scen,
#     ]
#     eqs = [eq_pfs]
#     return ineqs, eqs
def SA_constr_control(Gamma, Beta, ramp_up_down, T, alpha_0, delta_alpha, samples):

    n = len(alpha_0)
    I_alpha = np.hstack((np.zeros((n, n)), np.eye(n)))
    I_x = np.hstack((np.eye(n), np.zeros((n, n))))
    # initial generation feasibility
    ## matrix
    Gamma_feas = np.hstack((Gamma, np.zeros(Gamma.shape)))
    ## RHS
    rhs_feas = Beta

    # proximity
    Gamma_pfs = np.vstack((I_alpha, -I_alpha))
    rhs_pfs = np.hstack((delta_alpha + alpha_0, delta_alpha - alpha_0))

    # participation factors \in \S^1_{\Delta}
    Gamma_simplex = -I_alpha
    rhs_simplex = np.zeros(I_alpha.shape[0])

    # trajectory feasibility
    ## below feasibility and ramp_up_down constraints are going
    ### x part
    Gamma_0 = np.vstack((Gamma, np.zeros((2 * n, n))))
    Gamma_0_T = np.tile(Gamma_0, (T, 1))
    ### alpha part
    Gamma_I = np.vstack((Gamma, np.eye(n), -np.eye(n)))
    Gamma_I_T = np.tile(Gamma_I, (T, 1)).reshape(T, Gamma_I.shape[0], Gamma_I.shape[1])
    ### applying scenarios to the alpha part
    res = np.tile(samples.flatten(), (Gamma_I_T.shape[-1], Gamma_I_T.shape[-2], 1)).T
    res = (
        res.reshape(
            samples.shape[0],
            res.shape[0] // samples.shape[0],
            res.shape[1],
            res.shape[2],
        )
        * Gamma_I_T
    )  # .reshape().shape#.flatten()
    Gamma_I_T_scens = res.reshape(-1, res.shape[-1])
    ## Assembling
    Gamma_0_T_scens = np.tile(Gamma_0_T, (samples.shape[0], 1))
    Gamma_traj = np.hstack((Gamma_0_T_scens, Gamma_I_T_scens))
    ## RHS
    rhs_traj = np.hstack((Beta, ramp_up_down, ramp_up_down))
    rhs_traj = np.tile(np.tile(rhs_traj, T), len(samples))

    # total ineqs stack
    Gamma_total = np.vstack((Gamma_feas, Gamma_pfs, Gamma_simplex, Gamma_traj))
    rhs_total = np.hstack((rhs_feas, rhs_pfs, rhs_simplex, rhs_traj))

    ineqs = [(Gamma_total, rhs_total)]

    # equalities
    eqs = [(np.hstack((np.zeros(n), np.ones(n))), 1)]

    return ineqs, eqs


def get_scenario_approx_constraints(
    Gamma: np.ndarray,
    Beta: np.ndarray,
    A: np.ndarray,
    samples: np.ndarray,
    optimize_samples: bool,
    include_slack: bool = False,
    eta: float = 0.01,
):
    """Assembles scenario approximation of chance constraint of a form
        `Gamma x <= Beta - A \chi`, where `x` is a control variable, `\chi` is a random variable

    Args:
        Gamma (numpy.ndarray): 2d array of normals. See description
        Beta (numpy.ndarray): 1d array of constants. See description
        A (numpy.ndarray): 2d array. See description
        samples (numpy.ndarray: scenarios or realization of random variable `\chi` to be used in scenario approximation
        optimize_samples (bool): for each unique normal in Gamma, keeps only the constant Beta that can be potentially active -- exclude planes that are 100% inactive
        include_slack (bool): if to include \cO - slack constraints
        eta: (float): confidence level for original chance constraint
    Returns:
        tuple: matrix of normals and vector of constants for scenario approximation
    """

    if optimize_samples:
        out_Gamma = Gamma
        out_Beta = Beta - (A.dot(samples.T)).max(axis=1)
    else:
        out_Gamma = np.concatenate([Gamma for i in range(samples.shape[0])], axis=0)
        out_Beta = np.concatenate(
            [Beta - A.dot(samples[i]) for i in range(samples.shape[0])], axis=0
        )
    if include_slack:
        Phi_inv = stats.norm.ppf(eta)
        Beta_O = Beta + Phi_inv
        out_Gamma = np.concatenate([out_Gamma, Gamma], axis=0)
        out_Beta = np.concatenate([out_Beta, Beta_O], axis=0)

    return out_Gamma, out_Beta
