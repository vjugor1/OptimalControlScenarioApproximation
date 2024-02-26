import numpy as np
from scipy import stats
import cvxpy as cp


import sys

sys.path.append("..")
import os

# print(os.listdir())
from src.samplers import utils as sampling
from src.solvers import scenario_approx as SA


def solve_glpk(eqs, ineqs, x0, c):
    """Solves linear program
        min c^T x
        x
        s.t. scenario_Gamma.dot(x) <= scenario_Beta

    Args:
        scenario_Gamma (np.ndarray): matrix of linear constraints - upper bounds
        scenario_Beta (np.ndarray): vector of linear constraints - upper bounds
        c (np.ndarray): cost function vector
        x0 (np.ndarray): initial guess

    Returns:
        tuple: solution and solution status from GLPK
    """
    # x0 = np.hstack((np.zeros(Gamma.shape[1]), alpha_0))
    x = cp.Variable(len(x0))
    x.value = x0
    obj = cp.Minimize(cp.sum(cp.multiply(x, c)))

    constraints = []
    for idx, ineq in enumerate(ineqs):
        constraints += [ineq[0] @ x <= ineq[1]]
    if eqs is not None:
        constraints += [eqs[0][0] @ x == eqs[0][1]]
    prob = cp.Problem(obj, constraints)
    # prob.solve(
    #     solver=cp.GLPK, verbose=False, glpk={"msg_lev": "GLP_MSG_OFF"}, warm_start=True,
    # )
    prob.solve(
        solver=cp.SCIPY,
        verbose=False,
        scipy_options={"method": "highs"},
        warm_start=True,
    )
    # print("opt val = ", prob.value)
    sol = prob.solution.primal_vars
    x_opt = np.array(list(sol.values()))
    # return x_opt, prob.solution.status
    return x_opt, prob.solution.status


def solve_approximations(
    Gamma: np.ndarray,
    Beta: np.ndarray,
    Pi_tau_sample: np.ndarray,
    Delta_poly: np.ndarray,
    t_factors: np.ndarray,
    ramp_up_down: np.ndarray,
    T: int,
    alpha_0: np.ndarray,
    delta_alpha: np.ndarray,
    N: int,
    c: np.ndarray,
    eta: float,
    x0: dict,
    optimize_samples: bool,
):
    """Utils function that solves with N scenarios
        1) Ordinary scenario approx
        2) Scenario approx with slack constraints
        3) SAIMIN scenario approx
        for
        min c^T x
        x
        s.t. Prob{Gamma.dot(x) + A.dot(chi) <= Beta} >= 1 - eta
    Args:
        Gamma (np.ndarray): Matrix of linear inequalities - upper bound
        Beta (np.ndarray): Vector of linear inequalities - upper bound
        A (np.ndarray): Matrix before standard random vector
        N (int): Number of samples
        c (np.ndarray): Cost function vector
        eta (float): Reliability level
        x0 (dict): initial guesses for methods - use for warm start
        optimize_samples (bool): for each unique normal in Gamma, keeps only the constant Beta that can be potentially active -- exclude planes that are 100% inactive

    Returns:
        dict: dictionary of optimal points and their statuses
    """
    # Gamma, Beta, A, N, c, eta, x0 = args
    # assert np.allclose(
    #     np.linalg.norm(A, axis=1), np.ones(A.shape[0])
    # ), "Planes (chi - random vector) normal vectors must have unit lengths"
    # SAIMIN
    samples_SAIMIN = (
        sampling.get_samples_SAIMIN(N, eta, len(Delta_poly), Pi_tau_sample, Delta_poly)
        * t_factors
    )
    ineqs, eqs = SA.SA_constr_control(
        Gamma, Beta, ramp_up_down, T, alpha_0, delta_alpha, samples_SAIMIN
    )
    SAIMIN_sol, SAIMIN_status = solve_glpk(eqs, ineqs, x0, c)

    # SA ordinary

    samples_SA = stats.multivariate_normal(cov=np.diag(t_factors ** 2)).rvs(
        size=N
    )  # sampling.get_samples_SAIMIN(N, eta, len(Delta_poly), Pi_tau_sample, Delta_poly) * t_factors
    ineqs, eqs = SA.SA_constr_control(
        Gamma, Beta, ramp_up_down, T, alpha_0, delta_alpha, samples_SA
    )
    SCSA_sol, SCSA_status = solve_glpk(eqs, ineqs, x0, c)

    out_dict = {
        "SCSA": [SCSA_sol.flatten(), SCSA_status],
        "SAIMIN": [SAIMIN_sol.flatten(), SAIMIN_status],
    }
    # json compatibility
    for k__ in out_dict.keys():
        out_dict[k__][0] = [float(v) for v in out_dict[k__][0]]
    status = True
    for v in out_dict.values():
        status = status and v[1]
    assert status
    return out_dict
