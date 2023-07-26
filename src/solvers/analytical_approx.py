from scipy import stats
import numpy as np
import scipy.optimize as optim
from src.solvers.barrier_solver import *
import cvxpy as cp


def inner_polyhedral_constraint(
    x, x_An, x_bn, eta,
):
    """
    Returns the value of constraint from the inner polyhedral approximation (Boole's inequality)
    NB!!!! Inequalities must be standartized first, for this approximation to be valid!!!!!!
    args:
        x(n,) array-like floats: control variable (optimization variable)
        x_An(n, m) array-like floats: matrix with rows a_i
        x_bn(n,) array-like-like floats: vector of d_i
        eta float: confidence level, i.e., the probability of violating at least one linear constraint
    """
    # vec = np.dot(x_An, x) - x_bn
    vec = x_An @ x - x_bn
    cdf_vec = [stats.norm.cdf(v) for v in vec]
    sum_ = np.sum(cdf_vec)
    return sum_ - eta


def inner_polyhedral_constraint_grad(x, x_An, x_bn, eta, eta_var=True):
    """return the gradient of
        inner polyhedral (union bound) constraint at (x, eta) or x, depends on eta_var
    Args:
        x (ndarray(n,)): point (e.g., generations)
        x_An (ndarray(m,n)): matrix of linear constraints
        x_bn (ndarray(m,)): vector of R.H.S.s
        eta (float): relibaility parameters
        eta_var (bool, optional): if include gradient on eta into the returning value. Defaults to True.

    Returns:
        ndarray(n,) ndarray(n+2,): grad on x or x and eta_1, eta_2
    """
    sum_term_grad_x = stats.norm.pdf(np.dot(x_An, x) - x_bn).reshape(-1, 1) * x_An
    grad_x = sum_term_grad_x.sum(axis=0)
    grad_eta = np.array([0, -1])
    if eta_var:
        full_grad = np.hstack((grad_x, grad_eta))
        return full_grad
    else:
        return grad_x


def inner_polyhedral(eta, x_An, x_bn, r=1000000.0, x0=[-0.1, -0.1], c=None):
    """
    Returns scipy minimize result of the inner approximation of the original chance constrained problem: ''The optimization result represented as a OptimizeResult object. Important attributes are: x the solution array, success a Boolean flag indicating if the optimizer exited successfully and message which describes the cause of the termination. See OptimizeResult for a description of other attributes.''

    args:
        eta float: confidence level, i.e., the probability of violating at least one linear constraint
        x_An(n, m) array-like floats: matrix with rows a_i
        x_bn(n,) array-like-like floats: vector of d_i
        r float: penalization parameter
        x0 (n,) array-like floats: initial guess
    """
    m = x_An.shape[1]
    if c is None:
        cost_coeffs = np.ones(m)
    else:
        cost_coeffs = c

    res = optim.minimize(
        fun=lambda x: objective_barrier_multiple(
            cost_coeffs,
            x,
            [lambda x_: inner_polyhedral_constraint(x_, x_An, x_bn, eta)],
            r,
        ),
        jac=lambda x: objective_barrier_multiple_grad(
            cost_coeffs,
            x,
            [lambda x_: inner_polyhedral_constraint(x_, x_An, x_bn, eta)],
            [
                lambda x_: inner_polyhedral_constraint_grad(
                    x_, x_An, x_bn, eta, eta_var=False
                )
            ],
            r=r,
            eta_var=False,
        ),
        x0=x0,
        method="Nelder-Mead",
    )
    print("Optimiziation Succeeded:", res.success)

    return res


def scc(x0, c, Gamma, Beta, alpha_0, rampup_down, delta_alpha, T, t_factors, eps):
    # x0 = np.hstack((np.zeros(Gamma.shape[1]), alpha_0))
    x = cp.Variable(len(x0))
    x.value = x0
    obj = cp.Minimize(cp.sum(cp.multiply(x, c)))
    constraints = []
    Phi_inv = stats.norm.ppf(1 - eps)
    # equality
    I_alpha = np.hstack((np.zeros((len(alpha_0), len(alpha_0))), np.eye(len(alpha_0))))
    I_x = np.hstack((np.eye(len(alpha_0)), np.zeros((len(alpha_0), len(alpha_0)))))
    constraints.append(cp.sum(I_alpha @ x) == 1)
    # ineqs feas
    for plane_idx in range(Gamma.shape[0]):
        Gamma_i = Gamma[plane_idx]
        constraints.append(Gamma_i @ (I_x @ x) <= Beta[plane_idx])
    for time_idx in range(T):
        for plane_idx in range(Gamma.shape[0]):
            Gamma_i = Gamma[plane_idx]
            constraints.append(
                Gamma_i @ (I_x @ x)
                + cp.abs(Gamma_i @ (I_alpha @ x)) * t_factors[time_idx] * Phi_inv
                <= Beta[plane_idx]
            )
    # ineqs rampupdown
    for time_idx in range(T):
        for plane_idx in range(I_alpha.shape[0]):
            Gamma_i = I_alpha[plane_idx]
            
            constraints.append(
                Gamma_i @ x <= rampup_down[plane_idx] / t_factors[time_idx] / Phi_inv
            )
            constraints.append(
                -Gamma_i @ x <= rampup_down[plane_idx] / t_factors[time_idx] / Phi_inv
            )
    # proximity
    for plane_idx in range(I_alpha.shape[0]):
        Gamma_i = I_alpha[plane_idx]
        constraints.append(Gamma_i @ x <= delta_alpha[plane_idx] + alpha_0[plane_idx])
        constraints.append(-Gamma_i @ x <= delta_alpha[plane_idx] - alpha_0[plane_idx])
    #
    # for idx, ineq in enumerate(ineqs):
    #     constraints += [
    #         cp.sum(cp.multiply(ineq[0][i], x)) <= ineq[1][i]
    #         for i in range(ineq[0].shape[0])
    #     ]
    # constraints += [cp.sum(cp.multiply(eqs[0][0], x)) == eqs[0][1]]
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.GUROBI, verbose=False, warm_start=True)
    # print("opt val = ", prob.value)
    sol = prob.solution.primal_vars
    x_opt = np.array(list(sol.values()))
    # return x_opt, prob.solution.status
    return x_opt, prob.solution.status
