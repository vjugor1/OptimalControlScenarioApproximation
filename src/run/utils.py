import numpy as np
from tqdm import tqdm
import os, json
import pandas as pd

from src.data_utils import grid_data
from src.samplers import preprocessing as pre
from src.solvers import scenario_approx as SA
from src.solvers import utils as SU
from src.samplers.utils import check_feasibility_out_of_sample
from src.samplers import utils as sampling
from scipy import stats
import cvxpy as cp


def initialize(grid_name: str, eta: float) -> tuple:

    (
        Gamma,
        Beta,
        gens,
        cost_coeffs,
        cost_correction_term,
        cost_dc_opf,
    ) = grid_data.get_linear_constraints(grid_name, check_pp_vs_new_form=True)
    x0 = gens[1:]
    # print(Gamma, Beta)
    mu = np.zeros(len(gens) - 1)
    Sigma = np.eye(len(gens) - 1) * 0.01
    # making matrix psd
    Sigma = Sigma.dot(Sigma.T)
    # A = Gamma
    Gamma, Beta, A = pre.standartize(Gamma, Beta, mu, Sigma)
    # print(len(Gamma), len(Beta))
    J = Gamma.shape[0]
    # print(len(A))
    c = cost_coeffs

    return Gamma, A, Beta, gens, x0, c, J, cost_correction_term, cost_dc_opf, mu, Sigma


def initialize_multistep(grid_name: str, eta: float, T: int):
    (
        Gamma,
        Beta,
        gens,
        cost_coeffs,
        cost_correction_term,
        cost_dc_opf,
    ) = grid_data.get_linear_constraints(grid_name, check_pp_vs_new_form=False)
    x0 = gens[1:]
    alpha_0 = np.ones(len(x0)) / len(x0)  # np.array([0.5, 0.5])
    x0 = np.hstack((x0, alpha_0))
    print(Gamma, Beta)
    mu = np.zeros(len(gens) - 1)
    Sigma = np.eye(len(gens) - 1) * 0.01
    # making matrix psd
    Sigma = Sigma.dot(Sigma.T)
    # A = Gamma
    Gamma, Beta, A = pre.standartize(Gamma, Beta, mu, Sigma)
    print(len(Gamma), len(Beta))
    J = Gamma.shape[0]
    print(len(A))
    c = cost_coeffs
    c = np.hstack((c, np.zeros(len(c))))
    sigmas_sq = np.ones((T, Gamma.shape[1])) * 0.001
    Sigma = sigmas_sq
    kappa_t = sigmas_sq.cumsum()[Gamma.shape[1] - 1 :][
        :: Gamma.shape[1]
    ]  # equivalent to np.array([sigmas[:i, :] for i in range(T)])

    t_factors = np.sqrt(kappa_t)

    Pi_tau_sample, Delta_poly = sampling.get_sampling_poly(
        Gamma, Beta, np.ones(len(alpha_0)), T, eta, sigmas_sq
    )
    # N = 400
    ramp_up_down = np.ones(len(x0) // 2) * 2  # np.array([5, 7])
    delta_alpha = np.ones(len(x0) // 2) * 0.2

    return (
        Gamma,
        A,
        Beta,
        gens,
        x0,
        alpha_0,
        delta_alpha,
        c,
        J,
        cost_correction_term,
        cost_dc_opf,
        mu,
        Sigma,
        t_factors,
        Delta_poly,
        Pi_tau_sample,
        ramp_up_down,
    )


def generate_samples_multistep(
    N0: int,
    ks: list,
    t_factors: list,
    L: int,
    eta: float,
    Delta_poly: np.ndarray,
    Pi_tau_sample: np.ndarray,
):
    all_samples_SAIMIN = np.empty((N0 * ks[-1], len(t_factors), L))
    for l in range(L):
        samples_SAIMIN = (
            sampling.get_samples_SAIMIN(
                N0 * ks[-1], eta, len(Delta_poly), Pi_tau_sample, Delta_poly
            )
            * t_factors
        )
        all_samples_SAIMIN[:, :, l] = samples_SAIMIN
    all_samples_SCSA = (
        stats.multivariate_normal(cov=np.diag(t_factors ** 2))
        .rvs(size=(N0 * ks[-1], L))
        .transpose(0, 2, 1)
    )
    assert all_samples_SCSA.shape == (N0 * ks[-1], len(t_factors), L)

    return all_samples_SAIMIN, all_samples_SCSA


def multiple_solve(
    x0,
    N0,
    ks,
    L,
    all_samples_SAIMIN,
    all_samples_SCSA,
    Sigma,
    mu,
    Gamma,
    Beta,
    ramp_up_down,
    T,
    alpha_0,
    delta_alpha,
    c,
):
    results = {
        "Sigma": [[float(v) for v in row] for row in Sigma],
        "mu": [float(v) for v in mu],
    }
    x0_dict = {"SAIMIN": x0, "SCSA": x0}
    for k in tqdm(ks):
        N = N0 * k
        print(k, " / ", ks[-1])
        for l in range(L):

            samples_SAIMIN = all_samples_SAIMIN[:N, :, l]
            ineqs, eqs = SA.SA_constr_control(
                Gamma, Beta, ramp_up_down, T, alpha_0, delta_alpha, samples_SAIMIN
            )
            try:
                SAIMIN_sol, SAIMIN_status = SU.solve_glpk(eqs, ineqs, x0, c)
            except cp.error.SolverError:
                pass

            samples_SCSA = all_samples_SCSA[:N, :, l]
            ineqs, eqs = SA.SA_constr_control(
                Gamma, Beta, ramp_up_down, T, alpha_0, delta_alpha, samples_SCSA
            )
            try:
                SCSA_sol, SCSA_status = SU.solve_glpk(eqs, ineqs, x0, c)
            except cp.error.SolverError:
                pass
            res = {
                "SCSA": [SCSA_sol.flatten(), SCSA_status],
                "SAIMIN": [SAIMIN_sol.flatten(), SAIMIN_status],
            }
            for k__ in res.keys():
                res[k__][0] = [float(v) for v in res[k__][0]]
            status = True
            for v in res.values():
                status = status and v[1]
            assert status
            for k in x0_dict.keys():
                x0_dict[k] = res[k][0]
            try:
                results[N].append(res)
            except KeyError:
                results[N] = []
                results[N].append(res)
        print("Finished N = ", N)
    return results


def map_names(
    results, new_names=["SA-ScenarioApprox", "SAIS-ScenarioApproxImportanceSampling"],
):
    for r in results.keys():
        if r not in ["Sigma", "mu"]:
            for l in results[r]:
                keys = list(l.keys())
                for i in range(len(keys)):
                    l[new_names[i]] = l.pop(keys[i])
    return results


def save_results(save_dir, results, N0, ks, eta):
    json_file = os.path.join(
        # "N_" + str(N0 * ks[-1]) + "_eta_" + str(np.round(eta, 2)) + ".json"
        "N_"
        + str(N0 * ks[-1])
        + "_eta_"
        + str(np.round(eta, 2))
        + ".json"
    )
    try:
        with open(os.path.join(save_dir, json_file), "w") as fp:
            json.dump(results, fp, indent=4)
    except FileNotFoundError:
        os.makedirs(save_dir)
        with open(os.path.join(save_dir, json_file), "w") as fp:
            json.dump(results, fp, indent=4)


def unpack_results(results, c, k, cost_correction_term, A, N0):
    try:
        names = list(results[N0][k].keys())
    except KeyError:
        names = list(results[str(N0)][k].keys())
    fns = []
    xs = []
    for r in results.keys():
        if r not in ["Sigma", "mu"]:
            for v in results[r][k].values():
                try:
                    if len(v[0]) == 0:
                        xs.append(np.zeros(A.shape[1] * 2))
                    else:
                        xs.append(v[0])
                    fns.append(np.dot(xs[-1], c) + cost_correction_term)
                except ValueError:
                    fns.append(np.nan)
    fns = np.array(fns).reshape(-1, len(names))
    xs = np.array(xs).reshape(-1, len(names), A.shape[1] * 2)
    return fns, xs, names


def estimate_probs(results, eta, Gamma, ramp_up_down, Beta, L, T, t_factors, ks, c, cost_correction_term, A, N0):
    scenario_prob_estimate = np.zeros((len(names), len(ks)))
    scenario_probs_several_starts = []
    for k in tqdm(range(L)):
        fns, xs, names = unpack_results(
            results=results, c=c, cost_correction_term=cost_correction_term, k=k, A=A, N0=N0
        )
        scenarios_probs = np.array(
            [
                np.apply_along_axis(
                    arr=xs[:, i, :],
                    func1d=lambda x: check_feasibility_out_of_sample(
                        np.around(x, 5), Gamma, ramp_up_down, Beta, T, t_factors, 10000
                    ),
                    axis=1,
                )
                for i in range(len(names))
            ]
        )

        scenario_prob_estimate += scenarios_probs - (1 - eta) >= 0.0
        scenario_probs_several_starts.append(scenarios_probs)
    scenario_prob_esimate = scenario_prob_estimate / L
    scenario_probs_several_starts = np.array(np.stack(scenario_probs_several_starts))

    return scenario_probs_several_starts

def estimates_to_pandas(scenario_probs_several_starts, ks, N0, names, eta, save_dir):
    pd_boxplot = pd.DataFrame({"N": [], "Method": [], r"$(\hat{\mathbb{P}}_N)_l$": []})
    for method_idx in range(scenario_probs_several_starts.shape[1]):
        data = scenario_probs_several_starts[:, method_idx, :]
        pd_boxplot_tmp = pd.DataFrame(
            {"N": [], "Method": [], r"$(\hat{\mathbb{P}}_N)_l$": []}
        )
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                pd_boxplot_tmp = pd.concat(
                    [
                        pd_boxplot_tmp,
                        pd.DataFrame(
                            {
                                "N": [ks[j] * N0],
                                "Method": [names[method_idx]],
                                r"$(\hat{\mathbb{P}}_N)_l$": [data[i, j]],
                            }
                        ),
                    ],
                    ignore_index=True,
                )
        pd_boxplot = pd.concat([pd_boxplot, pd_boxplot_tmp])
    # save to csv
    pandas_name = (
        "multistarts_N_" + str(N0 * ks[-1]) + "_eta_" + str(np.round(eta, 2)) + ".csv"
    )
    # save to csv
    # pandas_name = 'multistarts.csv'
    pd_boxplot.to_csv(os.path.join(save_dir, pandas_name))
    return pd_boxplot