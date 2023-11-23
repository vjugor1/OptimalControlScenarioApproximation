import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import scipy.optimize as optim
from scipy import stats
import cvxpy as cp
from tqdm import tqdm
import json
from multiprocessing.pool import Pool
import os
import sys
import seaborn as sns
import pandas as pd

sys.path.append("..")

from src.samplers.importance_sampler import *
from src.samplers.utils import check_feasibility_out_of_sample
from src.samplers import preprocessing as pre
from src.data_utils import synthetic as synth
from src.data_utils import plotting
from src.data_utils import snapshots as ss
from src.solvers import scenario_approx as SA
from src.solvers import utils as SU
from src.solvers import analytical_approx as AA
from src.samplers import utils as sampling

mu = np.ones(2)
Sigma = np.array([[1, 0], [0, 1]]) * 0.1
# making matrix psd
Sigma = Sigma.dot(Sigma.T)
J = 5
tau = 10
Gamma, Beta = synth.regular_polyhedron(J, tau)
# A = Gamma
Gamma, Beta, A = pre.standartize(Gamma, Beta, mu, Sigma)
c = np.array([-1, 1, 0, 0])

eta = 0.05
T = 3  # time snapshots
alpha_0 = np.array([0.5, 0.5])
mu = np.zeros(2)
sigmas_sq = np.ones((T, Gamma.shape[1])) * 1
kappa_t = sigmas_sq.cumsum()[Gamma.shape[1] - 1 :][
    :: Gamma.shape[1]
]  # equivalent to np.array([sigmas[:i, :] for i in range(T)])
# assert np.allclose(chi_t, np.array([sigmas_sq[:i, :] for i in range(1,T)]))
t_factors = np.sqrt(kappa_t)
J = 5
tau = 30
Gamma, Beta = synth.regular_polyhedron(J, tau)  # base polyhedron
c = np.array([-1, 1, 0, 0])


Pi_tau_sample, Delta_poly = sampling.get_sampling_poly(
    Gamma, Beta, alpha_0, T, eta, sigmas_sq
)

# N = 400
ramp_up_down = np.array([5, 7])
delta_alpha = np.ones(Gamma.shape[1]) * 0.1
optimize_samples = True
# samples_SAIMIN = sampling.get_samples_SAIMIN(N, eta, len(Delta_poly), Pi_tau_sample, Delta_poly) * t_factors

x0 = np.hstack((np.zeros(Gamma.shape[1]), alpha_0))
# x_opt, prob_status = SU.solve_glpk(eqs, ineqs, x0, c)

x_scc, x_scc_status = AA.scc(
    x0, c, Gamma, Beta, alpha_0, ramp_up_down, delta_alpha, T, t_factors ** 2, (eta) / 2
)

Gamma_OOS, rhs_OOS, Pi_OOS = sampling.prepare_planes_OOS(
    x_scc.flatten(), Gamma, Beta, ramp_up_down, T
)
scc_prob = check_feasibility_out_of_sample(
    x_scc.flatten(), Gamma_OOS, rhs_OOS, Pi_OOS, t_factors, 100000
)
scc_prob

# Store sigma and mu, next, the solutions for approximation will be pushed
results = {
    "Sigma": [[float(v) for v in row] for row in sigmas_sq],
    "mu": [float(v) for v in mu],
}
N0 = 2
# ks = list(range(1, 55))[::5]
ks = list(range(1, 15))[::5]
L = 40

for k in tqdm(ks):
    N = N0 * k
    print(k, "/", ks[-1])

    with Pool() as pool:

        def call_solve():
            return SU.solve_approximations(
                Gamma,
                Beta,
                Pi_tau_sample,
                Delta_poly,
                t_factors ** 2,
                ramp_up_down,
                T,
                alpha_0,
                delta_alpha,
                N,
                c,
                eta,
                x0,
                True,
            )

        results[N] = pool.starmap(
            func=call_solve, iterable=[(i, np.random.randn()) for i in range(L)]
        )
    # for l in range(L):
    #     res = SU.solve_approximations(Gamma, Beta, Pi_tau_sample, Delta_poly, t_factors**2, ramp_up_down, T, alpha_0, delta_alpha, N, c, eta, x0, True)

    #     try:
    #         results[N].append(res)
    #     except KeyError:
    #         results[N] = []
    #         results[N].append(res)


save_dir = os.path.join("saves", "synthetic")
json_file = os.path.join(
    "J_" + str(J) + "_tau_" + str(tau) + "_N_" + str(N0 * ks[-1]) + ".json"
)

try:
    with open(os.path.join(save_dir, json_file), "w") as fp:
        json.dump(results, fp, indent=4)
except FileNotFoundError:
    os.makedirs(save_dir)
    with open(os.path.join(save_dir, json_file), "w") as fp:
        json.dump(results, fp, indent=4)


# load if necessary
# with open(os.path.join(save_dir, json_file), 'r') as fp:
#     results = json.load(fp)


plt.figure(figsize=(15, 5))


def unpack_results(results, k=0):
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
                    xs.append(v[0])
                    fns.append(np.dot(v[0], c))
                except ValueError:
                    fns.append(np.nan)
    fns = np.array(fns).reshape(-1, len(names))
    xs = np.array(xs).reshape(-1, len(names), A.shape[1] * 2)
    return fns, xs, names


fns, xs, names = unpack_results(results=results, k=1)
for i in range(fns.shape[1]):
    plt.plot(np.array(ks) * N0, fns[:, i], label=names[i])

# plt.hlines(y = [res_boole.fun], xmin = 0, xmax = ks[-1] * N0, label='Boole', color='black', linestyle='dotted')
plt.legend()
plt.xlabel("Number of samples")
plt.ylabel("Objective")

print("scenario probs")
scenario_probs = []
for i in range(len(names)):

    def check(x):
        Gamma_OOS, rhs_OOS, Pi_OOS = sampling.prepare_planes_OOS(
            x, Gamma, Beta, ramp_up_down, T
        )
        return check_feasibility_out_of_sample(
            x.flatten(), Gamma_OOS, rhs_OOS, Pi_OOS, 100000
        )

    scenario_probs.append(
        np.apply_along_axis(arr=xs[:, i, :], func1d=lambda x: check(x), axis=1)
    )
scenarios_probs = np.array(scenario_probs)

print("boxplots")
scenario_prob_estimate = np.zeros((len(names), len(ks)))
scenario_probs_several_starts = []
for k in tqdm(range(L)):
    _, xs, names = unpack_results(results=results, k=k)
    # boole_prob = check_feasibility_out_of_sample(res_boole.x, Gamma, Beta, A, 100000)
    def check(x):
        Gamma_OOS, rhs_OOS, Pi_OOS = sampling.prepare_planes_OOS(
            x, Gamma, Beta, ramp_up_down, T
        )
        return check_feasibility_out_of_sample(
            x.flatten(), Gamma_OOS, rhs_OOS, Pi_OOS, 100000
        )

    scenarios_probs = np.array(
        [
            np.apply_along_axis(arr=xs[:, i, :], func1d=lambda x: check(x), axis=1)
            for i in range(len(names))
        ]
    )
    scenario_prob_estimate += scenarios_probs - eta >= 0.0
    scenario_probs_several_starts.append(scenarios_probs)
scenario_prob_esimate = scenario_prob_estimate / L
scenario_probs_several_starts = np.array(np.stack(scenario_probs_several_starts))

pd_boxplot = pd.DataFrame({"N": [], "Method": [], "Prob_est - 1-eta": []})
for method_idx in range(scenario_probs_several_starts.shape[1]):
    data = scenario_probs_several_starts[:, method_idx, :]
    pd_boxplot_tmp = pd.DataFrame({"N": [], "Method": [], "Prob_est - 1-eta": []})
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            pd_boxplot_tmp = pd.concat(
                [
                    pd_boxplot_tmp,
                    pd.DataFrame(
                        {
                            "N": [ks[j] * N0],
                            "Method": [names[method_idx]],
                            "Prob_est - 1-eta": [data[i, j] - (1 - eta)],
                        }
                    ),
                ],
                ignore_index=True,
            )
    pd_boxplot = pd.concat([pd_boxplot, pd_boxplot_tmp])


plt.figure(figsize=(10, 10))
fsize = 16
figure_path_1_beta = os.path.join(
    save_dir,
    "figures",
    "1_beta_J_" + str(J) + "_tau_" + str(tau) + "_N_" + str(N0 * ks[-1]) + ".png",
)
for i in range(len(names)):
    pdSeries_tmp = pd_boxplot.loc[
        (pd_boxplot["Method"] == names[i]) & (pd_boxplot["N"] > 2)
    ]
    # pdSeries_tmp.loc[:, "Prob_est - 1-eta"] = pdSeries_tmp["Prob_est - 1-eta"].apply(lambda x: 1.0 if x >= 0 else 0.0)
    pdSeries_tmp.loc[:, "Prob_est - 1-eta"] = (
        pdSeries_tmp["Prob_est - 1-eta"] > 0
    ).values
    pdSeries_tmp = pdSeries_tmp.groupby("N").mean()
    x_plot = pdSeries_tmp.index
    y_plot = pdSeries_tmp["Prob_est - 1-eta"].values
    plt.plot(x_plot, y_plot, label=names[i], alpha=0.5)
    # plt.plot(np.array(ks)[1:] * N0, scenario_prob_esimate[i, 1:], label=names[i])
plt.xlabel("N", fontsize=fsize)
plt.ylabel(r"$1 - \hat{\beta}$", fontsize=fsize)
plt.grid()
plt.legend(prop={"size": fsize})
print("saved to ", figure_path_1_beta)
plt.savefig(figure_path_1_beta)
