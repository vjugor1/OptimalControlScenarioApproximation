import numpy as np
from matplotlib import pyplot as plt
import os

import seaborn as sns
import pandas as pd


from src.run import utils
from src.data_utils import plotting
from src.samplers.importance_sampler import *


from src.solvers import analytical_approx as AA


def map_names(
    results,
    new_names=[
        "SAO-ScenarioApproxWithO",
        "SA-ScenarioApprox",
        "SAIS-ScenarioApproxImportanceSampling",
    ],
):
    for r in results.keys():
        if r not in ["Sigma", "mu"]:
            for l in results[r]:
                keys = list(l.keys())
                for i in range(len(keys)):
                    l[new_names[i]] = l.pop(keys[i])
    return results


def main():
    # config
    grid_name = "grid14"
    save_dir = "saves"
    save_dir = os.path.join(save_dir, grid_name)
    eta = 0.01
    optimize_samples = True
    T = 3  # time snapshots

    (
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
        optimize_samples,
    ) = utils.initialize_multistep(grid_name=grid_name, eta=eta, T=T)

    # Solve scenario approximations
    # Store sigma and mu, next, the solutions for approximation will be pushed

    N0 = 3
    ks = list(range(1, 221))[::30]
    L = 200

    np.random.seed(228)

    # parallel and discard useless planes and samples

    all_samples_SAIMIN, all_samples_SCSA = utils.generate_samples_multistep(
        N0, ks, t_factors, L, eta, Delta_poly, Pi_tau_sample
    )
    # parallel and discard useless planes and samples
    results = utils.multiple_solve(
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
    )
    # save the results
    utils.save_results(save_dir, results, N0, ks, eta)

    results = utils.map_names(
        results,
        new_names=["SA-ScenarioApprox", "SAIS-ScenarioApproxImportanceSampling"],
    )

    # processing results for plotting average behaviour on L different computations
    try:
        names = list(results[N0][ks[0]].keys())
    except KeyError:
        names = list(results[str(N0)][ks[0]].keys())

    scenario_probs_several_starts = utils.estimate_probs(
        results,
        eta,
        Gamma,
        ramp_up_down,
        Beta,
        L,
        T,
        t_factors,
        ks,
        c,
        cost_correction_term,
        A,
        N0,
    )

    # shaping into pandas
    pd_boxplot = utils.estimates_to_pandas(
        scenario_probs_several_starts, ks, N0, names, eta, save_dir
    )

    # 1 - beta plot
    plotting.plot_1_minus_beta(pd_boxplot, save_dir, N0, ks, eta, names)

    # box plots
    plotting.plot_boxplots(pd_boxplot, save_dir, N0, ks, eta)


# Gamma, Beta = synth.regular_polyhedron(10, 6)
if __name__ == "__main__":
    main()
