import numpy as np
import os
import hydra
from omegaconf import DictConfig, OmegaConf

from src.run import utils
from src.samplers.importance_sampler import *


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

@hydra.main(version_base=None, config_path="conf", config_name="config_grid6")
def main(cfg: DictConfig) -> None:
    # config
    grid_name = cfg.grid
    # save_dir = os.path.join("saves", "dd-dro")
    # save_dir = os.path.join(save_dir, grid_name)
    save_dir = os.path.join(cfg.paths.dro_results, cfg.grid)
    eta = cfg.solution.eta #0.1 for grid14, grid30, 0.15 for grid56
    # optimize_samples = True
    T = cfg.solution.T #3 for grid14, grid30, 2 for grid57  # time snapshots

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
    ) = utils.initialize_multistep(cfg, grid_name=grid_name, eta=eta, T=T)

    # Solve scenario approximations
    # Store sigma and mu, next, the solutions for approximation will be pushed

    N0 = cfg.solution.N0
    ks = list(range(1, cfg.solution.k))[::cfg.solution.N_step]
    L = cfg.solution.L
    if cfg.solution.dro is not None:
        M = cfg.solution.dro.M
    else:
        M = None
    if cfg.solution.dro is not None:
        theta = cfg.solution.dro.theta
    else:
        theta = None
    np.random.seed(228)

    # parallel and discard useless planes and samples

    all_samples_SAIMIN, all_samples_SCSA = utils.generate_samples_multistep(
        N0, ks, t_factors, L, eta, Delta_poly, Pi_tau_sample
    )
    # all_samples_SAIMIN *= 0
    # all_samples_SCSA *= 0
    results = utils.multiple_solve_dro(
    cfg,
    x0,
    N0,
    ks,
    L,
    all_samples_SCSA,
    all_samples_SAIMIN,
    Sigma,
    mu,
    Gamma,
    Beta,
    ramp_up_down,
    T,
    alpha_0,
    delta_alpha,
    c,
    M = M, 
    theta = theta, 
    eta=eta,
    cost_correction_term=cost_correction_term
)


    # save the results
    utils.save_results(save_dir, results, N0, ks, eta)


    # results = utils.map_names(
    #     results,
    #     new_names=["SA-ScenarioApprox", "SAIS-ScenarioApproxImportanceSampling"],
    # )

    # # # processing results for plotting average behaviour on L different computations
    # try:
    #     names = list(results[N0][ks[0]].keys())
    # except KeyError:
    #     names = list(results[str(N0)][ks[0]].keys())

    # scenario_probs_several_starts = utils.estimate_probs(
    #     results,
    #     eta,
    #     Gamma,
    #     ramp_up_down,
    #     Beta,
    #     L,
    #     T,
    #     t_factors,
    #     ks,
    #     c,
    #     cost_correction_term,
    #     A,
    #     N0,
    # )

    # # # shaping into pandas
    # pd_boxplot = utils.estimates_to_pandas(
    #     scenario_probs_several_starts, ks, N0, names, eta, save_dir
    # )

    # # # 1 - beta plot
    # plotting.plot_1_minus_beta(pd_boxplot, save_dir, N0, ks, eta, names)

    # # # box plots
    # plotting.plot_boxplots(pd_boxplot, save_dir, N0, ks, eta)


# Gamma, Beta = synth.regular_polyhedron(10, 6)
if __name__ == "__main__":
    main()
