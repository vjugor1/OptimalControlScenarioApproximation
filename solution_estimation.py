import numpy as np
import os
import hydra
from omegaconf import DictConfig, OmegaConf

from src.run import utils
from src.data_utils import plotting
from src.samplers.importance_sampler import *

def estimate_to_pd(cfg, eta, T, N0, results, ks, names, save_dir):
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
    ) = utils.initialize_multistep(cfg, grid_name=cfg.grid, eta=eta, T=T)
    scenario_probs_several_starts, exec_times = utils.estimate_probs(
        results,
        eta,
        Gamma,
        ramp_up_down,
        Beta,
        cfg.solution.L,
        cfg.solution.T,
        t_factors,
        ks,
        c,
        cost_correction_term,
        A,
        N0,
        names
    )

    # shaping into pandas
    pd_boxplot = utils.estimates_to_pandas(
        scenario_probs_several_starts, ks, cfg.solution.N0, names, eta, save_dir
    )
    if cfg.estimation.plot_exec_time:
        pd_boxplot_times = utils.exec_time_to_pandas(
            exec_times, ks, cfg.solution.N0, names, eta, save_dir
        )
    else:
        pd_boxplot_times = None
    return pd_boxplot, pd_boxplot_times

def make_plots(cfg, pd_boxplot, pd_time, save_dir, N0, ks, eta, names):
    # 1 - beta plot
    plotting.plot_1_minus_beta(pd_boxplot, save_dir, N0, ks, eta, names)
    if cfg.estimation.plot_exec_time:
        plotting.plot_1_minus_beta(pd_time, save_dir, N0, ks, eta, names, col_name='Exec. Time')
    # box plots
    plotting.plot_boxplots(pd_boxplot, save_dir, N0, ks, eta)
    if cfg.estimation.plot_exec_time:
        plotting.plot_boxplots(pd_time, save_dir, N0, ks, eta, col_name="Exec. Time")

@hydra.main(version_base=None, config_path="conf", config_name="config_grid6_reduction")
def main(cfg: DictConfig) -> None:
    # results = utils.map_names(
    #         results,
    #         new_names=["SA-ScenarioApprox", "SAIS-ScenarioApproxImportanceSampling"],
    #     )
    jsons = utils.load_results(cfg)
    # processing results for plotting average behaviour on L different computations
    json_res, json_dro = jsons
    # print(json_res[str(cfg.solution.N0)][0].keys())
    # raise ValueError

    try:
        names_res = list(json_res[cfg.solution.N0][0].keys())
    except KeyError:
        names_res = list(json_res[str(cfg.solution.N0)][0].keys())
    except TypeError:
        names_res = None
    try:
        names_dro = list(json_dro[cfg.solution.N0][0].keys())
    except KeyError:
        names_dro = list(json_dro[str(cfg.solution.N0)][0].keys())

    ks = list(range(1, cfg.solution.k)[::cfg.solution.N_step])
    eta = cfg.estimation.eta
    # save_dir = os.path.join(cfg.paths.saves_dir, cfg.grid)
    N0 = cfg.solution.N0
    T = cfg.solution.T

    # save_dir = os.path.join(cfg.paths.saves_dir, cfg.grid)
    # results = json_res
    # names = names_res
    # pd_boxplot = estimate_to_pd(cfg, eta, T, N0, results, ks, names, save_dir)
    # make_plots(pd_boxplot, save_dir, N0, ks, eta, names)
    
    save_dir = os.path.join(cfg.paths.saves_dir, 'dd-dro', cfg.grid, )
    results = json_dro
    names = names_dro
    pd_boxplot, pd_boxplot_exec_time = estimate_to_pd(cfg, eta, T, N0, results, ks, names, save_dir)
    make_plots(cfg, pd_boxplot, pd_boxplot_exec_time, save_dir, N0, ks, eta, names)


if __name__ == "__main__":
    main()