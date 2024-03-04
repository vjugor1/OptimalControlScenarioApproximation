import numpy as np
import os
import hydra
from omegaconf import DictConfig, OmegaConf

from src.run import utils
from src.data_utils import plotting
from src.samplers.importance_sampler import *

@hydra.main(version_base=None, config_path="conf", config_name="config_grid14")
def main(cfg: DictConfig) -> None:
    # results = utils.map_names(
    #         results,
    #         new_names=["SA-ScenarioApprox", "SAIS-ScenarioApproxImportanceSampling"],
    #     )
    json_res, json_dro = utils.load_results(cfg)
    pass
    # processing results for plotting average behaviour on L different computations
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

    # # shaping into pandas
    # pd_boxplot = utils.estimates_to_pandas(
    #     scenario_probs_several_starts, ks, N0, names, eta, save_dir
    # )

    # # 1 - beta plot
    # plotting.plot_1_minus_beta(pd_boxplot, save_dir, N0, ks, eta, names)

    # # box plots
    # plotting.plot_boxplots(pd_boxplot, save_dir, N0, ks, eta)

if __name__ == "__main__":
    main()