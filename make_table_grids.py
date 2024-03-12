import numpy as np
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import json
import collections


from src.samplers.importance_sampler import *
from src.data_utils import grid_data
from src.data_utils import plotting
from src.samplers.utils import check_feasibility_out_of_sample
from src.samplers import preprocessing as pre
from src.data_utils import synthetic as synth
from src.solvers import scenario_approx as SA
from src.solvers import utils as SU
from src.solvers import analytical_approx as AA
from src.run import utils
from src.data_utils import plotting
from src.samplers.importance_sampler import *

def unpack_cost_foos(results, c, k, cost_correction_term):
    try:
        some_key = list(results.keys())[-1]
        names = list(results[some_key][k].keys())
    except KeyError:
        some_key = list(results.keys())[0]
        names = list(results[some_key][k].keys())
    fns = []
    xs = []
    for r in results.keys():
        if r not in ["Sigma", "mu"]:
            for v in results[r][k].values():
                try:
                    xs.append(v[0])
                    fns.append(np.dot(v[0], c) + cost_correction_term)
                except ValueError:
                    fns.append(np.nan)
    fns = np.array(fns).reshape(-1, len(names))
    #xs = np.array(xs).reshape(-1, len(names), A.shape[1])
    return fns, names

# save_dir = os.path.join('..', 'saves')
def get_summary_dict(pds, costs_dict, eta):
    summary_dict = {}
    names = ['SA', 'AR-SA']
    for grid_name, pd_boxplot in pds.items():
        summary_dict[grid_name] = []
        for i in range(len(names)):
            pdSeries_tmp = (pd_boxplot.loc[(pd_boxplot["Method"] == names[i]) & (pd_boxplot["N"] > 2)])
            #pdSeries_tmp.loc[:, "Prob_est - 1-eta"] = pdSeries_tmp["Prob_est - 1-eta"].apply(lambda x: 1.0 if x >= 0 else 0.0)
            c_est = pdSeries_tmp.columns[-1]
            pdSeries_tmp.loc[:, c_est] = (pdSeries_tmp[c_est] > 1-eta).values
            pdSeries_tmp = pdSeries_tmp.drop(columns=['Method'])
            pdSeries_tmp = pdSeries_tmp.groupby("N").mean()
            x_plot = pdSeries_tmp.index
            y_plot = pdSeries_tmp[c_est].values
            # Extracting number of samples required by this method to reach 0.99 delta level (reliability)
            reliable_idx = np.where(y_plot >= 0.95)[0][0]
            # Here my homie
            n_samples_reliable = x_plot[reliable_idx]

            summary_dict[grid_name].append([n_samples_reliable,
                                             costs_dict[grid_name]['mean'][reliable_idx, i],
                                             costs_dict[grid_name]['std'][reliable_idx, i],
                                             costs_dict[grid_name]['dc-opf']
                                             ])
    summary_ord_dict = collections.OrderedDict(sorted(summary_dict.items(), key=lambda x: int(x[0].split('grid')[-1])))
    return summary_ord_dict

def read_stats(conf, pd_dict, costs_dict, eta_in):
    N_SA = conf.estimation.N_SA
    eta = conf.estimation.eta
    if eta == eta_in:
        path_to_res_dir = os.path.join(conf.paths.dro_results, conf.grid)
        path_to_csv = os.path.join(path_to_res_dir, f"multistarts_N_{N_SA}_eta_{eta}.csv")
        multistart_pd = pd.read_csv(path_to_csv).drop(columns=['Unnamed: 0'])
        pd_dict[conf.grid] = multistart_pd
        json_file = "N_" + str(N_SA) + "_eta_" + str(eta) + ".json"
        _, _, _, c, cost_correction_term, cost_dc_opf = grid_data.get_linear_constraints(conf.grid, check_pp_vs_new_form=False)
        c = np.hstack((c, np.zeros(len(c))))
        # # load if necessary
        
        with open(os.path.join(path_to_res_dir, json_file), 'r') as fp:
            results = json.load(fp)
        # results = map_names(results, new_names=['SA-ScenarioApprox', 'SAIS-ScenarioApproxImportanceSampling'])
        L = len(results[list(results.keys())[-1]])
        fns_L = []
        for l in range(L):
            fns, names = unpack_cost_foos(results=results, c=c, k=0, cost_correction_term=cost_correction_term)
            fns_L.append(fns)
        fns_mean = np.stack(fns_L).mean(axis=(0))
        fns_std = np.stack(fns_L).std(axis=(0))
        costs_dict[conf.grid] = {}
        costs_dict[conf.grid]['mean'] = fns_mean
        costs_dict[conf.grid]['std'] = fns_std
        costs_dict[conf.grid]['dc-opf'] = cost_dc_opf


def main() -> None:
    path_config_folder = '/app/conf'
    list_configs = ['config_grid14.yaml', 'config_grid14_005.yaml', 'config_grid30.yaml']
    # list_configs = ['config_grid14.yaml', 'config_grid14_005.yaml', 'config_grid30_005.yaml', 'config_grid30.yaml']
    costs_dict_005 = {}
    pds_dict_005 = {}
    costs_dict_001 = {}
    pds_dict_001 = {}

    for conf_pth in list_configs:
        conf = OmegaConf.load(os.path.join(path_config_folder, conf_pth))
        
        read_stats(conf, pds_dict_001, costs_dict_001, 0.01)
        
        read_stats(conf, pds_dict_005, costs_dict_005, 0.05)
        
    summary_dict = get_summary_dict(pds_dict_005, costs_dict_005, 0.05)
    summary_dict_001 = get_summary_dict(pds_dict_001, costs_dict_001, 0.01)

    table = pd.DataFrame(columns=['Case', r'$\eta$', 'SA No', 'SA Cost', 'AR-SA No', 'AR-SA Cost', 'DC-OPF Cost'])
    table
    for i, key in enumerate(summary_dict.keys()):
        curr_grid = summary_dict[key]
        SA_res = curr_grid[0]
        ISSA_res = curr_grid[1]
        get_cost_str = lambda x, y: "{:.1e}".format(x) + r'$\pm$' + "{:.1e}".format(y)
        row = [key, 0.05, int(SA_res[0]), get_cost_str(SA_res[1], SA_res[2]),
        int(ISSA_res[0]), get_cost_str(ISSA_res[1], ISSA_res[2]), "{:.1e}".format(SA_res[-1])]
        table.loc[i] = row
    for j, key in enumerate(summary_dict_001.keys()):
        curr_grid = summary_dict_001[key]
        SA_res = curr_grid[0]
        ISSA_res = curr_grid[1]
        get_cost_str = lambda x, y: "{:.1e}".format(x) + r'$\pm$' + "{:.1e}".format(y)
        row = [key, 0.01, int(SA_res[0]), get_cost_str(SA_res[1], SA_res[2]),
        int(ISSA_res[0]), get_cost_str(ISSA_res[1], ISSA_res[2]), "{:.1e}".format(SA_res[-1])]
        table.loc[i + j + 1] = row

    table_text = print(table.to_latex(escape=False, index=False))
    print(table_text)



if __name__ == "__main__":
    main()