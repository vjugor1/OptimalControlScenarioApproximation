from matplotlib import pyplot as plt
import os
import numpy as np
import seaborn as sns
import matplotlib as mpl
import src.data_utils.snapshots as ss


def plot_polygon(Gamma, Beta, label, color_idx=1, xmin=-40, xmax=40, ax=None):
    J = Gamma.shape[0]
    cmap = plt.cm.terrain  # define the colormap
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # force the first color entry to be grey
    cmaplist[0] = (0.5, 0.5, 0.5, 1.0)

    # create the new map
    T = 10
    cmap = mpl.colors.LinearSegmentedColormap.from_list("Custom cmap", cmaplist, cmap.N)
    tags = np.arange(T)  # np.random.randint(0, 20, 20)
    tags[10:12] = 0
    bounds = np.linspace(0, T, T + 1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    # plt.figure(figsize=(5, 5))
    x_range = np.linspace(xmin, xmax, (xmax - xmin) * 100)
    tag = tags[color_idx]
    for i_plane in range(len(Gamma)):
        left_Gamma = Gamma[i_plane - 1]  # / Gamma.shape[1] / i
        left_Beta = Beta[i_plane - 1]  # / Gamma.shape[1] / i
        right_Gamma = Gamma[(i_plane + 1) % J]  # / Gamma.shape[1] / i
        right_Beta = Beta[(i_plane + 1) % J]  # / Gamma.shape[1] / i
        curr_Gamma = Gamma[i_plane]  # / Gamma.shape[1] / i
        c2 = (curr_Gamma)[-1]
        c1 = (curr_Gamma)[0]
        b = Beta[i_plane]  # / Gamma.shape[1] / i
        if (np.abs(c2) < 1e-3) and (np.abs(c1) != 0.0):
            ys_plane = x_range
            xs_plane = np.ones(len(ys_plane)) * b / c1
        elif (np.abs(c1) < 1e-3) and (np.abs(c2) != 0.0):
            xs_plane = x_range
            ys_plane = np.ones(len(xs_plane)) * b / c2
        else:
            xs_plane = x_range
            ys_plane = 1 / c2 * (b - c1 * xs_plane)

        idxs = np.where(
            (left_Gamma @ np.vstack((xs_plane, ys_plane)) <= left_Beta)
            & (right_Gamma @ np.vstack((xs_plane, ys_plane)) <= right_Beta)
        )
        # plt.scatter(xs_plane[idxs], ys_plane[idxs], c=tag , cmap=cmap, norm=norm, s=1, label=i)
        if i_plane == (len(Gamma) - 1):
            if ax is None:
                plt.plot(
                    xs_plane[idxs],
                    ys_plane[idxs],
                    color=cmap(norm(tag)),
                    # s=0.7,
                    label=label,
                )
            else:
                ax.plot(
                    xs_plane[idxs],
                    ys_plane[idxs],
                    color=cmap(norm(tag)),
                    # s=0.7,
                    label=label,
                )
        else:
            if ax is None:
                plt.plot(
                    xs_plane[idxs], ys_plane[idxs], color=cmap(norm(tag))
                )  # , s=1)
            else:
                ax.plot(xs_plane[idxs], ys_plane[idxs], color=cmap(norm(tag)))  # , s=1)
        if ax is None:
            plt.scatter(0, 0, s=3, color="black")
        else:
            ax.scatter(0, 0, s=3, color="black")
    if ax is None:
        ax = plt.gca()
    else:
        pass
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_aspect("equal", adjustable="box")


def plot_polygon_multistep_AGC(
    Gamma, Beta, alphas=np.array([0.5, 0.5]), T=40, up_to_T=20, xmin=-40, xmax=40
):
    J = Gamma.shape[0]
    cmap = plt.cm.terrain  # define the colormap
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # force the first color entry to be grey
    cmaplist[0] = (0.5, 0.5, 0.5, 1.0)

    # create the new map
    cmap = mpl.colors.LinearSegmentedColormap.from_list("Custom cmap", cmaplist, cmap.N)
    tags = np.arange(T)  # np.random.randint(0, 20, 20)
    tags[10:12] = 0
    bounds = np.linspace(0, T, T + 1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    Gamma_s, Beta_s, Aalpha = ss.get_snapshots_planes(
        Gamma, Beta, np.ones((T, Gamma.shape[1])), alphas, T
    )

    xi = np.random.normal(size=T * 2)
    # xi = np.random.normal() * np.ones(100)
    # xi = 1 * np.ones(100)
    # alphas = np.array([0.5, 0.5])
    plt.figure(figsize=(5, 5))
    x_range = np.linspace(xmin, xmax, (xmax - xmin) * 100)
    for i in list(range(1, up_to_T))[::4]:
        tag = tags[i]
        for i_plane in range(len(Gamma)):
            left_Gamma = Gamma_s[
                i_plane - 1 + i * Gamma.shape[0]
            ]  # / Gamma.shape[1] / i
            left_Beta = (
                Beta_s[i_plane - 1 + i * Gamma.shape[0]]  # / Gamma.shape[1] / i
                - (Aalpha * xi[i - 1])[i_plane - 1]
            )
            right_Gamma = Gamma_s[
                (i_plane + 1) % J + i * Gamma.shape[0]
            ]  # / Gamma.shape[1] / i
            right_Beta = (
                Beta_s[(i_plane + 1) % J + i * Gamma.shape[0]]  # / Gamma.shape[1] / i
                - (Aalpha * xi[(i - 1) % len(xi)])[(i_plane + 1) % J]
            )
            curr_Gamma = Gamma_s[i_plane + i * Gamma.shape[0]]  # / Gamma.shape[1] / i
            c2 = (curr_Gamma)[-1]
            c1 = (curr_Gamma)[0]
            b = (
                Beta_s[i_plane + i * Gamma.shape[0]]  # / Gamma.shape[1] / i
                - (Aalpha * xi[i - 1])[i_plane]
            )
            if (np.abs(c2) < 1e-3) and (np.abs(c1) != 0.0):
                ys_plane = x_range
                xs_plane = np.ones(len(ys_plane)) * b / c1
            elif (np.abs(c1) < 1e-3) and (np.abs(c2) != 0.0):
                xs_plane = x_range
                ys_plane = np.ones(len(xs_plane)) * b / c2
            else:
                xs_plane = x_range
                ys_plane = 1 / c2 * (b - c1 * xs_plane)

            idxs = np.where(
                (left_Gamma @ np.vstack((xs_plane, ys_plane)) <= left_Beta)
                & (right_Gamma @ np.vstack((xs_plane, ys_plane)) <= right_Beta)
            )
            # plt.scatter(xs_plane[idxs], ys_plane[idxs], c=tag , cmap=cmap, norm=norm, s=1, label=i)
            if i_plane == (len(Gamma) - 1):
                plt.scatter(
                    xs_plane[idxs],
                    ys_plane[idxs],
                    color=cmap(norm(tag)),
                    s=1,
                    label=str(i),
                )
            else:
                plt.scatter(xs_plane[idxs], ys_plane[idxs], color=cmap(norm(tag)), s=1)
            plt.scatter(0, 0, s=3, color="black")
        ax = plt.gca()
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        ax.set_aspect("equal", adjustable="box")


def plot_1_delta(
    pd_boxplot, save_dir, eta, fsize=16, fig_xsize=10, fig_ysize=10, save=True
):
    N = int(max(pd_boxplot.N))
    # fsize = 16
    # fig_xsize = 10
    # fig_ysize = 10
    xlims = [
        0,
        N,
    ]  # further will be refined accoridng to the last method reached \hat{\delta} = 1.0
    N_reached_1 = []
    plt.figure(figsize=(fig_xsize, fig_ysize))
    names = pd_boxplot["Method"].unique()
    # fsize = 16
    if eta != 0.05:
        figure_path_1_beta = os.path.join(
            save_dir,
            "figures",
            "1_beta_N_" + str(N) + "_eta_" + str(np.round(eta, 2)) + ".png",
        )
    else:
        figure_path_1_beta = os.path.join(
            save_dir, "figures", "1_beta_N_" + str(N) + ".png",
        )
    for i in range(len(names)):
        pdSeries_tmp = pd_boxplot.loc[
            (pd_boxplot["Method"] == names[i]) & (pd_boxplot["N"] > 2)
        ]
        pdSeries_tmp.loc[:, r"$(\hat{\mathbb{P}}_N)_l$"] = (
            pdSeries_tmp[r"$(\hat{\mathbb{P}}_N)_l$"] > 1 - eta
        ).values
        pdSeries_tmp = pdSeries_tmp.groupby("N").mean()
        x_plot = pdSeries_tmp.index
        y_plot = pdSeries_tmp[r"$(\hat{\mathbb{P}}_N)_l$"].values
        # find N that yielded 1.0 delta
        idxs_reached_1 = np.where(y_plot >= 1 - 1e-9)[0]
        if len(idxs_reached_1) == 0:
            N_reached_1.append(N)
        else:
            N_reached_1.append(x_plot.values[idxs_reached_1[0]])
        plt.plot(x_plot, y_plot, label=names[i], alpha=0.5, linewidth=2.5)
        # plt.plot(np.array(ks)[1:] * N0, scenario_prob_esimate[i, 1:], label=names[i])
    plt.xlabel("N", fontsize=fsize)
    plt.ylabel(r"$1 - \hat{\delta}$", fontsize=fsize)
    plt.grid()
    xlims[-1] = np.max(N_reached_1) + 20
    plt.xlim(xlims)
    plt.legend(prop={"size": fsize}, loc="lower right")
    if save:
        try:
            plt.savefig(figure_path_1_beta)
        except FileNotFoundError:
            os.makedirs(os.path.join(save_dir, "figures"))
            plt.savefig(figure_path_1_beta)
        print("Saved to ", figure_path_1_beta)
    return xlims


def plot_boxplot(
    pd_boxplot, save_dir, eta, fsize=16, fig_xsize=10, fig_ysize=10, save=True
):
    N = int(max(pd_boxplot.N))
    # fsize = 16
    # fig_xsize = 10
    # fig_ysize = 5
    if eta != 0.05:
        figure_path_box = os.path.join(
            save_dir,
            "figures",
            "boxplot_J_N_" + str(N) + "_eta_" + str(np.round(eta, 2)) + ".png",
        )
    else:
        figure_path_box = os.path.join(
            save_dir, "figures", "boxplot_J_N_" + str(N) + ".png",
        )
    plt.figure(figsize=(fig_xsize, fig_ysize))
    ax = sns.boxplot(
        x="N",
        y=r"$(\hat{\mathbb{P}}_N)_l$",
        hue="Method",
        data=pd_boxplot[pd_boxplot["N"] > 6],
        palette="Set3",
    )
    ax.axhline(
        1 - eta,
        0,
        1,
        label=r"$1 - \eta$",
        color="black",
        linewidth=2,
        alpha=0.7,
        linestyle="dotted",
    )
    plt.ylim((1 - 2 * eta, 1.0))
    plt.legend(prop={"size": fsize}, loc="lower right")
    if save:
        try:
            plt.savefig(figure_path_box)
        except FileNotFoundError:
            os.makedirs(os.path.join(save_dir, "figures"))
            plt.savefig(figure_path_box)
        print("Saved to ", figure_path_box)


def plot_grids(pds, save_dir, eta, include_O=True, truncate_names=True):
    if save_dir is None:
        save_dir = ""
    for grid_name in pds.keys():
        pd_boxplot = pds[grid_name]
        if truncate_names:
            method_names = pd_boxplot.Method.unique()
            pd_boxplot.Method = pd_boxplot.Method.map(
                {mn: mn.split("-")[0] for mn in method_names}
            )
        fig_xsize = 10
        fig_ysize = 5
        fsize = 20
        if include_O:
            no_SAO = pd_boxplot
            no_SAO["N"] = no_SAO["N"].astype(int)
        else:
            if truncate_names:
                no_SAO = pd_boxplot.drop(
                    pd_boxplot[pd_boxplot["Method"] == "SAO"].index
                )
            else:
                no_SAO = pd_boxplot.drop(
                    pd_boxplot[pd_boxplot["Method"] == "SAO-ScenarioApproxWithO"].index
                )
            no_SAO["N"] = no_SAO["N"].astype(int)
        save_dir_grid30 = os.path.join(save_dir, grid_name)
        xlims = plot_1_delta(
            no_SAO,
            save_dir_grid30,
            eta,
            save=True if save_dir is not None else False,
            fig_xsize=fig_xsize,
            fig_ysize=fig_ysize,
            fsize=fsize,
        )

        pd_boxplot = pds[grid_name]
        fig_xsize = 10
        fig_ysize = 5
        fsize = 15
        no_SAO_lim = no_SAO.drop(no_SAO[no_SAO["N"] >= xlims[-1]].index)
        save_dir_grid30 = os.path.join(save_dir, grid_name)
        plot_boxplot(
            no_SAO_lim,
            save_dir_grid30,
            eta,
            save=True if save_dir is not None else False,
            fig_xsize=fig_xsize,
            fig_ysize=fig_ysize,
            fsize=fsize,
        )

def plot_1_minus_beta(pd_boxplot, save_dir, N0, ks, eta, names):
    plt.figure(figsize=(10, 10))
    fsize = 16
    figure_path_1_beta = os.path.join(
        save_dir,
        "figures",
        "1_beta_N_" + str(N0 * ks[-1]) + "_eta_" + str(np.round(eta, 2)) + ".png",
    )
    for i in range(len(names)):
        pdSeries_tmp = pd_boxplot.loc[
            (pd_boxplot["Method"] == names[i]) & (pd_boxplot["N"] > 2)
        ]
        # pdSeries_tmp.loc[:, "Prob_est - 1-eta"] = pdSeries_tmp["Prob_est - 1-eta"].apply(lambda x: 1.0 if x >= 0 else 0.0)
        pdSeries_tmp.loc[:, r"$(\hat{\mathbb{P}}_N)_l$"] = (
            pdSeries_tmp[r"$(\hat{\mathbb{P}}_N)_l$"] > 1 - eta
        ).values
        pdSeries_tmp = pdSeries_tmp.groupby("N").mean()
        x_plot = pdSeries_tmp.index
        y_plot = pdSeries_tmp[r"$(\hat{\mathbb{P}}_N)_l$"].values
        plt.plot(x_plot, y_plot, label=names[i], alpha=0.8)
        # plt.plot(np.array(ks)[1:] * N0, scenario_prob_esimate[i, 1:], label=names[i])
    plt.xlabel("N", fontsize=fsize)
    plt.ylabel(r"$1 - \hat{\delta}$", fontsize=fsize)
    plt.grid()
    plt.legend(prop={"size": fsize})
    try:
        plt.savefig(figure_path_1_beta)
        pass
    except FileNotFoundError:
        os.makedirs(os.path.join(save_dir, "figures"))
        plt.savefig(figure_path_1_beta)

def plot_boxplots(pd_boxplot, save_dir, N0, ks, eta):
    figure_path_box = os.path.join(
        save_dir,
        "figures",
        "boxplot_J_N_" + str(N0 * ks[-1]) + "_eta_" + str(np.round(eta, 2)) + ".png",
    )
    plt.figure(figsize=(20, 5))
    ax = sns.boxplot(
        x="N",
        y=r"$(\hat{\mathbb{P}}_N)_l$",
        hue="Method",
        data=pd_boxplot[pd_boxplot["N"] > 1],
        palette="Set3",
    )
    ax.axhline(
        1 - eta,
        0,
        1,
        label=r"$1 - \eta$",
        color="black",
        linewidth=2,
        alpha=0.7,
        linestyle="dotted",
    )
    # plt.ylim((1 - 2 * eta, 1.))
    plt.legend()
    # plt.hlines(y = [1-eta], xmin = pd_boxplot["N"].min(), xmax = pd_boxplot["N"].max(), label=r'$1 - \eta$', color='black', linestyle='dotted')
    # plt.grid()
    try:
        plt.savefig(figure_path_box)
        pass
    except FileNotFoundError:
        os.makedirs(os.path.join(save_dir, "figures"))
        plt.savefig(figure_path_box)
