import copy
import itertools

import numpy as np
import pandas as pd
from matplotlib import colors, cm
from matplotlib.axes import Axes
from matplotlib.cbook import flatten
from matplotlib.colors import ListedColormap

from utils import settings
from inhib_level.math import accumulation_index
from utils.plot_utils import (
    adjust_spines,
    copy_lines,
    recolor_shapeplot2d,
    plot_save,
    new_gridspec,
    create_zoom,
    letter_axes,
)

import logging

logger = logging.getLogger("IL location")


def figure_optimal_loc():
    """ How does dynamic Cl- affect IL

    """
    from matplotlib.gridspec import GridSpecFromSubplotSpec
    from main import run_inhib_level
    from shared import env_var
    import seaborn as sns

    logger.info("*" * 50 + "figure_dynamic_il_loc" + "*" * 50)

    # local vars common between nested methods
    tstop = 500
    tm = 10
    e_offset = -5
    sample_N = 4
    IL_measure = "IL"
    row_order = {"morph": -1, "measure": 0, "maxIL": 1}
    linestyles = {"0": ":", "AccIdx": "-", "i": "--"}
    plot_names = ["0", "i", "Focal", "Branch"]
    markers = ["o", "^", "d", "D"]
    measure_markers = dict(zip(plot_names, markers))

    fig, gs = new_gridspec(
        3,
        len(plot_names),
        figsize=(settings.PAGE_W_FULL, settings.PAGE_H_half),
        grid_kwargs=dict(height_ratios=[1, 1, 1]),
    )
    gs.update(top=0.95, left=0.1, right=0.93, bottom=0.07, hspace=0.5, wspace=0.5)

    def max_inhib_level(radials_diff=(2, 4, 6, 8, 16), kcc2="Y"):
        logger.info(
            "#" * 25
            + "\nCalculate and plot maximum IL for multiple numbers of branches\n"
            + "#" * 25
        )
        locs_diff = np.round(np.append(0.001, np.arange(0.01, 0.21, 0.01)), 5)
        loc_list = [0.0001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.9999]
        loc_list = np.round(loc_list, 6)
        full_loc_list = sorted(set(np.append(loc_list, locs_diff)))
        radials_diff_str = " ".join(str(r) for r in radials_diff)
        # line colors
        cs = [settings.cmap_dict["n_e"][n]["line"] for n in radials_diff]

        df_il_r = pd.DataFrame(
            index=full_loc_list,
            columns=pd.MultiIndex.from_product(
                [["AccIdx", "i", "0"], radials_diff], names=["Measure", "Branches"]
            ),
        )
        dist_list = ["diffused"]
        dist_list_str = " ".join(dist_list)
        dist_names = {"diffused": "Tree", "clustered_n": "Branch", "clustered": "Focal"}
        tree_shapes = {}
        egaba_dict = {"Tree": {}, "Branch": {}, "Focal": {}}

        # diffused (Tree)
        for num_branches, loc in itertools.product(radials_diff, full_loc_list):
            logger.info(f"DIFFUSED num_branches={num_branches} loc={loc}")
            main_args = (
                f'--radial {num_branches} --loc {" ".join([str(loc)]*num_branches)} '
                f"--e_offsets {e_offset} --synapse_dists {dist_list_str} --kcc2={kcc2} "
                # f'--plot_group_by=False '
                f"--plot_group_by=num_dendrites --plot_color_by=num_synapses "
                f"--tstop={tstop} --tm={tm} --diams {1} "
                f"--precise --sections radial_dends_1 --nseg=267 "
                f"--plot_shape"
            )
            plot_dict, sim_type, saved_args = run_inhib_level(main_args)
            il_dict = saved_args["il_dict"]
            ecl_dict = saved_args["ecl_dict"]
            index = il_dict["units"].index
            tree_shapes[(num_branches, loc)] = plot_dict
            # i = d
            loc_idx = abs(loc - index.levels[1]).argmin()
            for key, _df in il_dict.items():
                if key == "units":
                    continue
                n = int(key[key.index("n=") + 2 : key.index("/")])
                dist = key[key.index("\n") : key.index("x=")].strip()
                e = float(key[key.index("e=") + 2 : key.index("(")])
                cl = "(-)" not in key
                if cl:
                    if "diffused" in dist:
                        df_il_r.loc[loc, ("i", n)] = _df.loc[
                            ("radial_dends_1", tstop)
                        ].iloc[loc_idx]
                        df_il_r.loc[loc, ("0", n)] = _df.loc[("soma", tstop)].iloc[0]
                        df_il_r.loc[loc, ("AccIdx", n)] = (
                            df_il_r.loc[loc, ("0", n)] / df_il_r.loc[loc, ("i", n)]
                        )
                    elif "clustered" == dist:
                        df_il_r.loc[loc, ("clustered", n)] = _df.loc[
                            ("radial_dends_1", tstop)
                        ].iloc[loc_idx]
                    elif "clustered_n" == dist:
                        il = _df.loc[("radial_dends_1", tstop)]
                        idx = np.argmax(il)
                        max_il = il.iloc[idx]
                        df_il_r.loc[idx, ("clustered_n", n)] = max_il

                    ev = env_var()
                    pcl, ecl, phco3, ehco3, egaba, vinit = (
                        ev["pcl"],
                        ev["ecl"],
                        ev["phco3"],
                        ev["ehco3"],
                        ev["egaba"],
                        ev["v_init"],
                    )
                    egaba_dict[dist_names[dist]][(n, loc)] = (
                        pcl * ecl_dict[key]["radial_dends_1"].iloc[-1] + phco3 * ehco3
                    )

        # clustering
        df_cluster_focal = pd.DataFrame(
            index=loc_list, columns=pd.Index(radials_diff, name="Branches")
        )
        df_cluster_focal.name = "Focal"
        df_cluster_branch = pd.DataFrame(
            columns=pd.Index(radials_diff, name="Branches")
        )
        df_cluster_branch.name = "Branch"
        focal_shapes = {}
        branch_shapes = {}
        for num_branches in df_cluster_focal.columns:
            for loc in df_cluster_focal.index:
                logger.info(f"CLUSTERED num_branches={num_branches} loc={loc}")
                main_args = (
                    f'--radial {num_branches} --loc {" ".join([str(loc)]*num_branches)} '
                    f"--e_offsets {e_offset} --synapse_dists clustered --kcc2={kcc2} "
                    f"--plot_group_by=False "
                    f"--plot_group_by=num_dendrites --plot_color_by=num_synapses "
                    f"--tstop={tstop} --tm={tm} --diams {1} "
                    f"--precise --sections radial_dends_1 radial_dends_2 --nseg=267 "
                    f"--plot_shape"
                )
                plot_dict, sim_type, saved_args = run_inhib_level(main_args)
                focal_shapes[(num_branches, loc)] = plot_dict
                il_dict = saved_args["il_dict"]
                ecl_dict = saved_args["ecl_dict"]
                df_eg = [v for k, v in il_dict.items() if k != "units"][0]
                index = df_eg.index
                # i = d
                loc_idx = abs(loc - index.levels[1]).argmin()
                for key, _df in il_dict.items():
                    if key == "units":
                        continue
                    n = int(key[key.index("n=") + 2 : key.index("/")])
                    dist = key[key.index("\n") : key.index("x=")].strip()
                    e = float(key[key.index("e=") + 2 : key.index("(")])
                    cl = "(-)" not in key
                    assert dist == "clustered"
                    if cl:
                        df_cluster_focal.loc[loc, n] = _df.loc[
                            ("radial_dends_1", tstop)
                        ].iloc[loc_idx]
                        egaba_dict[dist_names[dist]][(n, loc)] = (
                            pcl * ecl_dict[key]["radial_dends_1"].iloc[-1]
                            + phco3 * ehco3
                        )

            logger.info(f"CLUSTERED_N num_branches={num_branches}")
            main_args = (
                f"--radial {num_branches} --loc {num_branches} "
                f"--e_offsets {e_offset} --synapse_dists clustered --kcc2={kcc2} "
                f"--plot_group_by=False "
                f"--plot_group_by=num_dendrites --plot_color_by=num_synapses "
                f"--tstop={tstop} --tm={tm} --diams {1} "
                f"--precise --sections radial_dends_1 radial_dends_2 --nseg=267 "
                f" --plot_shape"
            )
            plot_dict, sim_type, saved_args = run_inhib_level(main_args)
            branch_shapes[num_branches] = plot_dict
            il_dict = saved_args["il_dict"]
            ecl_dict = saved_args["ecl_dict"]
            for key, _df in il_dict.items():
                if key == "units":
                    continue
                n = int(key[key.index("n=") + 2 : key.index("/")])
                cl = "(-)" not in key
                if cl:
                    il = pd.to_numeric(_df.loc[("radial_dends_1", tstop)])
                    df_cluster_branch[n] = il
                    ev = env_var()
                    pcl, ecl, phco3, ehco3, egaba, vinit = (
                        ev["pcl"],
                        ev["ecl"],
                        ev["phco3"],
                        ev["ehco3"],
                        ev["egaba"],
                        ev["v_init"],
                    )
                    egaba_dict["Branch"] = (
                        pcl * ecl_dict[key]["radial_dends_1"].iloc[-1] + phco3 * ehco3
                    )

        _max_il = df_cluster_branch.max(axis=0)
        df_branch_long = pd.DataFrame(
            {
                "Branches": _max_il.index,
                "idx": df_cluster_branch.idxmax(axis=0),
                "max IL": _max_il,
            }
        )

        df_cluster_branch.plot()

        # spacing of clustering
        inh_distances = [0.001, 0.01, 0.1]
        locs_i = np.round(np.arange(0.01, 0.11, 0.01), 5)
        locs_i = loc_list[1:-1]
        df_space = pd.DataFrame(index=locs_i,
                                columns=pd.MultiIndex.from_product([inh_distances, radials_diff],
                                                                   names=['Spacing', 'Branches']))
        df_long_space = pd.DataFrame(columns=['Spacing', 'Branches', 'i', 'd', 'IL'])
        for num_branches, loc in itertools.product([4], locs_i):
            for inh_dist in inh_distances:
                logger.info(f"num_branches={num_branches} loc={loc} inh_dist={inh_dist}")
                loc_xs = np.round(np.linspace(np.max([0.001, loc - inh_dist]),
                                              np.min([loc + inh_dist, 0.999]),
                                              num_branches),
                                  5)
                main_args = f'--radial {num_branches} --loc {str(loc_xs)[1:-1]} '\
                            f' --e_offsets {e_offset} --synapse_dists clustered --kcc2=Y '\
                            f'--plot_group_by=num_dendrites --plot_color_by=num_synapses'\
                            f' --tstop={tstop} --tm={tm} --diams {1} '\
                            ' --precise --sections=radial_dends_1 --nseg=267 --plot_shape'
                plot_dict, sim_type, saved_args = run_inhib_level(main_args)
        
                il_dict = saved_args['il_dict']
                ecl_dict = saved_args['ecl_dict']
                index = il_dict['units'].index
                # i = d
                loc_idx = abs(loc - index.levels[1]).argmin()
                for key, _df in il_dict.items():
                    if key == 'units':
                        continue
                    n = int(key[key.index('n=') + 2:key.index('/')])
                    dist = key[key.index('\n'):key.index('x=')].strip()
                    e = float(key[key.index('e=') + 2:key.index('(')])
                    cl = key[key.index('(') + 1:key.index(')')]
        
                    il = _df.loc[('radial_dends_1', tstop)].astype(float)
                    df_space.loc[loc, (inh_dist, n)] = (il.idxmax(), il.max())
                    df_long_space.loc[df_long_space.shape[0]] = (inh_dist, n, loc, il.idxmax(), il.max())
        
        df_long_space['d_round'] = df_long_space.d.round(2)
        # df_long_space['i space N'] = \
        #     df_long_space.apply(lambda x: f"{x['i']} +- {x['Spacing']} ({x['Branches']})", axis=1)
        df_long_space['i space'] =\
            df_long_space.apply(lambda x: f"{x['i']} +- {x['Spacing']}", axis=1)
        cs = [settings.cmap_dict['n_e'][n]['line'] for n in radials_diff]
        import seaborn as sns
        g = sns.relplot(x="i", y="IL",
                        hue="Branches",
                        col="Spacing",
                        style='Spacing',
                        size="d",
                        sizes=(20, 100),
                        palette=cs,
                        ec='k',
                        legend="brief",
                        data=df_long_space)
        g.fig.tight_layout()
        g = sns.relplot(x="i", y="IL",
                        row="Branches",
                        col="Spacing",
                        hue="d_round",
                        style='Spacing',
                        size="d_round",
                        sizes=(20, 100),
                        palette="Set2",
                        ec='k',
                        legend="brief",
                        facet_kws=dict(margin_titles=True),
                        data=df_long_space)
        g.set(ylim=(-0.5, 2))
        g.fig.tight_layout()
        g = sns.relplot(x="i", y="d",
                        col="Branches",
                        hue="IL",
                        style='Spacing',
                        size="Spacing",
                        sizes=(100, 20),
                        hue_norm=(0, 2),
                        palette="viridis",
                        legend="brief",
                        data=df_long_space)
        g.fig.tight_layout()

        df_il_r.index = pd.Index(np.round(df_il_r.index, 2), name=settings.LOCATION_X_)

        max_il = {r: (-1, None, -1) for r in radials_diff}
        df_i = df_il_r["i"]
        df_i.name = "i"
        df_0 = df_il_r["0"]
        df_0.name = "0"

        _df_list = [
            _df
            for _df in [df_0, df_i, df_cluster_focal, df_cluster_branch]
            if _df.name in plot_names
        ]
        # individual axes
        for m, df_measure in enumerate(_df_list):
            measure = df_measure.name
            ax_radials_diff = fig.add_subplot(gs[row_order["measure"], m])
            ax_radials_diff.set_title(f"IL at {measure}", fontsize="medium")
            df_max = df_measure.copy()
            df_max.iloc[:, :] = np.nan  # set all to nan
            max_idxs = np.where(
                df_measure == df_measure.max(axis=0)
            )  # get (row_idxs,col_idxs)
            for row, col in zip(*max_idxs):
                df_max.iloc[row, col] = df_measure.iloc[
                    row, col
                ]  # assign only max values
                ax_radials_diff.annotate(
                    df_measure.columns[col],
                    xy=(df_measure.index[row], df_measure.iloc[row, col]),
                    xytext=(10, 0),
                    textcoords="offset points",
                    color=settings.cmap_dict["n_e"][df_measure.columns[col]]["line"],
                    fontsize="xx-small",
                )
            cmap = [
                c for n, c in sorted(settings.n_branches_cmap.items()) if n in df_max
            ]
            # plot all points
            ls = linestyles.get(measure, "--")
            df_measure.plot(
                ax=ax_radials_diff,
                linestyle=ls,
                lw=1,
                marker=markers[m],
                ms=3,
                mec="k",
                mew=0.1,
                cmap=ListedColormap(cmap),
                legend=False,
                clip_on=True,
            )
            # plot max point
            df_max.plot(
                ax=ax_radials_diff,
                linestyle="None",
                marker=markers[m],
                cmap=ListedColormap(cmap),
                ms=8,
                mec="k",
                legend=False,
                clip_on=False,
                zorder=99,
            )
            # calculate maximum IL for each radial (compared across all measures)
            for r in radials_diff:
                s = df_max[r].dropna()
                idx, val = s.index[0], s.values[0]
                max_idx, max_measure, max_val = max_il[r]
                if val > max_val:
                    max_il[r] = (idx, measure, val)
            if ax_radials_diff.get_ylim()[0] < -20:
                ax_radials_diff.set_ylim(bottom=np.nanmin(df_max.values) * 1.05)

            ax_radials_diff.set_ylim(top=np.nanmax(df_measure.values) * 1.1)

            ax_radials_diff.set_xlim(0, 0.2 if measure in ["0", "i"] else 1)
            if m == 0:
                ax_radials_diff.set_ylabel(f"max {settings.IL}")
            ax_radials_diff.set_xlabel(settings.LOCATION_X_, fontsize="xx-small")

        # merge plot
        from matplotlib import gridspec

        gs_merge = gridspec.GridSpecFromSubplotSpec(
            1,
            3,
            gs[row_order["maxIL"], :],
            width_ratios=[0.2, 1, 0.2],
            wspace=0,
            hspace=0,
        )
        gs_morph = gridspec.GridSpecFromSubplotSpec(
            2, len(radials_diff), gs[row_order["morph"], :], wspace=0, hspace=-0.95
        )
        ax_merge = fig.add_subplot(gs_merge[1])

        for m, (n, (idx, measure, val)) in enumerate(max_il.items()):
            df_measure = df_il_r[measure][n]
            ls = linestyles.get(measure, "--")
            df_measure.plot(
                ax=ax_merge,
                linestyle=ls,
                lw=1,
                marker=measure_markers[measure],
                ms=3,
                mec="k",
                mew=0.1,
                color=settings.cmap_dict["n_e"][n]["line"],
                legend=False,
                clip_on=True,
            )
            ax_merge.plot(
                idx,
                val,
                linestyle="None",
                marker=measure_markers[measure],
                color=settings.cmap_dict["n_e"][n]["line"],
                ms=8,
                mec="k",
            )
            if n == 4:
                # plot i, 0, and AccIdx measures for 4 branches in a zoom plot
                df_four = (
                    df_il_r.xs(n, axis="columns", level=1)
                    .reset_index()
                    .melt(
                        id_vars=settings.LOCATION_X_,
                        value_vars=df_il_r.columns.levels[0],
                        value_name=IL_measure,
                    )
                )
                df_four[IL_measure] = df_four[IL_measure].astype(float)

                ax_zoom = create_zoom(
                    ax_merge,
                    ("40%", "70%"),
                    lines=[],  # copy all lines
                    loc="upper center",
                    borderpad=0.0,
                    xlim=(0.0, 0.2),
                    ylim=(0.0, df_four[IL_measure].max() * 1.05),
                    connector_kwargs="None",
                )

                sns.lineplot(
                    x=settings.LOCATION_X_,
                    y=IL_measure,
                    # hue='Branches',
                    style="Measure",
                    style_order=["AccIdx", "0", "i"],
                    # markers=['.']+markers, mec='k',
                    color=settings.cmap_dict["n_e"][n]["line"],
                    ax=ax_zoom,
                    data=df_four,
                ).legend(loc=(1.01, 0), fontsize="xx-small")
                ax_zoom.plot(
                    idx,
                    val,
                    linestyle="None",
                    marker=measure_markers[measure],
                    color=settings.cmap_dict["n_e"][n]["line"],
                    ms=8,
                    mec="k",
                )
                ax_zoom.set(xlabel="", ylabel="")
            # setup morph axes
            ax_morph = fig.add_subplot(gs_morph[0, m])
            ax_morph_egaba = fig.add_subplot(gs_morph[1, m])
            ax_morph.set_facecolor("None")
            ax_morph_egaba.set_facecolor("None")
            ax_morph_egaba.set_zorder(-99)
            adjust_spines(ax_morph, [])
            adjust_spines(ax_morph_egaba, [])

            # get original morph axis and IL values
            if measure == "i" or measure == "0":
                _axes = tree_shapes[(n, idx)][f"{n}"][2]
                cvals = tree_shapes[(n, idx)][f"{n}"][1][0].get_lines()[4].get_ydata()
            elif measure == "Focal":
                _axes = focal_shapes[(n, idx)][f"{n}"][2]
                cvals = focal_shapes[(n, idx)][f"{n}"][1][0].get_lines()[4].get_ydata()
            elif measure == "Branch":
                _axes = branch_shapes[n][f"{n}"][2]
                cvals = branch_shapes[(n, idx)][f"{n}"][1][0].get_lines()[4].get_ydata()
            else:
                raise NameError("unknown measure")

            # morph ax and annotations
            ax_shape = [_ax for key, _ax in _axes.items() if key.endswith("AX")][0]
            annotations = [
                _ax for key, _ax in _axes.items() if key.endswith("ANNOTATIONS")
            ][0]

            # copy morph to this figure
            copy_lines(ax_shape, ax_morph, linewidth=1)
            copy_lines(ax_shape, ax_morph_egaba, linewidth=1)

            # add annotations
            ann = {"xy": (0, 50)}
            for annotation in annotations:
                ann = copy.deepcopy(annotation)
                _c = annotation["arrowprops"]["facecolor"]
                ann["xytext"] = (0, 4)
                ax_morph.annotate(**ann)

            # color according to what was recorded (IL, EGABA)
            recolor_shapeplot2d(
                ax_morph, cmap=settings.IL_config["IL"]["cmap"], cvals=cvals[1:-1]
            )
            recolor_shapeplot2d(
                ax_morph_egaba,
                cmap=settings.IL_config["EGABA"]["cmap"],
                cvals=egaba_dict["Tree" if measure in ["i", "0"] else measure][
                    n, idx
                ].iloc[1:-1],
            )

            # point of measuring IL (d location)
            xy = (0, 50) if measure == "0" else ann["xy"]
            ax_morph.annotate(
                "",
                xy=xy,
                xytext=(0, 5),
                textcoords="offset points",
                arrowprops=dict(arrowstyle="->"),
            )

        # clean up merged axis
        ax_merge.set_ylim(0, max([val for _, _, val in max_il.values()]) * 1.1)

        ax_merge.set_ylabel(f"max {settings.IL}")
        ax_merge.set_xlabel(settings.LOCATION_X_)
        ax_merge.set_xlim(0, 1)
        ax_merge.set_xticks(np.arange(0, 1.1, 0.1))
        ax_merge.set_xticks(df_il_r.index, minor=True)

    max_inhib_level()

    if settings.SAVE_FIGURES:
        plot_save("output/figure_optimal_loc.png", figs=[fig], close=False)
        plot_save("output/figure_optimal_loc.pdf")
    else:
        import shared

        shared.show_n(1)
