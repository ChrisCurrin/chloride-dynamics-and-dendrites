import numpy as np
import pandas as pd
from matplotlib import patheffects as pe, colors, cm
from matplotlib.axes import Axes
from matplotlib.cbook import flatten
from matplotlib.gridspec import GridSpecFromSubplotSpec

from utils import settings
from inhib_level.math import accumulation_index
from utils.plot_utils import (
    adjust_spines,
    letter_axes,
    copy_lines,
    recolor_shapeplot2d,
    plot_save,
    new_gridspec,
)

import logging

logger = logging.getLogger()


def figure_ai_cl():
    """ AI insensitive to dynamic chloride
        A   |   B   |   C

    A: AccIdx vs time for different branches

    B: Erev vs time for different branches

    C: ILdiff vs time for different branches




    """
    from main import run_inhib_level
    from shared import env_var

    logger.info("*" * 50 + "figure_ai_cl" + "*" * 50)

    X = 0.2
    sample_radial = 4
    sample_e = -5.0
    loc_list = [X]
    loc_list_str = " ".join(f"[{_loc}]" for _loc in loc_list)
    radial_list = [2, sample_radial, 6, 8]
    radial_list_str = " ".join(str(r) for r in radial_list)
    e_offset_list = [sample_e]
    e_offsets = " ".join(str(e) for e in e_offset_list)

    tstop = 1000
    tm = 5
    delta = settings.DELTA
    t_points = [20, 100, 260, 580, tstop]
    morphs = ["EGABA", settings.ILdiff, "relative"]
    morph_rows, morph_cols = morphs, t_points
    cb_pos = ["all", "once"][1]
    # fig, (ax_accidx, ax_erev, ax_ILdiff) = plt.subplots(nrows=3, ncols=1)

    # Figure settings
    gs_size = 100
    wpad = hpad = int(0.08 * gs_size)
    hm_space = 3
    hm_shrink = 2

    fig, gs = new_gridspec(
        2,
        1,
        figsize=(settings.PAGE_W_half, settings.PAGE_H_half),
        grid_kwargs=dict(height_ratios=[3, 5]),
    )
    gs.update(top=0.98, left=0.24, right=0.82, hspace=0.15, bottom=0.1)
    # top row height
    _height_r1 = 0.25  # percentage
    row1_h = int(gs_size * _height_r1)
    row2_h = row3_h = int((gs_size - row1_h) / 2)
    if cb_pos == "once":
        gs0 = GridSpecFromSubplotSpec(
            len(morph_rows),
            len(morph_cols) + 1,
            subplot_spec=gs[0],
            wspace=0.05,
            hspace=0.2,
            width_ratios=[5] * len(morph_cols) + [1],
        )
        ax_cb = []
        for i in range(len(morph_rows)):
            ax_cb.append(fig.add_subplot(gs0[i, -1]))
    else:
        gs0 = GridSpecFromSubplotSpec(
            len(morph_rows),
            len(morph_cols) + 1,
            subplot_spec=gs[0],
            wspace=0,
            hspace=0.1,
        )
    gs1 = GridSpecFromSubplotSpec(3, 2, subplot_spec=gs[1], height_ratios=[2, 2, 3])

    ax_branches = np.empty(shape=(len(morph_rows), len(morph_cols)), dtype=Axes)
    for i in range(len(morph_rows)):
        for j in range(len(morph_cols)):
            ax_branches[i, j] = fig.add_subplot(gs0[i, j])
    adjust_spines(ax_branches, [], 0)

    ax_eXl = fig.add_subplot(gs1[0, :])
    ax_il0 = fig.add_subplot(gs1[1, :], sharex=ax_eXl)
    ax_e0 = ax_eXl
    ax_ilXl = ax_il0.twinx()
    ax_ilXl.set_zorder(-1)
    ax_accidx = fig.add_subplot(gs1[-1, :], sharex=ax_eXl)

    # letter_axes([ax_accidx, ax_erev, ax_ILdiff], xy=(-0.1, 1.))
    letter_axes(
        ax_branches[0, 0],
        ax_eXl,
        ax_il0,
        ax_accidx,
        xy=[(0, 0.95), (0, 0.60), (0, 0.43), (0, 0.28)],
        xycoords="figure fraction",
    )

    logger.info(
        """
    ####################################################################################################################
    # AccIdx vs time
    ####################################################################################################################
    """
    )

    plot_dict, sim_type, saved_args = run_inhib_level(
        f"--radial {radial_list_str} --loc {loc_list_str} --e_offsets {e_offsets}"
        " --synapse_dists=diffused_matched "
        f" --kcc2=C --tstop={tstop} --tm={tm}"
        f" --plot_group_by=e_offsets --with-t-trace {X} --plot_shape"
        f" --sections=radial_dends_1 "
        " --precise "
    )

    il_dict = saved_args["il_dict"]
    ecl_dict = saved_args["ecl_dict"]
    t_index = np.linspace(tm, tstop, int(tstop / tm))
    columns = pd.MultiIndex.from_product(
        [radial_list, e_offset_list, ["-", delta]],
        names=["Branches", settings.EGABA, "Chloride"],
    )
    loc_index = il_dict["units"].loc["radial_dends_1"].index
    df_accum = pd.DataFrame(index=t_index, columns=columns)
    df_morph = pd.DataFrame(
        index=t_index,
        columns=pd.MultiIndex.from_product(
            [set(morphs + [settings.IL]), loc_index], names=["var", "X"]
        ),
    )
    df_il = pd.DataFrame(index=t_index, columns=columns)
    df_il_junction = pd.DataFrame(index=t_index, columns=columns)
    df_erev_loc = pd.DataFrame(columns=columns)
    df_erev_junction = pd.DataFrame(columns=columns)
    chosen_il = None

    # inhibitory level
    for key, _df in il_dict.items():
        if key == "units":
            continue
        n = int(key[key.index("n=") + 2 : key.index("/")])
        e = float(key[key.index("e=") + 2 : key.index("(")])
        cl = key[key.index("(") + 1 : key.index(")")]
        accum = accumulation_index(_df, loc_list)
        df_accum.loc[:, (n, e, cl)] = accum[X]
        idx = abs(X - _df.loc["radial_dends_1"].index).argmin()
        df_il.loc[:, (n, e, cl)] = _df.loc["radial_dends_1"].iloc[idx]
        df_il_junction.loc[:, (n, e, cl)] = _df.loc[("soma", 0.5)]
        if n == sample_radial and e == sample_e:
            if cl == delta:
                df_morph[settings.IL] = _df.loc["radial_dends_1"].T
            else:
                df_morph[settings.ILdiff] = _df.loc["radial_dends_1"].T

    df_morph[settings.ILdiff] = df_morph[settings.ILdiff] - df_morph[settings.IL]

    # EGABA
    ev = env_var()
    pcl, ecl, phco3, ehco3, egaba, vinit = (
        ev["pcl"],
        ev["ecl"],
        ev["phco3"],
        ev["ehco3"],
        ev["egaba"],
        ev["v_init"],
    )
    for key, _df in ecl_dict.items():
        if key == "units":
            continue
        n = int(key[key.index("n=") + 2 : key.index("/")])
        e = float(key[key.index("e=") + 2 : key.index("(")])
        cl = key[key.index("(") + 1 : key.index(")")]
        _df_erev = pcl * _df + phco3 * ehco3
        idx = abs(X - _df_erev["radial_dends_1"].columns).argmin()
        df_erev_loc[(n, e, cl)] = _df_erev["radial_dends_1"].iloc[:, idx]
        df_erev_junction[(n, e, cl)] = _df_erev["soma", 0.5]
        if n == sample_radial and e == sample_e and cl == delta:
            df_morph["EGABA"] = (
                _df_erev["radial_dends_1"]
                .loc[0 : tstop + tm : tm * int(1 / 0.025)]
                .iloc[1:]
                .values
            )

    # morphology
    if f"{sample_e:.1f}" in plot_dict:
        # create morphology plots
        for ax_key, _ax in plot_dict[f"{sample_e:.1f}"][2].items():
            if ax_key.startswith("SHAPE") and ax_key.endswith("AX"):
                n = int(ax_key[ax_key.index("n=") + 2 : ax_key.index("/")])
                # dist = ax_key[ax_key.index('\n'):ax_key.index('x=')].strip()
                x = ax_key[ax_key.index("x=[") + 3 : ax_key.index("]")].split(" ")
                e = float(ax_key[ax_key.index("e=") + 2 : ax_key.index("(")])
                cl = ax_key[ax_key.index("(") + 1 : ax_key.index(")")]
                if n == sample_radial and sample_e == e and cl == delta:
                    underscore_idx = ax_key.rfind("_")
                    annotation_key = ax_key[:underscore_idx] + "_ANNOTATIONS"
                    annotations = plot_dict[f"{sample_e:.1f}"][2][annotation_key]
                    for ax_loc in flatten(ax_branches):
                        copy_lines(
                            _ax,
                            ax_loc,
                            linewidth=2,
                            path_effects=[
                                # pe.Stroke(linewidth=2.2, foreground='k'),
                                pe.SimpleLineShadow(
                                    offset=(0.1, 0.1), shadow_color="k", alpha=0.1
                                ),
                                pe.Normal(),
                            ],
                        )
                        # add annotations
                        for annotation in annotations:
                            ax_loc.annotate(**annotation)
        # change color of morphology plots
        clims = {}
        for j, key in enumerate(morphs):
            if key == "relative":
                continue
            clims[key] = [df_morph[key].min().min(), df_morph[key].max().max()]
            if key == "EGABA":
                clims[key] = [egaba, np.max([vinit, clims[key][1]])]
            elif key == settings.ILdiff:
                clims[key][0] = 0
        for j, key in enumerate(morphs):
            val = settings.IL_config[key]
            if key in morph_rows:
                ax_branches[j, 0].annotate(
                    val["label"],
                    xy=(-0.1, 0.5),
                    xycoords="axes fraction",
                    ha="right",
                    va="center",
                    fontsize="small",
                )
            else:
                ax_branches[0, j].set_title(val["label"], fontsize="small")

            for t, t_point in enumerate(t_points):
                _ax = ax_branches[t, j] if t_point in morph_rows else ax_branches[j, t]
                cmap = val["cmap"]
                if key == "relative":
                    cvals = df_morph[settings.IL].loc[t_point].iloc[1:-1]
                    clim = [
                        cvals.min(),
                        cvals.max(),
                    ]  # get the min/max values for this time point
                else:
                    clim = clims[key]
                    cvals = df_morph[key].loc[t_point].iloc[1:-1]

                recolor_shapeplot2d(_ax, cmap=cmap, cvals=cvals, clim=clim)
                # heatmap
                norm = colors.Normalize(*clim)
                if cb_pos == "once" and t_points == morph_rows:
                    cb_kwargs = dict(orientation="horizontal", shrink=0.8)
                else:
                    cb_kwargs = dict(orientation="vertical", fraction=0.45)
                if cb_pos == "once" and t + 1 == len(t_points):
                    cb = fig.colorbar(
                        cm.ScalarMappable(norm=norm, cmap=cmap),
                        cax=ax_cb[j],
                        **cb_kwargs,
                    )
                elif cb_pos == "all":
                    cb = fig.colorbar(
                        cm.ScalarMappable(norm=norm, cmap=cmap), ax=_ax, **cb_kwargs
                    )
                else:
                    continue

                cb.set_label(
                    val["cb_label"],
                    fontsize="x-small",
                    rotation=0,
                    ha="left",
                    va="center_baseline",
                )
                cb.ax.tick_params(labelsize="x-small")
                if key == "EGABA" and cb_pos == "all":
                    cb.set_ticks(
                        sorted([egaba, vinit, round(clim[1]), round(cvals.mean())])
                    )
                # elif key == 'EGABA':
                #     cb.set_ticks(np.round(np.linspace(egaba, vinit, round(vinit - egaba) + 1)))
                # elif key == settings.ILdiff or key == settings.IL:
                #     cb.set_ticks(np.round(clim))
                elif key == "relative":
                    cb.set_ticks(clim)
                    cb.set_ticklabels(["min", "max"])

        for t, t_point in enumerate(t_points):
            if t_point in morph_rows:
                ax_branches[t, 0].annotate(
                    f"{t_point:.0f}",
                    xy=(0.0, 0.5),
                    xycoords="axes fraction",
                    ha="right",
                    va="center",
                    fontsize="small",
                )
            else:
                ax_branches[-1, t].annotate(
                    f"{t_point:.0f}",
                    xy=(0.5, 0.0),
                    xycoords="axes fraction",
                    ha="center",
                    va="top",
                    fontsize="small",
                )

    # select from dataframes
    # EGABA and IL/ILdiff from just sample_radial
    select_erev = df_erev_loc[sample_radial].xs(delta, axis=1, level=1)
    select_erev_junc = df_erev_junction[sample_radial].xs(delta, axis=1, level=1)
    select_il = df_il[sample_radial].xs(delta, axis=1, level=1)
    select_il_junc = df_il_junction[sample_radial].xs(delta, axis=1, level=1)
    df_il_diff = (
        df_il[sample_radial, sample_e, "-"] - df_il[sample_radial, sample_e, delta]
    )
    df_il_diff_junction = (
        df_il_junction[sample_radial, sample_e, "-"]
        - df_il_junction[sample_radial, sample_e, delta]
    )
    # accumulation for all radial_list
    select_acc = df_accum.xs(sample_e, axis=1, level=1).xs(delta, axis=1, level=1)

    # plot EGABA and IL/IL_dff for sample_radial
    styles = dict(zorder=99, alpha=1.0)

    cmap = settings.COLOR.truncate_colormap(
        settings.cmap_dict["num_synapses"][sample_radial], 0.3, 0.8
    )
    select_erev.plot(ax=ax_eXl, cmap=cmap, legend=False, ls="-", **styles)
    select_erev_junc.plot(ax=ax_e0, legend=False, ls=":", c="k", **styles)
    df_il_diff.plot(ax=ax_ilXl, cmap=cmap, legend=False, ls="-", **styles)
    df_il_diff_junction.plot(ax=ax_il0, legend=False, ls=":", c="k", **styles)
    ax_ilXl.spines["right"].set_color(cmap(1))
    line_colors = []
    for n_dend in radial_list:
        cmap = settings.COLOR.truncate_colormap(
            settings.cmap_dict["num_synapses"][n_dend], 0.3, 0.8
        )
        line_colors.append(cmap(1))

    # plot accumulation for all radial_list
    select_acc.plot(ax=ax_accidx, color=line_colors)

    ax_eXl.legend(
        ["synapse", "junction"], loc=(0.9, 0.1), fontsize="x-small", frameon=False
    )
    ax_accidx.legend(
        loc="upper left",
        bbox_to_anchor=(0.93, 1.0),
        fontsize="x-small",
        title="Branches",
        title_fontsize="small",
        frameon=False,
    )

    # egaba hlines
    cmap = settings.COLOR.truncate_colormap(
        settings.IL_config[settings.EGABA]["cmap"], 0.3, 0.8
    )
    style = dict(linestyle="-", lw=1, alpha=0.8, zorder=-99)
    egabas = np.linspace(egaba, vinit, round(vinit - egaba) + 1)
    egabas_x = []
    egabas_y = []
    line_colors = []
    df_egaba = df_erev_loc[sample_radial].xs(delta, axis=1, level=1)
    for e in egabas:
        _df_ediff = abs(df_egaba - e)
        idxmin = _df_ediff.idxmin()
        if isinstance(idxmin, pd.Series):
            idxmin = idxmin.mean()
        if _df_ediff.loc[idxmin].values < 0.01:
            egabas_y.append(idxmin)
        egabas_x.append(idxmin)
        line_colors.append(cmap(int((e - egaba) * 255 / (vinit - egaba))))
    ax_eXl.hlines(egabas, xmin=0, xmax=egabas_x, colors=line_colors, **style)
    ax_eXl.vlines(egabas_y, ymin=egaba, ymax=egabas, colors=line_colors, **style)

    # labels and legends
    # adjust_spines(ax_e0, ['right'], 0)
    adjust_spines(ax_ilXl, ["right"], 0)
    ax_eXl.tick_params(labelbottom=False)
    # ax_e0.tick_params(labelbottom=False)
    from matplotlib.ticker import AutoMinorLocator

    ax_eXl.yaxis.set_minor_locator(AutoMinorLocator())
    ax_eXl.set_ylim(egaba)
    # ax_e0.set_ylim(egaba)
    ax_ilXl.set_ylim(0)
    ax_il0.set_ylim(0)
    ax_eXl.set_ylabel(f"{settings.EGABA}\n({settings.mV})")
    # ax_e0.set_ylabel(f"{settings.EGABA}\n({settings.mV})")
    ax_ilXl.set_ylabel(f"${settings.ILdiff.replace('$', '')}_{{\mathit{{synapse}}}}$")
    ax_il0.set_ylabel(f"${settings.ILdiff.replace('$', '')}_{{\mathit{{junction}}}}$")
    ax_ilXl.set(xlabel="Time (ms)")
    ax_il0.set(xlabel="Time (ms)")
    ax_accidx.set(xlabel="Time (ms)", ylabel=f"{settings.ACC_IDX}")
    ax_accidx.set_xlim(0)
    fig.align_ylabels([ax_eXl, ax_ilXl, ax_accidx])

    # ax_eXl.annotate("$\mathit{synapse}$", xy=(0, 1.05), xycoords='axes fraction',
    #                 fontsize='x-small', ha='right', va='bottom')
    # ax_eXl.annotate("$\mathit{junction}$", xy=(1, 1.05), xycoords='axes fraction',
    #                 fontsize='x-small', ha='left', va='bottom')

    if settings.SAVE_FIGURES:
        plot_save("output/figure_ai_cl.png", figs=[fig], close=False)
        plot_save("output/figure_ai_cl.pdf")
    else:
        import shared

        shared.show_n()
