import pandas as pd
import seaborn as sns
from matplotlib import cm, colors, patheffects as path_effects
from matplotlib.axes import Axes
from matplotlib.cbook import flatten
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

from utils import settings
from inhib_level.math import accumulation_index
from utils.plot_utils import (
    letter_axes,
    copy_lines,
    adjust_spines,
    plot_save,
    new_gridspec,
    create_zoom,
)
from utils.settings import (
    GS_C,
    GS_C_third,
    WPAD,
    GS_R_half,
    GS_R_third,
    GS_R_4th,
    HPAD,
    GS_C_half,
)

import logging

logger = logging.getLogger()


def figure_input_structure_eff():
    """ How input structure (effective number of branches) affect IL accum

    Ai   ii   iii   iv
        B   |     C

    A: Shape plots for different effective number of branches
        1/4, 2/4, 4/4, 8/4

    B: IL vs Distance for different effective number of branches
        1/4, 2/4, 4/4, 8/4

    C: Accumulation Index (AI) vs Effective number of branches
        Use for branches of 4 and 8 so that effective numbers are in %
        as 12.5 or 25, 50, 100, 200

    """
    from main import run_inhib_level

    logger.info("*" * 50 + "figure_input_structure_eff" + "*" * 50)
    fig_settings = {"heatmap_location": "right"}
    ncols = GS_C + GS_C_half
    fig, gs = new_gridspec(
        nrows=4,
        ncols=5,
        figsize=(settings.PAGE_W_FULL, settings.PAGE_H_FULL),
        grid_kwargs=dict(
            width_ratios=[0.1, 0.3, 1.5, 1, 1], height_ratios=[1.2, 0.6, 0.6, 0.8]
        ),
    )
    gs.update(left=0.03, right=0.96, top=0.95, hspace=0.5, wspace=0.6)
    ax_eff_branches = []
    ax_cbs = []
    plot_synapses = [1, 2, 4, 8]
    il_x_lines_to_plot = [0.25, 0.5, 1, 2]
    gs_morph = GridSpecFromSubplotSpec(
        4, 2, gs[0:2, 0:2], width_ratios=[10, 0.5], hspace=0.05, wspace=0
    )
    for i in range(len(plot_synapses)):
        ax_eff_branches.append(fig.add_subplot(gs_morph[i, 0]))
    ax_cb = fig.add_subplot(gs_morph[0, -1])
    ax_cbs.append(ax_cb)
    ax_il_dist = fig.add_subplot(gs[0:2, 2])
    ax_il_n = fig.add_subplot(gs[0, 3])
    ax_il_e = fig.add_subplot(gs[0, 4])
    ax_il_dist_eff: Axes = fig.add_subplot(gs[1, 3:])
    ax_il_n_eff: Axes = ax_il_dist_eff
    divider = make_axes_locatable(ax_il_dist_eff)
    ax_il_e_eff: Axes = divider.append_axes("top", size=0.45, pad=0.05)
    adjust_spines(ax_il_e_eff, ["left"])

    ax_accidx_eff: Axes = fig.add_subplot(gs[2:, 1:-1])  # bottom
    ax_accidx_eff4: Axes = fig.add_subplot(gs[3:, -1])  # bottom

    # adjust_spines(ax_il_n, ["left", "top"], 0)
    # adjust_spines(ax_il_dist, ["left", "top"], 0)
    # adjust_spines(ax_il_e, ["left", "top"], 0)

    letter_axes(ax_eff_branches, subscript="A", xy=(0.0, 1.1), ha="right")
    letter_axes(ax_il_dist, "B", xy=(-0.2, 1.0), ha="right", va="bottom")
    letter_axes(ax_il_n, "C", xy=(-0.2, 1.0), ha="right")
    letter_axes(ax_il_e, "D", xy=(-0.1, 1.0), ha="right")
    letter_axes(ax_accidx_eff, start="E", xy=(-0.15, 1.0), ha="right")
    for ax_branch in ax_eff_branches:
        ax_branch.set_aspect(1.2)
    # for ax_branch in ax_eff_branches + ax_cbs:
    #     pos = ax_branch.get_position()
    #     ax_branch.set_position([pos.x0 - 0.01, pos.y0, pos.width, pos.height])

    logger.info(
        """
    ####################################################################################################################
    # IL vs Distance and Accumulation Index (AI) vs Effective number of branches
    ####################################################################################################################
    """
    )
    tm = settings.TSTOP
    branch_bases = [4, 8]
    il_plots = {4: {0: ax_il_dist, -2: ax_il_e}, 8: {0: ax_il_n}}
    il_eff_plots = {
        4: {0: ax_il_dist_eff, -2: ax_il_e_eff},
        8: {0: ax_il_n_eff, -2: ax_il_e_eff},
    }
    shape_cmap = settings.IL_config[settings.IL]["cmap"]
    loc = [0.2]
    e_off = [0, -2, -5]
    sub_e_off = [0, -2]
    e_off_s = [str(e) for e in e_off]
    df_accum_others = {}
    for base in branch_bases:
        locs = [
            f"{loc}*{int(l)}"
            for l in [
                base / 4,
                base / 2,
                base * 3 / 4,
                base,
                base * 5 / 4,
                base * 6 / 4,
                base * 7 / 4,
                base * 2,
                base * 9 / 4,
                base * 10 / 4,
                base * 11 / 4,
                base * 12 / 4,
                base * 13 / 4,
                base * 14 / 4,
                base * 15 / 4,
                base * 4,
            ]
        ]
        # shape_cmap = settings.cmap_dict["num_synapses"][base].replace('_r', '')
        plot_dict, sim_type, saved_args = run_inhib_level(
            f'--radial {base} --e_offsets {" ".join(e_off_s)} '
            f'--loc {" ".join(locs)} '
            f"--synapse_dist=diffused "
            f"--plot_group_by=e_offsets "
            f"--plot_shape {shape_cmap}"
            " --precise"
        )
        xlim = (0, 1)
        cmap = settings.cmap_dict["n_e"][base]["cmap"]

        il_dict = saved_args["il_dict"]
        index = il_dict["units"].loc["radial_dends_1"].index
        loc_idx = abs(loc[0] - index).argmin()
        df_accum = pd.DataFrame()
        df_il_0 = pd.DataFrame()
        df_il_i = pd.DataFrame()
        df_il_i4 = pd.DataFrame()

        df_accum_4 = pd.DataFrame()
        for key, df in il_dict.items():
            if key == "units":
                continue
            n = key[key.index("n=") + 2 : key.index("/")]
            int_n = int(n)
            e = key[key.index("e=") + 2 : key.index("(")]
            e_num = float(e)
            e = f"{e:>5s}"
            accum = accumulation_index(df, loc, dend="radial_dends_1")
            df_accum.loc[100 * int_n / base, e] = accum[loc[0]][tm]
            df_il_i.loc[100 * int_n / base, e_num] = df.loc["radial_dends_1"].iloc[
                loc_idx, 0
            ]
            df_il_0.loc[100 * int_n / base, e_num] = df.loc["radial_dends_1"].iloc[0, 0]
            if int_n >= base:
                accum4 = accumulation_index(df, loc, dend=f"radial_dends_{base}")
                df_accum_4.loc[100 * int_n / base, e] = accum4[loc[0]][tm]
                df_il_i4.loc[100 * int_n / base, float(e.strip())] = df.loc[
                    f"radial_dends_{base}"
                ].iloc[loc_idx, 0]

        df_accum.plot(
            ax=ax_accidx_eff,
            marker="v",
            linestyle="--",
            colors=[settings.cmap_dict["n_e"][base][e] for e in e_off],
            label=base,
            markeredgecolor="k",
            clip_on=False,
        )
        if base == 8:
            eff_4_lot = [f"{e:>5.1f}" for e in sub_e_off]
            df_accum[eff_4_lot].plot(
                ax=ax_accidx_eff4,
                marker="v",
                linestyle="--",
                ms=4,
                lw=2,
                markeredgewidth=0.1,
                colors=[settings.cmap_dict["n_e"][base][e] for e in sub_e_off],
                label=base,
                markeredgecolor="k",
                clip_on=True,
                legend=False,
            )
            df_accum_others[base] = df_accum_4[eff_4_lot]

        if base in il_eff_plots:
            for key, _ax_eff in il_eff_plots[base].items():
                # df_il_i[key].plot(ax=_ax_eff, marker='v', ms=6, linestyle='--', cmap=cmap, label=base,
                #                   markeredgecolor='k', clip_on=False, legend=False, zorder=99)
                # if df_il_i4.shape[0]:
                #     df_il_i4[key].plot(ax=_ax_eff, marker='v', ms=6, linestyle='None', cmap=cmap, label=None,
                #                        markeredgecolor='k', clip_on=False, alpha=0.5, legend=False)
                df_il_0[key].plot(
                    ax=_ax_eff,
                    marker="o",
                    ms=6,
                    linestyle="--",
                    colors=[settings.cmap_dict["n_e"][base][key]],
                    label=base,
                    markeredgecolor="k",
                    clip_on=False,
                    legend=False,
                    zorder=99,
                )
                _ax_eff.axvline(
                    x=il_x_lines_to_plot[-1] * 100,
                    lw=0.5,
                    ls=":",
                    alpha=0.5,
                    color="k",
                    zorder=-2,
                )
                # _ax_eff.set_ylim(0)
                _ax_eff.set_xlim(0)
                _ax_eff.xaxis.set_major_locator(MaxNLocator(4))

        if base in il_plots:
            n = b = base
            color = settings.cmap_dict["n_e"][base]["line"]
            for key, _ax in il_plots[base].items():
                marker_color = settings.cmap_dict["n_e"][base][key]
                all_lines = plot_dict[str(key)][1][0].get_lines()
                new_ax_lines = []
                temp_lines = []
                for idx, line in enumerate(all_lines):
                    temp_lines.append(line)
                    label = line.get_label()
                    if "n=" in label:
                        # every (base+2) lines is a line with a detailed label
                        n = int(label[label.index("n=") + 2 : label.index("/")])
                        b = int(label[label.index("/") + 1 : label.index("\n")])
                        if n / b in il_x_lines_to_plot:
                            # if the synapses/branches is to be plotted, add to list
                            new_ax_lines += temp_lines
                        elif n / b == 1.5 and _ax != ax_il_dist:
                            zoom_ax = create_zoom(
                                _ax,
                                1.7,
                                temp_lines,
                                loc="upper right",
                                xlim=(0, loc_idx + 1),
                                xunit=1,  # use indices to cut off x-range
                                xticks=2,
                                yticks=3,
                                inset_kwargs=dict(borderpad=0.5),
                                connector_kwargs="None",
                                box_kwargs="None",
                                lw=2 * n / b,
                                clip_on=False,
                                zorder=99,
                                color=color,
                                mfc=marker_color,
                                path_effects=[
                                    path_effects.SimpleLineShadow(offset=(0, -0.5)),
                                    path_effects.Normal(),
                                ],
                            )
                            zoom_ax.set_xlim(0, 0.2)
                            zoom_ax.annotate(
                                f"{100*n/b:.0f} %",
                                xy=(0.5, 0.95),
                                xycoords="axes fraction",
                                fontsize="x-small",
                                ha="center",
                                va="top",
                                zorder=101,
                            )
                            zoom_ax.tick_params(axis="both", labelsize="xx-small")
                        # clear temp list
                        temp_lines = []
                copy_lines(
                    new_ax_lines,
                    _ax,
                    rel_lw=1,
                    color=color,
                    mfc=marker_color,
                    path_effects=[
                        path_effects.SimpleLineShadow(offset=(0, -0.5)),
                        path_effects.Normal(),
                    ],
                )
                _ax.annotate(
                    f"{base} branches",
                    xy=(0.5, 1.05),
                    xycoords="axes fraction",
                    va="bottom",
                    ha="center",
                    c=color,
                )
                _ax.annotate(
                    f"{settings.NABLAEGABA} = {key} mV",
                    xy=(0.5, 1.05),
                    xycoords="axes fraction",
                    va="top",
                    ha="center",
                    c=marker_color,
                )

                for line in _ax.get_lines()[::-1]:
                    label = line.get_label()
                    line.set_color(color)
                    line.set_markerfacecolor(marker_color)
                    if "n=" in label:
                        n = int(label[label.index("n=") + 2 : label.index("/")])
                        b = int(label[label.index("/") + 1 : label.index("\n")])
                    line.set_linewidth(2 * n / b)

                xlabel = plot_dict[str(key)][1][0].get_xlabel()
                # ylabel = plot_dict[str(key)][1][0].get_ylabel()
                ylabel = settings.SHUNT_LEVEL if key == 0 else settings.INHIBITORY_LEVEL
                _ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlim)

                if _ax == ax_il_dist:
                    _ax.set_ylabel(settings.SHUNT_LEVEL)
                    # add legend for the first axis
                    handles, labels = [], []
                    for handle, label in zip(*ax_il_dist.get_legend_handles_labels()):
                        if "n=" in label:
                            n = int(label[label.index("n=") + 2 : label.index("/")])
                            b = int(label[label.index("/") + 1 : label.index("\n")])
                            handles.append(handle)
                            labels.append(f"{100*n/b:>5.0f}%")

                    legend = ax_il_dist.legend(
                        handles,
                        labels,
                        title="No. synapses\n$\overline{No. branches}$",
                        title_fontsize="small",
                        fontsize="small",
                        frameon=False,
                        loc=2,
                        bbox_to_anchor=(0.45, 0.95),
                        ncol=1,
                        columnspacing=0.0,
                        labelspacing=0.2,
                        handletextpad=3,
                    )
                    for t in legend.get_texts():
                        t.set_ha("right")
                    legend._legend_box.align = "left"  # align title
                    # plot neuron shapes
                    for ax_key, ax in plot_dict["0"][2].items():
                        if ax_key.startswith("SHAPE") and ax_key.endswith("AX"):
                            n = int(ax_key[ax_key.find("n=") + 2 : ax_key.find("/")])
                            if n in plot_synapses:
                                ax_idx = plot_synapses.index(n)
                                copy_lines(ax, ax_eff_branches[ax_idx])
                                underscore_idx = ax_key.rfind("_")
                                annotation_key = (
                                    ax_key[:underscore_idx] + "_ANNOTATIONS"
                                )
                                annotations = plot_dict["0"][2][annotation_key]
                                for annotation in annotations:
                                    annotation["arrowprops"]["facecolor"] = marker_color
                                    ax_eff_branches[ax_idx].annotate(**annotation)

                                if fig_settings["heatmap_location"] == "right":
                                    ax_eff_branches[ax_idx].annotate(
                                        f"{100*n/base:.0f}%",
                                        xy=(0.75, 0.89),
                                        xycoords="axes fraction",
                                        ha="right",
                                        va="center",
                                        fontsize="small",
                                        rotation=28,
                                    )
                                else:
                                    ax_cb = ax_cbs[ax_idx]
                                    cb = fig.colorbar(
                                        cm.ScalarMappable(
                                            norm=colors.Normalize(0, 1),
                                            cmap=settings.IL_config[settings.IL][
                                                "cmap"
                                            ],
                                        ),
                                        cax=ax_cb,
                                        orientation="horizontal",
                                    )
                                    cb.set_label(
                                        settings.INHIBITORY_LEVEL, fontsize="x-small"
                                    )  # below/next to colorbar (same as ticks)
                                    cb.set_ticks([0, 1])
                                    cb.set_ticklabels(["min", "max"])
                                    cb.ax.tick_params(labelsize="x-small")
                                    cb.ax.set_title(
                                        f"{100*n/base:.0f}%", fontsize="small"
                                    )  # above colorbar

                    adjust_spines(ax_eff_branches, [], position=0)
                    if fig_settings["heatmap_location"] == "right":
                        ax_cb = ax_cbs[0]
                        cb = fig.colorbar(
                            cm.ScalarMappable(
                                norm=colors.Normalize(0, 1), cmap=shape_cmap
                            ),
                            cax=ax_cb,
                            orientation="vertical",
                        )
                        # cb.set_label(settings.IL, fontsize='x-small')  # below/next to colorbar (same as ticks)
                        cb.set_ticks([0, 1])
                        cb.set_ticklabels(["min", "max"])
                        cb.ax.tick_params(labelsize="x-small")
                        # cb.ax.set_title(f"{100*n/base:.0f}%", fontsize='small')  # above colorbar

    ax_accidx_eff.axhline(y=1.0, ls=":", c="k", alpha=0.6, zorder=-10)
    ax_accidx_eff4.axhline(y=1.0, ls=":", c="k", alpha=0.6, zorder=-10)
    legend = ax_accidx_eff.legend(
        title="Branches\n" + f"{' '*12}".join(str(b) for b in branch_bases) + "\n",
        loc=2,
        bbox_to_anchor=(1.02, 1.02),
        ncol=2,
        frameon=False,
        borderpad=0.0,
        borderaxespad=0,
        fontsize="small",
        columnspacing=1.0,
        handletextpad=0.0,
    )
    # legend = ax_accidx_eff.legend(list(flatten([lines[i::4] for i in range(4)])),
    #                              list(flatten([labels[i::4] for i in range(4)])),
    #                              title=f"{settings.NABLAEGABA} (mV)",
    #                              loc="lower left", bbox_to_anchor=(.0, 1.), ncol=4, frameon=False, borderpad=.0,
    #                              borderaxespad=0, fontsize='small', columnspacing=1., labelspacing=0.2,
    #                              handletextpad=0.,
    #                              )
    # add additional points after legend creation
    for base, df_accum_4 in df_accum_others.items():
        df_accum_4.plot(
            ax=ax_accidx_eff4,
            marker="v",
            linestyle=":",
            ms=4,
            lw=2,
            markeredgewidth=0.1,
            colors=[settings.cmap_dict["n_e"][base][e] for e in sub_e_off],
            label=None,
            markeredgecolor="k",
            legend=False,
        )
        # dummy_leg = ax_accidx_eff.legend()
        # dummy_leg.remove()
        # ax_accidx_eff.add_artist(legend)
    lines, labels = ax_accidx_eff4.get_legend_handles_labels()
    lines = [lines[-3], lines[-1]]
    labels = ["yes", "no"]
    ax_accidx_eff4.legend(
        lines,
        labels,
        title="branches with extra synapse(s)",
        title_fontsize="x-small",
        loc=(0.0, 1.0),
        ncol=2,
        mode="expand",
        borderaxespad=0,
        borderpad=0,
        fontsize="xx-small",
        frameon=False,
        # facecolor='w', framealpha=0.5, edgecolor='None',
    )

    ax_accidx_eff.annotate(
        f"\n{settings.NABLAEGABA} (mV)",
        xy=(1.015, 0.82),
        xycoords="axes fraction",
        rotation=00,
        va="bottom",
        ha="left",
        fontsize="small",
    )
    # ax_accidx_eff.annotate(f"Branches\n" + "\n".join(str(b) for b in branch_bases),
    #                        xy=(-0.03, 1.), xycoords='axes fraction',
    #                        rotation=00, va='bottom', ha='center', fontsize='small')
    ax_il_dist.set_xlim(0, 1)
    ax_il_dist.locator_params(axis="x", nbins=5)
    ax_accidx_eff.set_xlabel("Effective number of branches (%)")
    ax_accidx_eff.set_ylabel("Accumulation Index")
    ax_accidx_eff.set_xlim(0, auto=True)
    ax_accidx_eff4.set_xlim(0, auto=True)
    ax_accidx_eff4.set(xlabel="Effective number\nof branches (%)", ylabel="AccIdx")
    ax_il_dist_eff.set(
        xlabel="Effective number of branches (%)", ylabel=f"${settings.SL}_0$"
    )
    ax_il_n_eff.set(
        xlabel="Effective number of branches (%)", ylabel=f"${settings.SL}_0$"
    )
    ax_il_e_eff.set(xlabel="", ylabel=f"${settings.IL}_0$")
    ax_il_dist_eff.grid(True, axis="x")
    ax_il_n_eff.grid(True, axis="x")
    ax_il_e_eff.grid(True, axis="x")
    ax_accidx_eff.grid(True, axis="x")
    ax_accidx_eff4.grid(True, axis="x")
    ax_accidx_eff.set_xticks([0, 100, 200, 300, 400])
    ax_accidx_eff4.set_xticks([0, 100, 200, 300, 400])
    ax_il_n_eff.yaxis.set_major_locator(MaxNLocator(2))
    ax_il_e_eff.yaxis.set_major_locator(MaxNLocator(2, integer=True))

    ax_il_dist_eff.set_xticks(range(0, 400, 25), minor=True)
    ax_accidx_eff.set_xticks(range(0, 400, 25), minor=True)
    ax_accidx_eff4.set_xticks(range(0, 400, 25), minor=True)
    ax_accidx_eff4.set_xlim(100)
    # ax_il_dist_eff.set_ylim(0)
    # ax_il_n_eff.set_ylim(0)
    # ax_il_e_eff.set_ylim(0)
    import numpy as np
    from utils.plot_utils import CurvedText

    y_start = 0.2
    for line in ax_il_dist.get_lines():
        if line.get_linewidth() == 2 * 0.25:
            y_start = line.get_ydata()[0]
            break
    y_min = y_start
    for line in ax_il_dist.get_lines():
        y_min = min(y_min, line.get_ydata()[-1])
    x = np.linspace(0, 1, 100)
    text = CurvedText(
        x=x,
        y=y_min + y_start / np.exp(2.4 * x),
        text="silent branches",
        va="top",
        c="k",
        fontsize="x-small",
        axes=ax_il_dist,
    )
    ax_eff_branches[0].annotate(
        "silent branch",
        xy=(0.0, 0.45),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize="xx-small",
        rotation=-28,
    )
    if settings.SAVE_FIGURES:
        plot_save("output/figure_structure_eff.png", figs=[fig], close=False)
        plot_save("output/figure_structure_eff.pdf")
