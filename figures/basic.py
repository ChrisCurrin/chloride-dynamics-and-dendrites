import numpy as np
import pandas as pd
from matplotlib import cm, colors

from utils import settings
from inhib_level.math import accumulation_index
from utils.plot_utils import (
    letter_axes,
    copy_lines,
    adjust_spines,
    create_zoom,
    plot_save,
    new_gridspec,
)
from utils.settings import GS_C, GS_C_third, WPAD, HPAD, GS_R_third
import logging

logger = logging.getLogger("basic")


def figure_basic():
    """
    A  i   ii   iii   iv   v
    -------------------------
    B  i   ii   iii   iv   v
    -------------------------
       C    |         D

    A i-iv: Diagram of synapse distribution with different branches [1, 2, 4, 8]
    A v: IL vs Distance for branches at Erev =  0 mV
    B i-iv: Diagram of synapse distribution with different branches [1, 2, 4, 8]
    B v: IL vs Distance for branches at Erev =  -1 mV
    C: IL vs Distance for branches at Erev = -2 mV
        Also inset of accumulation index explanation

    D: Accumulation Index versus Number of Branches at Erev = [0, -1, -2, -5]

    :return:
    """
    from main import run_inhib_level

    logger.info("*" * 50 + "figure_basic" + "*" * 50)
    # FIGURE SETTINGS
    fig_settings = {
        "heatmap_title": 0,
        "heatmap_label": 0,
        "ax_ylabel": 1,
        "heatmap_location": "right",
        "heatmaps": 2,
    }

    fig, gs = new_gridspec(figsize=(settings.PAGE_W_FULL, 3 * settings.PAGE_H_4th))
    gs.update(left=0.08, top=0.93, right=0.95)
    # gca().set_aspect('equal', adjustable='box')
    plot_branches = [1, 2, 4, 8]
    colspan = int((GS_C - GS_C_third - WPAD) / len(plot_branches))
    hpad = HPAD + 2
    ax_SL_branches = []
    for i in range(len(plot_branches)):
        ax_SL_branches.append(
            fig.add_subplot(gs[0 : GS_R_third - HPAD, colspan * i : colspan * (i + 1)])
        )
    ax_SL = fig.add_subplot(gs[0 : GS_R_third - hpad, -GS_C_third:-WPAD])

    ax_IL_branches = []
    for i in range(len(plot_branches)):
        ax_IL_branches.append(
            fig.add_subplot(
                gs[GS_R_third : GS_R_third * 2 - HPAD, colspan * i : colspan * (i + 1)]
            )
        )

    ax_IL = fig.add_subplot(gs[GS_R_third : GS_R_third * 2 - hpad, -GS_C_third:-WPAD])

    ax_acc_idx_explain = fig.add_subplot(gs[-GS_R_third:, 0:GS_C_third])
    ax_acc_idx = fig.add_subplot(gs[-GS_R_third:, GS_C_third + WPAD : -WPAD])

    if fig_settings["heatmaps"] > 1 and fig_settings["heatmap_location"] == "right":
        ax_IL_cb = fig.add_subplot(
            gs[
                GS_R_third : GS_R_third * 2 - HPAD,
                -GS_C_third - WPAD - 2 : -GS_C_third - WPAD - 1,
            ]
        )
        ax_SL_cb = fig.add_subplot(
            gs[0 : GS_R_third - HPAD, -GS_C_third - WPAD - 2 : -GS_C_third - WPAD - 1]
        )
    elif fig_settings["heatmaps"] == 1 or fig_settings["heatmap_location"] == "right":
        ax_cb = fig.add_subplot(
            gs[
                GS_R_third - 5 : GS_R_third - 1,
                -GS_C_third - WPAD - 3 : -GS_C_third - WPAD - 2,
            ]
        )
    else:
        ax_SL_cb = ax_SL_branches[0]
        ax_IL_cb = ax_IL_branches[0]

    for ax_branch in [*ax_SL_branches, *ax_IL_branches]:
        pos = ax_branch.get_position()
        ax_branch.set_position([pos.x0 - 0.05, pos.y0, pos.width, pos.height])

    ha = "center"
    letter_axes(ax_SL_branches[0], subscript="A", xy=(-0.2, 1.07), ha=ha)
    letter_axes(ax_SL_branches[1:], start=2, subscript="A", xy=(-0, 1.07), ha=ha)
    letter_axes(ax_SL, start=5, subscript="A", xy=(-0.45, 1.07), ha=ha)
    letter_axes(ax_IL_branches[0], subscript="B", xy=(-0.2, 1.07), ha=ha)
    letter_axes(ax_IL_branches[1:], start=2, subscript="B", xy=(-0, 1.07), ha=ha)
    letter_axes(ax_IL, start=5, subscript="B", xy=(-0.45, 1.07), ha=ha)
    letter_axes(ax_acc_idx_explain, start="C", xy=(-0.26, 1.05), ha=ha)
    letter_axes(ax_acc_idx, start="D", xy=(-0.17, 1.05), ha=ha)
    # ax_oi0 = plt.subplot2grid(grid_spec_size, (7, 8), rowspan=7, colspan=6, fig=fig)
    loc = [0.2]
    tm = settings.TSTOP
    plot_dict, sim_type, saved_args = run_inhib_level(
        f"--precise --radial 1 2 4 8 16 --loc {loc} --e_offsets 0 -1 -2 -5 --plot_group_by=e_offsets"
        f" --sections=radial_dends_1 --plot_shape" 
        # " --with-v-trace"
    )
    xlim = (0, 1)
    copy_lines(plot_dict["0"][1][0], ax_SL, rel_lw=1.5)
    for line in ax_SL.get_lines():
        line.set_markerfacecolor(
            settings.COLOR.lighten_color(line.get_markerfacecolor(), 0.6)
        )
    copy_lines(plot_dict["-1"][1][0], ax_IL, rel_lw=1.5)
    for line in ax_IL.get_lines():
        line.set_markerfacecolor(
            settings.COLOR.lighten_color(line.get_markerfacecolor(), 1)
        )
    copy_lines(plot_dict["-2"][1][0], ax_acc_idx_explain, rel_lw=1.5)
    for line in ax_acc_idx_explain.get_lines():
        line.set_markerfacecolor(
            settings.COLOR.lighten_color(line.get_markerfacecolor(), 1.5)
        )
    for erev, ax_list in zip(["0", "-1"], [ax_SL_branches, ax_IL_branches]):
        for ax_key, ax in plot_dict[erev][2].items():
            if ax_key.startswith("SHAPE") and ax_key.endswith("AX"):
                n = int(ax_key[ax_key.find("n=") + 2 : ax_key.find("/")])
                if n in plot_branches:
                    ax_idx = plot_branches.index(n)
                    copy_lines(ax, ax_list[ax_idx])
                    underscore_idx = ax_key.rfind("_")
                    annotation_key = ax_key[:underscore_idx] + "_ANNOTATIONS"
                    annotations = plot_dict[erev][2][annotation_key]
                    for annotation in annotations:
                        if erev == "0":
                            annotation["arrowprops"][
                                "facecolor"
                            ] = settings.COLOR.lighten_color(
                                annotation["arrowprops"]["facecolor"], 0.6
                            )
                        elif erev == "-1":
                            annotation["arrowprops"][
                                "facecolor"
                            ] = settings.COLOR.lighten_color(
                                annotation["arrowprops"]["facecolor"], 1
                            )
                        ax_list[ax_idx].annotate(**annotation)
    adjust_spines([*ax_SL_branches, *ax_IL_branches], [], position=0)

    if fig_settings["heatmaps"] > 1:
        for _ax, _long_label, _short_label in zip(
            [ax_SL_cb, ax_IL_cb],
            [settings.SHUNT_LEVEL, settings.INHIBITORY_LEVEL],
            [settings.SL, settings.IL],
        ):
            cb_kwargs = dict()
            if fig_settings["heatmap_location"] == "right":
                cb_kwargs["orientation"] = "vertical"
                cb_kwargs["cax"] = _ax
            elif fig_settings["heatmap_location"] == "bottom":
                cb_kwargs["orientation"] = "horizontal"
                cb_kwargs["ax"] = _ax
            cb = fig.colorbar(
                cm.ScalarMappable(
                    norm=colors.Normalize(0, 1),
                    cmap=settings.IL_config[settings.IL]["cmap"],
                ),
                **cb_kwargs,
            )
            # cb.set_label(f"[{_label}]", fontsize='x-small')  # below/next to colorbar (same as ticks)
            cb.set_label(
                _long_label.replace(" ", "\n"), rotation=60, va="center"
            )  # below/next to colorbar (same as ticks)
            cb.set_ticks([0, 1])
            cb.set_ticklabels(["min", "max"])
            cb.ax.tick_params(labelsize="x-small")
            # cb.ax.set_title(_title, fontsize='small') # above colorbar
    elif fig_settings["heatmaps"] == 1:
        cb = fig.colorbar(
            cm.ScalarMappable(
                norm=colors.Normalize(0, 1),
                cmap=settings.IL_config[settings.IL]["cmap"],
            ),
            cax=ax_cb,
            orientation="vertical",
        )
        # cb.set_label(f"{settings.SHUNT_LEVEL} [SL] / \n{settings.INHIBITORY_LEVEL} [IL]", fontsize='small',
        #              rotation=0, ha='left', va='center')  # below/next to colorbar (same as ticks)
        # cb.set_label(f"{settings.SL}\n{settings.IL}", fontsize='small',
        #              rotation=0, ha='left', va='center')  # below/next to colorbar (same as ticks)

        cb.set_ticks([0, 1])
        cb.set_ticklabels(["min", "max"])
        cb.ax.tick_params(labelsize="x-small")

    # plot explainer for accumulation index
    sample_lines = []
    ymin, ymax = 0, 0
    xmax = loc[0] + 0.05  # some buffer
    lines = ax_acc_idx_explain.get_lines()
    # need to get the fourth 'radial_dends_1' for n=8
    n_radial = 0
    for line in lines:
        if line.get_label() == "radial_dends_1":
            n_radial += 1
        if n_radial == len(plot_branches):
            # should be at least 3 lines - 2 per branch (line and marker) plus legend
            sample_lines.append(line)
            if line.get_marker() == "v":
                ymin = line.get_ydata()[0]
            elif ymax == 0:
                ymax = line.get_ydata()[0]
    ax_to_zoom = ax_acc_idx_explain
    ax_inset = create_zoom(
        ax_to_zoom,
        2.0,
        lines=sample_lines,
        loc="upper right",
        xlim=(0, xmax),
        ylim=(np.floor(ymin), np.ceil(ymax)),
        xticks=2,
        yticks=2,
        connector_kwargs="None",
        inset_kwargs=dict(bbox_to_anchor=None, bbox_transform=None),
        box_kwargs=dict(zorder=999),
        rel_lw=1.5,
    )

    xlabel = plot_dict["0"][1][0].get_xlabel()
    ylabel = plot_dict["0"][1][0].get_ylabel()

    #
    il_dict = saved_args["il_dict"]
    index = il_dict["units"].index
    df_accum = pd.DataFrame()
    for key, df in il_dict.items():
        if key == "units":
            continue
        n = key[key.index("/") + 1 : key.index("\n")]
        e = key[key.index("e=") + 2 : key.index("(")]
        accum = accumulation_index(df, loc)
        df_accum.loc[int(n), e] = accum[loc[0]][tm]

    df_accum.plot(
        ax=ax_acc_idx, linestyle=":", cmap=settings.cmap_dict["num_synapses_e"][1]
    )
    acc_handles, acc_labels = ax_acc_idx.get_legend_handles_labels()

    # fig_oi, ax_oi0 = plt.subplots()
    # offset_dict = offset_index(il_dict, saved_args['num_dendrites_arr'])
    # df_oi = pd.DataFrame(index=index)
    # for key, df in offset_dict.items():
    #     n = key[key.index('/') + 1:key.index('\n')]
    #     df_oi[int(n)] = df
    # df_oi = df_oi.sort_index(axis=1)  # sort columns
    # df_oi.loc['soma', 0.5].plot(ax=ax_oi0, marker='.')
    # loc_idx = abs(loc[0] - index.levels[1]).argmin()
    # df_oi.loc['radial_dends_1'].iloc[loc_idx].plot(ax=ax_oi0, marker='v')
    # ax_oi0.legend(['soma', f'{loc[0]:.1f}'], title='Location (X)')
    # ax_oi0.set_ylabel("Offset Index (ratio)")
    # ax_oi0.set_xlabel("Number of branches")

    ax_acc_idx.hlines(
        [1],
        xmin=0,
        xmax=df_accum.index.max(),
        linestyle="--",
        lw=1.0,
        colors="k",
        alpha=0.5,
    )

    lines, labels = ax_SL.get_legend_handles_labels()
    each_n_line = [line for line in lines if line.get_label()[0] == "n"]
    each_n_color = [line.get_color() for line in each_n_line]
    legend = ax_SL.legend(
        each_n_line,
        "1 2 4 8 16".split(),
        loc="upper left",
        title="Branches (N)",
        fontsize="smaller",
        bbox_to_anchor=(0.95, 1.0),
        frameon=False,
        labelspacing=1.0,
        handletextpad=2.0,
    )
    for t in legend.get_texts():
        t.set_ha("right")

    # plot markers with color of N
    #   first get colors from an existing axis
    for n, _series in df_accum.iterrows():
        _df = _series.to_frame().T
        _df.plot(
            ax=ax_acc_idx,
            marker="v",
            markeredgecolor="k",
            ms=8,
            legend=False,
            label=None,
            cmap=settings.cmap_dict["num_synapses_e"][n],
        )
    for handle in acc_handles:
        handle.set_marker("v")
        handle.set_markeredgecolor("k")
        # handle.set_alpha(0.6)
        handle.set_markersize(8)
    legend = ax_acc_idx.legend(
        acc_handles,
        acc_labels,
        title=settings.GRADEGABA,
        loc="lower left",
        bbox_to_anchor=(1.0, 0.0),
        frameon=False,
        handletextpad=2.0,
    )
    for t in legend.get_texts():
        t.set_ha("right")

    # limits, labels, titles, etc.
    ax_SL.set_ylim(top=ax_SL.get_ylim()[1] * 1.1)
    ax_SL.set_xlim(xlim)
    ax_IL.set_xlim(xlim)
    ax_acc_idx_explain.set_xlim(xlim)
    ax_acc_idx.set_xlim(0, 17)
    ax_acc_idx.set_xticks([1, 2, 4, 8, 16])

    # ax_SL.set_ylabel(settings.SL)
    # ax_IL.set_ylabel(settings.IL)
    # ax_SL.set_ylabel(settings.SHUNT_LEVEL)
    # ax_IL.set_ylabel(settings.INHIBITORY_LEVEL)
    ax_acc_idx_explain.set_ylabel(f"{settings.INHIBITORY_LEVEL} [{settings.IL}]")
    ax_acc_idx.set_ylabel("Accumulation Index")
    ax_inset.plot(0, ymax, "o", markerfacecolor=settings.COLOR.O, markeredgecolor="k")
    ax_inset.annotate(
        f"{settings.ILd.replace('d','0')} (junction)",
        xy=(0, ymax),
        xytext=(20.0, 0),
        textcoords="offset points",
        va="center_baseline",
        ha="left",
        zorder=-10,
        fontsize="small",
        arrowprops=dict(arrowstyle="-", connectionstyle="arc3,rad=0", facecolor="C7"),
    )
    ax_inset.annotate(
        f"{settings.ILd.replace('d','d=i')}\n({settings.GABAAR})",
        xy=(loc[0], ymin),
        xytext=(0.0, 14.0),
        textcoords="offset points",
        va="bottom",
        ha="center",
        zorder=-10,
        fontsize="small",
        arrowprops=dict(arrowstyle="-", connectionstyle="arc3,rad=0"),
    )
    ax_inset.tick_params(color="grey")

    ax_SL.set_xlabel(xlabel)
    ax_IL.set_xlabel(xlabel)
    ax_acc_idx_explain.set_xlabel(xlabel)
    ax_acc_idx.set_xlabel("Number of branches (N)")

    ax_SL.set_title(f"{settings.GRADEGABA} =  0 mV", ha="center", fontsize="medium")
    ax_IL.set_title(f"{settings.GRADEGABA} = -1 mV", ha="center", fontsize="medium")
    ax_inset.set_title("Accumulation Index", fontsize="small", va="top")
    ax_acc_idx_explain.set_title(
        f"{settings.GRADEGABA} = -2 mV", ha="center", va="bottom", fontsize="medium"
    )

    if settings.SAVE_FIGURES:
        plot_save("output/figure_basic.png", figs=[fig], close=False)
        plot_save("output/figure_basic.pdf")


if __name__ == "__main__":
    figure_basic()
