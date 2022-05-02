import copy
import numpy as np
import pandas as pd
from matplotlib import cm, colors
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from utils import settings
from inhib_level.math import accumulation_index
from utils.plot_utils import (
    letter_axes,
    copy_lines,
    adjust_spines,
    plot_save,
    new_gridspec,
)

import logging

logger = logging.getLogger()


def figure_input_structure_loc_dist(dist_base=8, plot_e_off="0.0"):
    """ How input structure (location & distribution) affect IL accum

    Ai   ii   iii  |   B  |   C

    Di   ii   iii  |      E

    A: shape plots with inhibitory locations [0, 0.5, 1]
        Branches = 4
        E = -2

    B: IL vs Location
        Branches = 2,4,8
        Locations = [0.005, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.995]
        E = -2

    C: AI vs Location

    Di: Diffused across neuron - synapse per branch heatmap

    Dii: Diffused on one branch heatmap (clustered_n)

    Diii: Clustered heatmap (all synapses one location on one branch)

    E: IL vs Distance for D


    """
    from main import run_inhib_level

    logger.info("*" * 50 + "figure_input_structure_loc_dist" + "*" * 50)

    # params for simulations
    tm = settings.TSTOP
    loc_list = [0.005, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.995]
    radial_list = [1, 2, 4, 6, 8]
    e_offset_list = [0.0, -2.0, -5.0]
    dist_list = ["diffused", "clustered_n", "clustered"]
    dist_names = {"diffused": "Tree", "clustered_n": "Branch", "clustered": "Focal"}

    # convert to str for `run_inhib_level`
    loc_list_str = " ".join(f"[{loc}]" for loc in loc_list)
    radial_list_str = " ".join(str(r) for r in radial_list)
    e_offsets = " ".join(str(e) for e in e_offset_list)
    dist_list_str = " ".join(dist_list)

    # visual params
    plot_locs = [0.005, 0.5, 0.8]
    plot_radial = "4"  # str type as it is a consistent dict key for `plot_dict` with different `plot_group_by`
    plot_radial_key = int(plot_radial)
    plot_e_offs = ["0.0", "-2.0"]
    cmap_dict = settings.cmap_dict["n_e"][plot_radial_key]
    cmap = cmap_dict["cmap"]
    marker_colors = copy.copy(cmap_dict)

    # Setup figure look
    height_ratios = [1, 1, 1]
    width_ratios = [1, 40, 40, 1]
    fig, gs = new_gridspec(
        nrows=len(height_ratios),
        ncols=len(width_ratios),
        figsize=(settings.PAGE_W_FULL, settings.PAGE_H_FULL * 3 / 4),
        grid_kwargs=dict(height_ratios=height_ratios, width_ratios=width_ratios),
    )
    save_png_figs = [fig]
    gs.update(left=0.05, right=0.95, top=0.95, hspace=0.7, wspace=0.5)
    ax_loc_shapes = []
    ax_dist_shapes = []

    # new sub gridspec for shapes with varying locations
    gs_shape_loc = GridSpecFromSubplotSpec(
        nrows=len(plot_e_offs),
        ncols=len(plot_locs) + 1,
        subplot_spec=gs[0, 0:2],
        width_ratios=[10] * len(plot_locs) + [1],
        hspace=0.2,
        wspace=0.2,
    )
    for j in range(len(plot_e_offs)):
        for i in range(len(plot_locs)):
            _tmp_ax = fig.add_subplot(gs_shape_loc[j, i])
            adjust_spines(
                _tmp_ax, [], 0, sharedy=True, sharedx=True
            )  # `shared` means to not hide the axis, only the spine
            ax_loc_shapes.append(_tmp_ax)
            if i == 0:
                _tmp_ax.set_ylabel(
                    f"{settings.NABLAEGABA}\n= {plot_e_offs[j]} mV",
                    color=settings.cmap_dict["n_e"][plot_radial_key][plot_e_offs[j]],
                    fontsize="x-small",
                )
            if j == len(plot_e_offs) - 1:
                txt = f"{plot_locs[i]:.1f}"
                if i == len(plot_locs) // 2:
                    txt += f"\n{settings.LOCATION_X_}"
                _tmp_ax.set_xlabel(txt)
    ax_cb_loc = fig.add_subplot(gs_shape_loc[:, -1])

    # new sub gridspec for IL plot(s)
    gs_ILs = GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[0, 2:], wspace=0.5)
    ax_SL_loc = fig.add_subplot(gs_ILs[0, 0])
    ax_IL_loc = fig.add_subplot(gs_ILs[1, 0])
    ax_accidx_loc_n = fig.add_subplot(gs_ILs[:, -1])  # for both E's but a single N

    # Acc Idx directly in gs
    ax_accidx_loc_e0 = fig.add_subplot(gs[1, 1])  # for E=0
    ax_accidx_loc_e2 = fig.add_subplot(gs[1, 2])  # for E=-2

    # sub grid spec for shape plots of synapse distributions
    gs_shape_dist = GridSpecFromSubplotSpec(
        nrows=1,
        ncols=len(dist_list) + 1,
        subplot_spec=gs[-1, 0:2],
        width_ratios=[10] * len(dist_list) + [1],
        hspace=0.2,
        wspace=0.2,
    )
    for i in range(len(dist_list)):
        ax_dist_shapes.append(fig.add_subplot(gs_shape_dist[0, i]))
    ax_cb_dist = fig.add_subplot(fig.add_subplot(gs_shape_dist[0, -1]))
    ax_IL_dist = fig.add_subplot(gs[-1, 2])

    ax_IL_dict = {"0.0": ax_SL_loc, "-2.0": ax_IL_loc}
    ax_accidx_dict = {"0.0": ax_accidx_loc_e0, "-2.0": ax_accidx_loc_e2}

    letter_axes(ax_loc_shapes[0], start="A", xy=(-0.05, 1.07))
    letter_axes(ax_SL_loc, start="B", xy=(-0.4, 1.07))
    letter_axes(ax_accidx_loc_n, start="C", xy=(-0.4, 1.03))
    letter_axes(
        ax_accidx_loc_e0,
        ax_accidx_loc_e2,
        subscript="D",
        repeat_subscript=True,
        xy=(-0.3, 1.05),
    )
    letter_axes(ax_dist_shapes[0], start="E", xy=(-0.05, 1.05))
    letter_axes(ax_IL_dist, start="F", xy=(-0.18, 1.05))

    # logger.info(
    #     """
    # ####################################################################################################################
    # # IL, Accumulation vs LOCATION (A,B,C)
    # ####################################################################################################################
    # """
    # )

    # plot_dict, sim_type, saved_args = run_inhib_level(
    #     f"--radial {radial_list_str} --loc {loc_list_str} --e_offsets {e_offsets}"
    #     " --synapse_dists=diffused_matched "
    #     " --plot_group_by=num_dendrites_arr --plot_color_by=e_offsets --plot_shape"
    #     " --sections=radial_dends_1" + " --quick"
    # )

    # for ax_key, ax in plot_dict[plot_radial][2].items():
    #     if ax_key == settings.IL:
    #         ax_lines = ax.get_lines()
    #         lines = []
    #         marker_lines = []
    #         for li, line in enumerate(ax_lines):
    #             for e in plot_e_offs:
    #                 if f"e={e}" in line.get_label():
    #                     lines.append(ax_lines[li - 2])
    #                     marker_lines.append(ax_lines[li - 1])
    #                     copy_lines_kwargs = dict(
    #                         rel_lw=0.5, color=cmap_dict["line"], clip_on=False
    #                     )
    #                     copy_lines(
    #                         ax_lines[li - 2],
    #                         ax_IL_dict[e],
    #                         zorder=-99,
    #                         alpha=0.5,
    #                         **copy_lines_kwargs,
    #                     )
    #                     copy_lines(
    #                         ax_lines[li - 1],
    #                         ax_IL_dict[e],
    #                         zorder=+99,
    #                         markerfacecolor=marker_colors[e],
    #                         markeredgecolor="k",
    #                         ms=7,
    #                         **copy_lines_kwargs,
    #                     )
    #     elif ax_key.startswith("SHAPE") and ax_key.endswith("AX"):
    #         # plot neuron shapes
    #         n = int(ax_key[ax_key.index("n=") + 2 : ax_key.index("/")])
    #         e = ax_key[ax_key.index("e=") + 2 : ax_key.index("(")]
    #         x = float(ax_key[ax_key.index("x=[") + 3 : ax_key.index("]")])
    #         if x in plot_locs and e in plot_e_offs:
    #             ax_idx = int(plot_locs.index(x) + plot_e_offs.index(e) * len(plot_locs))
    #             copy_lines(ax, ax_loc_shapes[ax_idx], clip_on=False)
    #             underscore_idx = ax_key.rfind("_")
    #             annotation_key = ax_key[:underscore_idx] + "_ANNOTATIONS"
    #             annotations = plot_dict[plot_radial][2][annotation_key]
    #             for annotation in annotations:
    #                 # get consistent color across figure
    #                 annotation["arrowprops"]["facecolor"] = marker_colors[e]
    #                 ax_loc_shapes[ax_idx].annotate(**annotation)

    # il_dict = saved_args["il_dict"]
    # df_accum = pd.DataFrame(
    #     index=loc_list, columns=pd.MultiIndex.from_product([radial_list, e_offset_list])
    # )
    # for key, df in il_dict.items():
    #     if key == "units":
    #         continue
    #     n = int(key[key.index("n=") + 2 : key.index("/")])
    #     e = float(key[key.index("e=") + 2 : key.index("(")])
    #     accum = accumulation_index(df, loc_list)
    #     for x in loc_list:
    #         df_accum.loc[x, (n, e)] = accum[x][tm]

    # for n_dend in radial_list:
    #     cmap = settings.cmap_dict["n_e"][n_dend]
    #     for e, data in df_accum[n_dend].iteritems():
    #         plot_args = dict(
    #             linestyle="--",
    #             marker="v",
    #             markeredgecolor="k",
    #             color=cmap[e],
    #             alpha=1.0,
    #             clip_on=False,
    #             zorder=10,
    #         )
    #         ax_accidx_dict[str(e)].plot(data, **plot_args)
    #         if n_dend == plot_radial_key:
    #             ax_accidx_loc_n.plot(data, ms=7, lw=1, **plot_args)
    #             ax_accidx_loc_n.annotate(
    #                 f"{e:>4.1f} mV",
    #                 xy=(1, data.iloc[-1] * 0.9),
    #                 ha="center",
    #                 va="top",
    #                 fontsize="small",
    #                 c=cmap[e],
    #             )

    logger.info(
        """
    ####################################################################################################################
    # SYNAPSE DISTRIBUTION (D & E)
    ####################################################################################################################
    """
    )
    # dist_base = 8
    # plot_e_off = "0.0"
    plot_dict, sim_type, saved_args = run_inhib_level(
        f'--radial {dist_base} --e_offsets {e_offsets} --loc {" ".join(["0.2"]*dist_base)} '
        f" --synapse_dists {dist_list_str} "
        " --kcc2=N "
        " --plot_group_by=e_offsets --plot_color_by=synapse_dist"
        " --quick"
        " --plot_shape"
    )
    # plot neuron shapes
    for ax_key, ax in plot_dict[plot_e_off][2].items():
        if ax_key == settings.IL:
            copy_lines(ax, ax_IL_dist, ms=8, rel_lw=1.2)
            ax_IL_dist.set(xlabel=ax.get_xlabel(), ylabel=ax.get_ylabel())
            lines = [
                line for line in ax_IL_dist.get_lines() if "_1" in line.get_label()
            ]
            dummy_lines = [
                Line2D([], [], color=l.get_color(), marker="v", markeredgecolor="k")
                for l in lines
            ]
            ax_IL_dist.legend(
                dummy_lines,
                [dist_names[v] for v in dist_list],
                title="Distribution",
                loc=3,
                bbox_to_anchor=(1.0, 0.0),
                frameon=False,
                fontsize="small",
            )
        if ax_key.startswith("SHAPE") and ax_key.endswith("AX"):
            n = int(ax_key[ax_key.index("n=") + 2 : ax_key.index("/")])
            dist = ax_key[ax_key.index("\n") : ax_key.index("x=")].strip()
            x = list(ax_key[ax_key.index("x=[") + 3 : ax_key.index("]")])
            e = ax_key[ax_key.index("e=") + 2 : ax_key.index("(")]
            if e == plot_e_off and dist in dist_list:
                ax_idx = dist_list.index(dist)
                copy_lines(ax, ax_dist_shapes[ax_idx])
                underscore_idx = ax_key.rfind("_")
                annotation_key = ax_key[:underscore_idx] + "_ANNOTATIONS"
                annotations = plot_dict[plot_e_off][2][annotation_key]
                for annotation in annotations:
                    ax_dist_shapes[ax_idx].annotate(**annotation)
                ax_dist_shapes[ax_idx].annotate(
                    dist_names[dist],
                    xy=(0.5, -0.1),
                    xycoords="axes fraction",
                    ha="center",
                    va="top",
                    fontsize="smaller",
                )
    adjust_spines(ax_dist_shapes, [], 0)

    cb = fig.colorbar(
        cm.ScalarMappable(
            norm=colors.Normalize(0, 1), cmap=settings.IL_config[settings.IL]["cmap"]
        ),
        cax=ax_cb_loc,
        orientation="vertical",
    )
    cb.set_ticks([0, 1])
    cb.set_ticklabels(["min", "max"])
    # ax_cb_loc.annotate("max", xy=(0.5, 1.), xycoords='axes fraction', va='bottom', ha='center', fontsize='x-small')
    # ax_cb_loc.annotate("min", xy=(0.5, -0.02), xycoords='axes fraction', va='top', ha='center', fontsize='x-small')
    cb_dist = fig.colorbar(
        cm.ScalarMappable(
            norm=colors.Normalize(0, 1), cmap=settings.IL_config[settings.IL]["cmap"]
        ),
        cax=ax_cb_dist,
        orientation="vertical",
    )
    cb_dist.set_ticks([0, 1])
    cb_dist.set_ticklabels(["min", "max"])

    ax_dist_shapes[1].annotate(
        "Distribution",
        xy=(0.5, -0.3),
        xycoords="axes fraction",
        ha="center",
        va="top",
        fontsize="smaller",
    )
    ax_dist_shapes[0].annotate(
        f"{settings.NABLAEGABA} = {plot_e_off} mV",
        xy=(0.0, 0.5),
        xycoords="axes fraction",
        rotation=90,
        fontsize="small",
        ha="right",
        va="center",
    )
    # ax_loc_shapes[-2].annotate(settings.LOCATION_X_, xy=(.5, -0.3), xycoords='axes fraction', ha='center', va='top',
    #                            fontsize='small')

    ax_SL_loc.set(
        xticklabels=[],
        xlim=(0, 1),
        ylabel=settings.SL,
        ylim=np.array(ax_SL_loc.get_ylim()) * 1.1,
    )
    ax_IL_loc.set(
        xlabel=settings.DISTANCE_X_,
        xlim=(0, 1),
        ylabel=settings.IL,
        ylim=np.array(ax_IL_loc.get_ylim()) * 1.1,
    )
    ax_accidx_loc_n.set(
        xlabel=settings.LOCATION_X_,
        xlim=(0, 1),
        ylabel=settings.ACC_IDX,
        ylim=(0, ax_accidx_loc_n.get_ylim()[1] * 1.1),
    )
    ax_accidx_loc_e0.set(
        xlabel=settings.LOCATION_X_,
        xlim=(0, 1),
        ylabel=settings.ACC_IDX,
        ylim=(0, ax_accidx_loc_e0.get_ylim()[1] * 1.1),
    )
    ax_accidx_loc_e2.set(
        xlabel=settings.LOCATION_X_,
        xlim=(0, 1),
        ylabel=settings.ACC_IDX,
        ylim=(0, ax_accidx_loc_e2.get_ylim()[1] * 1.1),
    )
    ax_accidx_loc_n.yaxis.set_major_locator(MaxNLocator(3))
    ax_accidx_loc_e0.yaxis.set_major_locator(MaxNLocator(3))
    ax_accidx_loc_e2.yaxis.set_major_locator(MaxNLocator(3))
    ax_IL_dist.set(xlabel=settings.DISTANCE_X_, ylabel=settings.IL, xlim=(0, 1))

    legend = ax_accidx_loc_n.legend(
        [],
        title=f"{settings.NABLAEGABA}",
        title_fontsize="small",
        loc="lower center",
        bbox_to_anchor=(1.0, 0.9),
        frameon=False,
        fontsize="small",
        borderpad=0.0,
        borderaxespad=0.0,
    )
    ax_accidx_loc_e2.legend(
        radial_list,
        title="Branches",
        loc="upper left",
        bbox_to_anchor=(1, 1.23),
        ncol=1,
        frameon=False,
        fontsize="small",
        borderaxespad=0.3,
    )

    for e, _ax in ax_accidx_dict.items():
        _ax.set_title(
            f"{settings.NABLAEGABA} = {e} mV",
            color=settings.cmap_dict["n_e"][1][e],
            fontsize="medium",
            va="top",
            ha="center",
        )

    ax_SL_loc.set_yticks([0.2, 0.4])
    fig.align_ylabels([ax_SL_loc, ax_IL_loc])
    if settings.SAVE_FIGURES:
        plot_save(
            "output/figure_structure_loc_dist.png", figs=save_png_figs, close=False
        )
        plot_save(
            "output/figure_structure_loc_dist.svg", figs=save_png_figs, close=False
        )
        plot_save("output/figure_structure_loc_dist.pdf")
    else:
        import shared

        shared.show_n(1)
