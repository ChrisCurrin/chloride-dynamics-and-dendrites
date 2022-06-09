import copy
from typing import List

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


def figure_dynamic_il_loc():
    """ How does dynamic Cl- affect IL

    """
    from matplotlib.gridspec import GridSpecFromSubplotSpec
    from main import run_inhib_level
    from shared import env_var

    logger.info("*" * 50 + "figure_dynamic_il_loc" + "*" * 50)

    # local vars common between nested methods
    tstop = 500
    tm = 100
    truncate_kwargs = dict(minval=0.2, maxval=0.8, n=100)
    cmap_idx = 150  # index between 0 and 255
    sample_loc = 0.2
    sample_N = 1
    radial_list = [1, 4]
    radials_morph = [1, 4]
    e_offset = -5
    highlight_cmap = settings.truncate_colormap(
        settings.n_branches_cmaps[sample_N], **truncate_kwargs
    )
    hl_color = highlight_cmap(0)
    # Shorthands
    _STATIC = settings.STATIC_CHLORIDE_STR_ABBR
    _DYNAMIC = settings.DYNAMIC_CHLORIDE_STR_ABBR
    _ILDIFF = settings.ILdiff
    plot_settings = copy.deepcopy(settings.IL_config)

    plot_names = [
        _STATIC,
        _DYNAMIC,
        _ILDIFF,
        "relative",
        "EGABA",
    ]  # colormaps for morphologies
    plot_map = {"-": plot_names[0], settings.DELTA: plot_names[1]}
    morph_lw = 1.0

    # Figure settings
    gs_size = 200
    wpad = hpad = int(0.1 * gs_size)
    hm_space = 6
    hm_shrink = 2

    fig, gs = new_gridspec(
        2,
        1,
        figsize=(settings.PAGE_W_column, settings.PAGE_H_FULL_no_cap),
        grid_kwargs=dict(height_ratios=[7, 10]),
    )
    gs.update(top=0.95, left=0.2, right=0.88, hspace=0.2, bottom=0.07)
    # create sub-gridspecs for sections of the figure
    gs0 = GridSpecFromSubplotSpec(gs_size, gs_size, subplot_spec=gs[0])
    gs1 = GridSpecFromSubplotSpec(gs_size, gs_size, subplot_spec=gs[1])

    sub_hpad = hpad // 5
    heights = np.array([10, 5, 5])
    heights = heights / sum(heights)  # normalise
    heights = heights - sub_hpad / gs_size
    morph_sec_e = int(heights[0] * gs_size)
    ilX_s = morph_sec_e + hpad // 3
    ilX_e = ilX_s + int(heights[1] * gs_size)
    il0_s = ilX_e + sub_hpad
    # points for plotting morphology
    loc_points_morph = [0.0001, 0.2, 0.4, 0.6, 0.8, 0.9999]
    width_ratios = [10] * len(loc_points_morph) + [2]
    w_r = np.array(width_ratios)
    w_norm = w_r / np.sum(w_r)
    hm_space = int(w_norm[-1] * gs_size)
    r_l = len(radial_list)
    r2_h = int(gs_size / 2)
    r2_w = int((gs_size - hm_space) / len(loc_points_morph))

    ax_morph_loc = np.empty(
        shape=(len(radials_morph), len(plot_names), len(loc_points_morph) + 1),
        dtype=Axes,
    )  # shape is (# dendrites, [IL, egaba], points plus heatmap)

    loc_first_run = True
    ax_ilX_locs = []

    def cl_loc(
        radial, gridspec_r, loc_list=None, diam=1.0, constant_L=False, shared_clim=True
    ):
        nonlocal loc_first_run
        r = radials_morph.index(radial) if radial in radials_morph else -1
        shared_clim = shared_clim and _STATIC in plot_names and _DYNAMIC in plot_names
        # setup subgridspec
        gs_morph_loc = GridSpecFromSubplotSpec(
            len(plot_names),
            len(loc_points_morph) + 1,
            subplot_spec=gridspec_r[:morph_sec_e, :],
            hspace=0.3,
            width_ratios=width_ratios,
        )
        for y in range(len(plot_names)):
            for x in range(len(loc_points_morph)):
                ax_morph_loc[r, y, x] = fig.add_subplot(
                    gs_morph_loc[y : (y + 1), x : (x + 1)]
                )
            # heatmap
            if shared_clim and y == 0:
                ax_morph_loc[r, y, -1] = fig.add_subplot(gs_morph_loc[y : (y + 2), -1:])
            elif shared_clim and y == 1:
                ax_morph_loc[r, y, -1] = ax_morph_loc[r, y - 1, -1]
            else:
                ax_morph_loc[r, y, -1] = fig.add_subplot(gs_morph_loc[y : (y + 1), -1:])
            adjust_spines(ax_morph_loc[r, y, :-1], [], 0)  # don't hide heatmap spines
        # ax_ilXs_loc = fig.add_subplot(gridspec_r[ilXs_s:ilXs_e, int(r2_w/2):-hm_space - int(r2_w/2)])
        # ax_ilXd_loc = fig.add_subplot(gridspec_r[ilXd_s:ilXd_e, int(r2_w/2):-hm_space - int(r2_w/2)])
        ax_ilX_loc = fig.add_subplot(
            gridspec_r[ilX_s:, int(r2_w / 2) : -hm_space - int(r2_w / 2)]
        )
        # ax_il0_loc = fig.add_subplot(gridspec_r[il0_s:, int(r2_w/2):-hm_space - int(r2_w/2)])
        ax_ilX_locs.append(ax_ilX_loc)

        loc_list = loc_list or [
            0.0001,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            0.9999,
        ]
        loc_list = np.round(loc_list, 6)
        loc_list_zoom = np.round(np.arange(0, loc_list[1], 0.01)[1:], 6)
        loc_list_end = np.round(np.arange(0.9, 1, 0.01)[1:], 6)
        loc_index = np.round(np.linspace(0, 1, 101), 6)  # used for interpolation
        loc_index = np.append(loc_index, loc_list)
        loc_index = np.append(loc_index, loc_list_zoom)
        loc_index = np.append(loc_index, loc_list_end)
        loc_index = sorted(
            set(np.round(loc_index, 6))
        )  # sort a unique set with tiny differences ignored

        # Data structures
        df_di = pd.DataFrame(index=loc_index, columns=[_STATIC, _DYNAMIC])  # IL at x=X
        df_di0 = pd.DataFrame(index=loc_index, columns=[_STATIC, _DYNAMIC])  # IL at x=0
        df_accum = pd.DataFrame(index=loc_index, columns=[_STATIC, _DYNAMIC])  # AccIdx
        # variables for each shape plot
        df_loc = pd.DataFrame(
            columns=pd.MultiIndex.from_product([plot_names, loc_points_morph])
        )

        vinit = -65.0
        egaba = (
            vinit + e_offset
        )  # will be defined during run as long as loc_point and loc_list overlap

        arrowprops = None
        cmap = settings.cmap_dict["n_e"][radial]["cmap"]
        _c_line = settings.cmap_dict["n_e"][radial]["line"]
        _c_dark = settings.cmap_dict["n_e"][radial][e_offset]
        # run a simulation for each inhibitory location
        plotting_loc = list(loc_list) + list(loc_list_zoom) + list(loc_list_end)
        for li, loc in enumerate(plotting_loc):
            main_args = (
                f"--radial {radial} --loc={loc} --with-t-trace={loc} "
                f"--e_offsets {e_offset} --synapse_dists=diffused_matched --kcc2=C "
                f"--plot_color_by=num_synapses --plot_shape "
                f"--tstop={tstop} --tm={tm} --diams {diam} "
                f"--precise --sections=radial_dends_1 "
            )
            if constant_L:
                main_args += f" --constant_L={constant_L}"
            if r != -1 and loc in plotting_loc:
                main_args += " --nseg=267"
            plot_dict, sim_type, saved_args = run_inhib_level(main_args)

            il_dict = saved_args["il_dict"]
            ecl_dict = saved_args["ecl_dict"]
            index = il_dict["units"].index

            assert (
                len(il_dict) == 3
            )  # il_dict has length of 3 (static and dynamic and units)
            loc_idx = abs(loc - index.levels[1]).argmin()
            for key, _df in il_dict.items():
                if key == "units":
                    continue
                # n = int(key[key.index('n=') + 2:key.index('/')])
                # e = float(key[key.index('e=') + 2:key.index('(')])
                cl = key[key.index("(") + 1 : key.index(")")]
                df_di.loc[loc, plot_map[cl]] = _df.loc[("radial_dends_1", tstop)].iloc[
                    loc_idx
                ]
                df_di0.loc[loc, plot_map[cl]] = _df.loc[("radial_dends_1", tstop)].loc[
                    0
                ]
                accum = accumulation_index(_df, [loc])
                df_accum.loc[loc, plot_map[cl]] = accum[loc][tstop]
            if loc in loc_points_morph:
                ev = env_var()
                pcl, ecl, phco3, ehco3, egaba, vinit = (
                    ev["pcl"],
                    ev["ecl"],
                    ev["phco3"],
                    ev["ehco3"],
                    ev["egaba"],
                    ev["v_init"],
                )
                sta_iter_label = [key for key in il_dict if "(-)" in key][0]
                dyn_iter_label = [key for key in il_dict if settings.DELTA in key][0]
                df_loc[_STATIC, loc] = il_dict[sta_iter_label].loc[
                    ("radial_dends_1", tstop)
                ]
                df_loc[_DYNAMIC, loc] = il_dict[dyn_iter_label].loc[
                    ("radial_dends_1", tstop)
                ]
                df_loc["EGABA", loc] = (
                    pcl * ecl_dict[dyn_iter_label]["radial_dends_1"].iloc[-1]
                    + phco3 * ehco3
                )

                for ax_key, _ax in plot_dict[0][2].items():
                    if r != -1 and ax_key.startswith("SHAPE") and ax_key.endswith("AX"):
                        # one of the sims to have the morphology plotted
                        ax_loc_idx = loc_points_morph.index(loc)
                        underscore_idx = ax_key.rfind("_")
                        annotation_key = ax_key[:underscore_idx] + "_ANNOTATIONS"
                        annotations = plot_dict[0][2][annotation_key]
                        for ax_loc in flatten(ax_morph_loc[r, :, ax_loc_idx]):
                            lw = morph_lw + 2 if radial == 1 else morph_lw
                            copy_lines(_ax, ax_loc, linewidth=lw)

                            # add annotations
                            for annotation in annotations:
                                if arrowprops is None:
                                    arrowprops = annotation["arrowprops"]
                                ann = copy.deepcopy(annotation)
                                _c = annotation["arrowprops"]["facecolor"]
                                egaba_t_loc = df_loc["EGABA", loc].iloc[
                                    -1
                                ]  # last time point for this location
                                _egaba_perc_increase = (egaba_t_loc - egaba) / (
                                    vinit - egaba
                                )
                                ann["xytext"] = (0, 4)
                                ax_loc.annotate(**ann)

        df_loc[_ILDIFF] = df_loc[_STATIC] - df_loc[_DYNAMIC]

        df_di_fill = df_di.astype(float).interpolate(method="linear")
        df_di0_fill = df_accum.astype(float).interpolate(method="linear")

        # plot
        plot_args = dict(
            ls="--", color=_c_line, markeredgecolor="k", legend=False, clip_on=False
        )
        z = 99
        ax_ilX_loc.fill_between(
            x=df_di_fill.index,
            y1=df_di_fill[_STATIC],
            y2=df_di_fill[_DYNAMIC],
            alpha=0.2,
            color=_c_line,
            zorder=-99 - radial,
            label=settings.ILdiff,
        )
        df_di.loc[loc_list, _STATIC].plot(
            ax=ax_ilX_loc, marker="v", ms=8, lw=1, zorder=z, **plot_args
        )
        df_di.loc[loc_list[1:-1], _DYNAMIC].plot(
            ax=ax_ilX_loc, marker="^", ms=8, lw=1, zorder=z, **plot_args
        )

        handles, labels = ax_ilX_loc.get_legend_handles_labels()
        # specific points to (re)plot for zooms
        zoom_list_ms = 1
        idx_start = [loc_list[0]] + list(loc_list_zoom) + [loc_list[1]]
        idx_end = [loc_list[-2]] + list(loc_list_end) + [loc_list[-1]]
        df_to_zoom_start = df_di.loc[idx_start, _DYNAMIC]
        df_to_zoom_end = df_di.loc[idx_end, _DYNAMIC]
        df_to_zoom_start.plot(
            ax=ax_ilX_loc,
            marker="^",
            ms=zoom_list_ms,
            lw=0.5,
            zorder=z - 5,
            label="dynamic_start",
            **plot_args,
        )
        df_to_zoom_end.plot(
            ax=ax_ilX_loc,
            marker="^",
            ms=zoom_list_ms,
            lw=0.5,
            zorder=z - 5,
            label="dynamic_end",
            **plot_args,
        )
        lines_zoom_start = []
        lines_zoom_end = []
        from matplotlib.lines import Line2D

        line: Line2D
        for line in ax_ilX_loc.get_lines():
            if line.get_label() == "dynamic_start":
                lines_zoom_start.append(line)
            elif line.get_label() == "dynamic_end":
                lines_zoom_end.append(line)
        ax_il0_loc = create_zoom(
            ax_ilX_loc,
            inset_size=1.5,
            lines=lines_zoom_start,
            loc="lower right",
            loc1="all",
            loc2=3,
            xlim=(0, 0.1),
            xticks=list(np.linspace(0, 0.1, 3)),
            yticks=3,
            ec=settings.cmap_dict["n_e"][radial][-3],
            inset_kwargs=dict(bbox_to_anchor=(-0.18, -0.075),),
            connector_kwargs="None",
            lw=1,
            ms=6,
            clip_on=False,
            zorder=10,
        )
        ax_il0_loc.tick_params(labelsize="x-small")
        ax_il0_loc.fill_between(
            x=df_di_fill.index,
            y1=df_di_fill[_STATIC],
            y2=df_di_fill[_DYNAMIC],
            alpha=0.2,
            color=_c_line,
            zorder=-99 - radial,
        )
        ax_il0_loc.set_xticks(loc_list_zoom, minor=True)
        ax_il0_loc.set_xticklabels((0.0, "", 0.1))
        ax_inset_end = create_zoom(
            ax_ilX_loc,
            inset_size=1.5,
            lines=lines_zoom_end,
            loc="lower left",
            loc1=1,
            loc2=4,
            xlim=(0.9, 1),
            xticks=list(np.linspace(0.9, 1.0, 3)),
            yticks=3,
            ec=settings.cmap_dict["n_e"][radial][-5],
            inset_kwargs=dict(bbox_to_anchor=(1.1, -0.075)),
            connector_kwargs="None",
            ms=6,
            clip_on=False,
            zorder=10,
        )
        ax_inset_end.tick_params(labelsize="x-small")
        ax_inset_end.fill_between(
            x=df_di_fill.index,
            y1=df_di_fill[_STATIC],
            y2=df_di_fill[_DYNAMIC],
            alpha=0.2,
            color=_c_line,
            zorder=-99 - radial,
            lw=diam,
        )
        ax_inset_end.set_xticks(loc_list_end, minor=True)
        ax_inset_end.set_xticklabels((0.9, "", 1.0))

        # legend on the right for static + dynamic cl IL
        ax_ilX_loc.legend(
            handles,
            labels,
            loc="upper right",
            bbox_to_anchor=(-0.1, 1.0),
            borderaxespad=0,
            ncol=1,
            fontsize="x-small",
            markerfirst=False,
            labelspacing=1.0,
            handletextpad=0.1,
            frameon=False,
        )

        if r != -1:
            # change heatmap morphology to be different units and/or color
            for key in plot_names:
                _df = plot_settings[key]
                _df["clim"]: List[float] = [
                    df_loc[key].values.min(),
                    df_loc[key].values.max(),
                ]
                if key == "relative":
                    _df["clim"] = None
                    continue
                elif key == "EGABA":
                    _df["clim"][0] = egaba
                elif key == _ILDIFF:
                    _df["clim"] = [0, round(_df["clim"][1])]
                elif shared_clim:
                    _df["clim"]: List[float] = [
                        df_loc[[_STATIC, _DYNAMIC]].values.min(),
                        df_loc[[_STATIC, _DYNAMIC]].values.max(),
                    ]
            for v, key in enumerate(plot_names):
                _df = plot_settings[key]
                for li, loc_point in enumerate(loc_points_morph):
                    logger.debug(f"\t\t\t {radial}:{key}:{loc_point}")
                    _ax = ax_morph_loc[r, v, li]
                    if key == "relative":
                        cvals = df_loc[_DYNAMIC, loc_point].iloc[1:-1]
                        clim = [
                            cvals.min(),
                            cvals.max(),
                        ]  # get the min/max values for this time point
                    else:
                        cvals = df_loc[key, loc_point].iloc[1:-1]
                        clim = _df["clim"]
                    recolor_shapeplot2d(_ax, cmap=_df["cmap"], cvals=cvals, clim=clim)
                # heatmap
                _ax_hm = _ax = ax_morph_loc[r, v, -1]
                if _df["clim"] is None:
                    norm = colors.Normalize(0, 1)
                else:
                    norm = colors.Normalize(*_df["clim"])
                if shared_clim and v == 1:
                    # dont re-do the colorbar
                    continue
                cb = fig.colorbar(
                    cm.ScalarMappable(norm=norm, cmap=_df["cmap"]),
                    cax=_ax_hm,
                    orientation="vertical",
                )
                cb.set_label(
                    _df["cb_label"],
                    fontsize="x-small",
                    rotation=0,
                    ha="left",
                    va="center_baseline",
                )
                cb.ax.tick_params(labelsize="xx-small")
                if shared_clim and v == 0:
                    from matplotlib.ticker import MaxNLocator

                    cb.ax.yaxis.set_major_locator(MaxNLocator(4))
                # if key == 'EGABA':
                #     cb.set_ticks([egaba, vinit, round(val['clim'][1])])
                if key == "relative":
                    cb.set_ticks((0, 1))
                    cb.set_ticklabels(("min", "max"))

                if not shared_clim:
                    cb.ax.annotate(
                        "max",
                        xy=(0.5, 1.0),
                        xycoords="axes fraction",
                        va="bottom",
                        ha="center",
                        fontsize="x-small",
                    )
                    cb.ax.annotate(
                        "min",
                        xy=(0.5, -0.02),
                        xycoords="axes fraction",
                        va="top",
                        ha="center",
                        fontsize="x-small",
                    )

        # labels for # branches
        for v, key in enumerate(plot_names):
            ax_morph_loc[r, v, 0].annotate(
                plot_settings[key]["label"],
                xy=(-0.3, 0.5),
                xycoords="axes fraction",
                ha="right",
                va="center",
                fontsize="small",
            )

        ax_ilX_loc.set(xlabel=settings.LOCATION_X_, xlim=(0, 1))
        ilX_ylim = ax_ilX_loc.get_ylim()
        il0_ylim = ax_il0_loc.get_ylim()
        ax_ilX_loc.set_xticks(np.arange(0, 1.2, 0.2), minor=False)
        ax_ilX_loc.set_xticks(np.arange(0.1, 1, 0.2), minor=True)
        adjust_spines(ax_ilX_loc, ["left", "bottom", "top"], 0)
        ax_ilX_loc.tick_params(
            axis="x", bottom=True, top=True, labelbottom=True, labeltop=False
        )
        ax_ilX_loc.set_ylabel(
            settings.ILd.replace("d", "d=i"), ha="center", va="bottom"
        )

        # mid_y = df_di[_DYNAMIC].loc[loc_list[-1]]/2 + df_di[_STATIC].loc[loc_list[-1]]/2
        # ax_ilX_loc.annotate(settings.ILdiff, xy=(loc_list[-1] + 0.007, mid_y), xytext=(20.0, 0.),
        #                     textcoords='offset points',
        #                     xycoords='data', annotation_clip=False, ha='center', va='center',
        #                     arrowprops=dict(arrowstyle='-[',
        #                                     # ArrowStyle.BarAB(widthA=0., widthB=1.0),
        #                                     ec=settings.COLOR.K))
        return df_di, df_loc, arrowprops

    for radial, _gs in zip(radial_list, [gs0, gs1]):
        df_di, df_il, ap = cl_loc(radial, _gs)

    letter_axes(ax_morph_loc[:, 0, 0], ha="right")

    if settings.SAVE_FIGURES:
        plot_save("output/figure_dynamic_loc.png", figs=[fig], close=False)
        plot_save("output/figure_dynamic_loc.pdf")
    else:
        import shared

        shared.show_n(1)
