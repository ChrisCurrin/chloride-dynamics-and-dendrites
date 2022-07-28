import copy
from typing import List

import numpy as np
import pandas as pd
from matplotlib import colors, cm
from matplotlib.axes import Axes
from matplotlib.cbook import flatten
from matplotlib.ticker import MaxNLocator

try:
    from utils import settings
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    # find the script's directory
    script_dir = Path(__file__).parent.parent.absolute()
    sys.path.insert(0, str(script_dir))
    from utils import settings

from inhib_level.math import accumulation_index, ghk, inv_nernst
from utils.plot_utils import (
    adjust_spines,
    copy_lines,
    recolor_shapeplot2d,
    letter_axes,
    plot_save,
    new_gridspec,
)


import logging

logger = logging.getLogger()


def figure_dynamic_il_time():
    """ How does dynamic Cl- affect IL

    """
    from matplotlib.gridspec import GridSpecFromSubplotSpec
    from main import run_inhib_level
    from shared import env_var

    logger.info("*" * 50 + "figure_dynamic_il_time" + "*" * 50)

    # Variables
    tm = 5
    tstop = 1000
    t_points = np.linspace(0, tstop, 5)
    t_points[0] = tm
    # t_points = [20, 100, 260, 580, 1000]  # close to when egaba reaches an integer (e.g. -64)

    truncate_kwargs = dict(minval=0.1, maxval=0.9, n=100)
    cmap_idx = 150  # index between 0 and 255
    X = sample_loc = 0.2
    radial_list = [1, 2, 4, 8]  # all populations
    radial_morph_list = [1, 4]  # morphologies to plot
    e_offset = sample_e = -5

    loc_list = [sample_loc]
    loc_list_str = " ".join(f"[{_loc}]" for _loc in loc_list)
    radial_list_str = " ".join(str(r) for r in radial_list)
    e_offset_list = [e_offset]
    e_offsets = " ".join(str(e) for e in e_offset_list)

    # Shorthands
    _STATIC = settings.STATIC_CHLORIDE_STR_ABBR
    _DYNAMIC = settings.DYNAMIC_CHLORIDE_STR_ABBR
    _ILDIFF = settings.ILdiff
    delta = settings.DELTA
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
    hm_space = 3
    hm_shrink = 3

    fig, gs = new_gridspec(
        2,
        3,
        figsize=(settings.PAGE_W_FULL, settings.PAGE_H_FULL_no_cap),
        grid_kwargs=dict(height_ratios=[4, 5], width_ratios=[5, 1, 5]),
    )
    gs.update(top=0.98, left=0.14, bottom=0.05, hspace=0.25, wspace=0.1)
    # create sub-gridspecs for sections of the figure
    gs0 = GridSpecFromSubplotSpec(gs_size, gs_size, subplot_spec=gs[0, 0])  # N = 1
    gs1 = GridSpecFromSubplotSpec(gs_size, gs_size, subplot_spec=gs[0, -1])  # N = 4
    gs_comb = GridSpecFromSubplotSpec(
        gs_size, gs_size, subplot_spec=gs[1, :]
    )  # combined plots

    # morph ax dimensions
    r_h = int(gs_size / len(plot_names))  # height of each morph
    r_p = int(hpad / len(plot_names))  # height pad
    col_w = int((gs_size - hm_space) / len(t_points))  # width of each morph over time

    # [var] vs time dimensions
    left_start = int(0.04 * gs_size)
    right_end = int(0.04 * gs_size)

    _height_r1 = 0.25  # percentage
    r1_c1s = 0
    r1_c1e = int(_height_r1 * gs_size)
    r1_c2s = r1_c1e + hpad // 2
    r1_c2e = r1_c2s + int(_height_r1 * gs_size)
    r1_c3s = r1_c2e + hpad

    ax_il0 = []
    ax_ilXl = []
    ax_eXl = []
    ax_e0 = ax_eXl
    single_pad = int(0.1 * gs_size)
    total_pad = single_pad * (len(radial_list) - 1)  # one less than number of plots
    width = (gs_size - total_pad) // len(radial_list)
    for r, radius in enumerate(radial_list):
        _ax_il0 = fig.add_subplot(
            gs_comb[
                :r1_c1e, (width + single_pad) * r : width * (r + 1) + single_pad * r
            ]
        )
        _ax_ilXl = _ax_il0.twinx()
        _ax_ilXl.set_zorder(-1)
        ax_il0.append(_ax_il0)
        ax_ilXl.append(_ax_ilXl)

        # egaba
        _ax_eXl = fig.add_subplot(
            gs_comb[
                r1_c2s:r1_c2e,
                (width + single_pad) * r : width * (r + 1) + single_pad * r,
            ]
        )
        ax_eXl.append(_ax_eXl)

    ax_accidx = fig.add_subplot(gs_comb[r1_c3s:, :], sharex=None)

    shared_ax_time = True
    ax_time = None

    ##########################
    # TIME SECTION
    ##########################

    # ax_t_n1_l2[0, 0].annotate('Ai', xy=(0.01, .95), xycoords='figure fraction', fontsize='xx-large')
    # ax_time.annotate('ii', xy=(0.01, .76), xycoords='figure fraction', fontsize='xx-large')

    def cl_time(N, gridspec, shared_clim_t=True, shared_clim_il=True, labels=True):
        nonlocal ax_time
        morph_space = int(gs_size * 0.6)
        h_pad = int(0.04 * gs_size)
        gs_morph_time = GridSpecFromSubplotSpec(
            len(plot_names),
            len(t_points) + 1,
            subplot_spec=gridspec[: morph_space - h_pad, :],
            width_ratios=[5] * 5 + [1],
            wspace=0.1,
            hspace=0.3,
        )

        ax_t_n1_l2 = np.empty(shape=(len(plot_names), len(t_points)), dtype=Axes)
        ax_t_n1_l2_hm = []
        for i, name in enumerate(plot_names):
            # time columns
            for t, t_point in enumerate(t_points):
                ax_t_n1_l2[i, t] = fig.add_subplot(gs_morph_time[i : i + 1, t : t + 1])
            # heatmap
            ax_t_n1_l2_hm.append(fig.add_subplot(gs_morph_time[i : i + 1, -1:]))
        ax_t_n1_l2_hm_shared = fig.add_subplot(gs_morph_time[:2, -1:])  # span two rows

        ax_time = fig.add_subplot(
            gridspec[morph_space:, int(col_w / 2) : -hm_space - int(col_w / 2)],
            sharey=shared_ax_time and ax_time,
            sharex=shared_ax_time and ax_time,
        )
        adjust_spines(ax_t_n1_l2, [], 0)

        plot_dict, sim_type, saved_args = run_inhib_level(
            f"--radial {N} --loc={sample_loc}"
            f" --e_offsets {e_offset} --synapse_dists=diffused_matched "
            f"--kcc2=C "
            " --plot_color_by=kcc2"
            f" --tstop={tstop} --tm={tm} --with-t-trace={sample_loc}"
            " --precise --sections=radial_dends_1 --plot_shape"
        )

        il_dict = saved_args["il_dict"]
        ecl_dict = saved_args["ecl_dict"]
        t_index = np.linspace(tm, tstop, int(tstop / tm))
        # dataframe for time points
        df_t = pd.DataFrame(
            index=il_dict["units"].loc["radial_dends_1"].index,
            columns=pd.MultiIndex.from_product([plot_names, t_points]),
        )
        # dataframe for continuous time (win tm steps)
        df_time = pd.DataFrame(index=t_index)

        ev = env_var()
        pcl, ecl, phco3, ehco3, egaba, clo, hco3i, hco3o, vinit = (
            ev["pcl"],
            ev["ecl"],
            ev["phco3"],
            ev["ehco3"],
            ev["egaba"],
            ev["clo"],
            ev["hco3i"],
            ev["hco3o"],
            ev["v_init"],
        )
        
        loc_idx = abs(sample_loc - df_t.index).argmin()

        highlight_cmap = settings.truncate_colormap(
            settings.n_branches_cmaps[N], **truncate_kwargs
        )
        hl_color = highlight_cmap(0)

        for plot_key, _df in il_dict.items():
            if plot_key == "units":
                continue
            n = int(plot_key[plot_key.index("n=") + 2 : plot_key.index("/")])
            e = float(plot_key[plot_key.index("e=") + 2 : plot_key.index("(")])
            cl = plot_key[plot_key.index("(") + 1 : plot_key.index(")")]
            if cl == settings.DELTA:
                indx = ecl_dict[plot_key]["radial_dends_1"].index
                
                # egaba_t = pcl * ecl_dict[plot_key]["radial_dends_1"] + phco3 * ehco3
                for t_point in t_points:
                    t_idx = abs(t_point - indx).argmin()
                    cli_t = inv_nernst(ecl_dict[plot_key]["radial_dends_1"].iloc[t_idx], ev["clo"])
                    df_t[("EGABA", t_point)] = ghk([clo, hco3o],[cli_t, hco3i], [pcl, phco3], [-1, -1])
            for t_point in t_points:
                df_t[(plot_map[cl], t_point)] = _df.loc["radial_dends_1", t_point]
            df_time[plot_map[cl]] = (
                _df.loc["radial_dends_1"].iloc[loc_idx].astype(float)
            )

        df_t[_ILDIFF] = df_t[_STATIC] - df_t[_DYNAMIC]

        # df_time.plot(ax=ax_time, style=['-.', '-'], color=[hl_color], legend=False)
        for (label, series), ls in zip(df_time.iteritems(), ["-.", "-"]):
            ax_time.plot(series, ls, color=hl_color)
        df_time[_ILDIFF] = df_time[_STATIC] - df_time[_DYNAMIC]

        # ax_time.fill_between(df_time[_STATIC].index, df_time[_STATIC], df_time[_DYNAMIC],
        #                      cmap=plot_settings[_ILDIFF]['cmap'], alpha=0.2, zorder=-99)

        # complex IL over time by creating a contour across the entire height
        # and then overlaying with white fill_between where IL static is above and IL dynamic is below
        ymax = np.max([df_time[_DYNAMIC], df_time[_STATIC]]) + 1
        ymin = np.min([0, np.min([df_time[_DYNAMIC], df_time[_STATIC]])])
        X, Z = np.meshgrid(df_time[_STATIC].index, df_time[_ILDIFF])
        Z = Z.T

        CS = ax_time.contourf(
            df_time[_ILDIFF].index,
            df_time[_DYNAMIC] * 1.2,
            Z,
            100,
            cmap=plot_settings[_ILDIFF]["cmap"],
            zorder=-100,
        )
        ax_time.fill_between(
            df_time[_STATIC].index,
            [ymax] * df_time[_STATIC].size,
            df_time[_STATIC],
            color="w",
        )
        ax_time.fill_between(
            df_time[_STATIC].index,
            df_time[_DYNAMIC],
            [ymin] * df_time[_DYNAMIC].size,
            color="w",
        )

        # ax_time.legend(loc=(0, 1), fontsize='small', frameon=False)
        ax_time.annotate(
            _STATIC,
            xy=(tstop * 1.01, df_time[_STATIC].iloc[-1]),
            xycoords="data",
            annotation_clip=False,
            fontsize="small",
            ha="left",
            va="center",
        )
        ax_time.annotate(
            _DYNAMIC,
            xy=(tstop * 1.01, df_time[_DYNAMIC].iloc[-1]),
            xycoords="data",
            annotation_clip=False,
            fontsize="small",
            ha="left",
            va="center",
        )
        ax_time.annotate(
            _ILDIFF,
            xy=(
                tstop - 10,
                np.min(df_time[_STATIC]) / 2 + np.min(df_time[_DYNAMIC]) / 2,
            ),
            ha="right",
            va="center_baseline",
        )

        annotations = None
        for ax_key, ax in plot_dict[0][2].items():
            if ax_key.startswith("SHAPE") and ax_key.endswith("AX"):
                n = int(ax_key[ax_key.index("n=") + 2 : ax_key.index("/")])
                dist = ax_key[ax_key.index("\n") : ax_key.index("x=")].strip()
                x = list(ax_key[ax_key.index("x=[") + 3 : ax_key.index("]")])
                e = ax_key[ax_key.index("e=") + 2 : ax_key.index("(")]
                cl = ax_key[ax_key.index("(") + 1 : ax_key.index(")")]
                underscore_idx = ax_key.rfind("_")
                annotation_key = ax_key[:underscore_idx] + "_ANNOTATIONS"
                annotations = plot_dict[0][2][annotation_key]
                for ax_t in flatten(ax_t_n1_l2):
                    lw = morph_lw + 2 if N == 1 else morph_lw
                    copy_lines(ax, ax_t, linewidth=lw)
                    for annotation in annotations:
                        ann = copy.deepcopy(annotation)
                        if cl != "-":
                            # subtle change color of synapse according to reversal over time
                            _c = annotation["arrowprops"]["facecolor"]
                            egaba_t_loc = df_t["EGABA", t_point].iloc[loc_idx]
                            _egaba_perc_increase = (egaba_t_loc - egaba) / (
                                vinit - egaba
                            )
                            ann["arrowprops"]["facecolor"] = settings.lighten_color(
                                _c, 1 - _egaba_perc_increase
                            )
                        ann["xytext"] = (ann["xytext"][0], lw)
                        ax_t.annotate(**ann)

        # change heatmap morphology to be different units and/or color
        for plot_key in plot_names:
            val = plot_settings[plot_key]
            if plot_key == "relative":
                val["clim"] = None
                continue
            val["clim"] = [
                df_t[plot_key].min().min(),
                df_t[plot_key].max().max(),
            ] if shared_clim_t else None
            if plot_key == "EGABA":
                val["clim"] = [egaba, vinit]
            elif plot_key == _ILDIFF:
                val["clim"][0] = 0
        if shared_clim_il:
            _clim: List[float] = [
                np.min(
                    [
                        plot_settings[_STATIC]["clim"][0],
                        plot_settings[_DYNAMIC]["clim"][0],
                    ]
                ),
                np.max(
                    [
                        plot_settings[_STATIC]["clim"][1],
                        plot_settings[_DYNAMIC]["clim"][1],
                    ]
                ),
            ]
            plot_settings[_STATIC]["clim"] = _clim
            plot_settings[_DYNAMIC]["clim"] = _clim

        for _pd, plot_key in enumerate(plot_names):
            val = plot_settings[plot_key]
            for _t, t_point in enumerate(t_points):
                _ax = ax_t_n1_l2[_pd, _t]
                if plot_key == "relative":
                    cvals = df_t[_DYNAMIC, t_point].iloc[1:-1]
                    clim = [
                        cvals.min(),
                        cvals.max(),
                    ]  # get the min/max values for this time point
                else:
                    clim = val["clim"]
                    cvals = df_t[plot_key, t_point].iloc[1:-1]
                recolor_shapeplot2d(_ax, cmap=val["cmap"], cvals=cvals, clim=clim)

                # add annotations (same for all)
                for annotation in annotations:
                    ann = copy.deepcopy(annotation)
                    if plot_key != _STATIC:
                        # subtle change color of synapse according to reversal over time
                        _c = annotation["arrowprops"]["facecolor"]
                        _clim = plot_settings["EGABA"]["clim"]
                        egaba_t_loc = df_t["EGABA", t_point].iloc[loc_idx]
                        _egaba_perc_increase = (egaba_t_loc - _clim[0]) / (
                            _clim[1] - _clim[0]
                        )
                        ann["arrowprops"]["facecolor"] = settings.lighten_color(
                            _c, 1 - _egaba_perc_increase
                        )
                    ann["xytext"] = (ann["xytext"][0], morph_lw)
                    _ax.annotate(**ann)

            # annotate rows
            if labels:
                ax_t_n1_l2[_pd, 0].annotate(
                    val["label"],
                    xy=(-0.1, 0.5),
                    xycoords="axes fraction",
                    fontsize="small",
                    va="center",
                    ha="right",
                )
            # add heatmap at end
            _ax_hm = ax_t_n1_l2_hm[_pd]
            if shared_clim_il and _pd <= 1:
                # use a shared heatmap
                adjust_spines(_ax_hm, [], 0)  # hide heatmap
                if _pd == 1:
                    continue  # done this already on 0
                else:
                    _ax_hm = ax_t_n1_l2_hm_shared
            elif not shared_clim_il and _pd == 0:
                adjust_spines(ax_t_n1_l2_hm_shared, [], 0)  # hide heatmap
            if val["clim"] is None:
                val["clim"] = [0, 1]
            
            norm = colors.Normalize(*val["clim"])
            cb = fig.colorbar(
                cm.ScalarMappable(norm=norm, cmap=val["cmap"]),
                cax=_ax_hm,
                orientation="vertical",
            )
            cb.set_label(
                val["cb_label"],
                fontsize="x-small",
                rotation=0,
                ha="left",
                va="center_baseline",
            )
            cb.ax.tick_params(labelsize="xx-small")

            # tweak ticks
            # if plot_key == 'EGABA':
            #     cb.set_ticks(np.round(np.linspace(egaba, vinit, round(vinit - egaba) + 1)))
            if plot_key == _STATIC or plot_key == _DYNAMIC:
                cb.ax.yaxis.set_major_locator(MaxNLocator(5))
            elif plot_key == settings.ILdiff:
                cb.ax.yaxis.set_major_locator(MaxNLocator(3))
            elif plot_key == "relative":
                cb.set_ticks(val["clim"])
                # cb.set_ticklabels(['min', 'max'])
                cb.ax.set_yticklabels(["min", "max"])

            if not shared_clim_t:
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
        adjust_spines(ax_time, ["left", "bottom", "top"], 0)
        ax_time.set_xlim(0, tstop)
        ax_time.set_xticks(np.append(0, t_points[1:]))
        ax_time.set_xticks(np.arange(0, t_points[-1], 50), minor=True)
        ax_time.tick_params(
            axis="x", bottom=True, top=True, labelbottom=True, labeltop=False
        )
        ax_time.set_yticks(np.arange(0, np.ceil(ymax), 2))
        ax_time.set_yticks(np.arange(0, np.ceil(ymax), 1), minor=True)
        ax_time.set(xlabel="Time (ms)", ylabel=f"${settings.IL}_{{d={sample_loc}}}$")
        ax_time.set_ylim(0, ymax)

        return (
            (df_t, df_time),
            (ax_t_n1_l2, ax_t_n1_l2_hm, ax_t_n1_l2_hm_shared, ax_time),
        )

    def accidx_v_time():
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
            f" --plot_group_by=e_offsets --with-t-trace {sample_loc} --plot_shape"
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
        df_il = pd.DataFrame(index=t_index, columns=columns)
        df_il_junction = pd.DataFrame(index=t_index, columns=columns)
        df_erev_loc = pd.DataFrame(columns=columns)
        df_erev_junction = pd.DataFrame(columns=columns)
        df_shape = pd.DataFrame(
            index=il_dict["units"].loc["radial_dends_1"].index, columns=radial_list
        )

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
            if cl == delta:
                df_shape[n] = _df.loc["radial_dends_1", tstop].values

        # EGABA
        ev = env_var()
        pcl, ecl, phco3, ehco3, egaba, clo, hco3i, hco3o, vinit = (
            ev["pcl"],
            ev["ecl"],
            ev["phco3"],
            ev["ehco3"],
            ev["egaba"],
            ev["clo"],
            ev["hco3i"],
            ev["hco3o"],
            ev["v_init"],
        )
        for key, _df in ecl_dict.items():
            if key == "units":
                continue
            n = int(key[key.index("n=") + 2 : key.index("/")])
            e = float(key[key.index("e=") + 2 : key.index("(")])
            cl = key[key.index("(") + 1 : key.index(")")]

            cli = inv_nernst(_df, ev["clo"])
            _df_erev = ghk([clo, hco3o],[cli, hco3i], [pcl, phco3], [-1, -1])
            # _df_erev = pcl * _df + phco3 * ehco3
            idx = abs(X - _df_erev["radial_dends_1"].columns).argmin()
            df_erev_loc[(n, e, cl)] = _df_erev["radial_dends_1"].iloc[:, idx]
            df_erev_junction[(n, e, cl)] = _df_erev["soma", 0.5]

        ax_morphs = {}
        for ax_key, ax in plot_dict[str(e_offset)][2].items():
            if ax_key.startswith("SHAPE") and ax_key.endswith("AX"):
                n = int(ax_key[ax_key.index("n=") + 2 : ax_key.index("/")])
                dist = ax_key[ax_key.index("\n") : ax_key.index("x=")].strip()
                x = list(ax_key[ax_key.index("x=[") + 3 : ax_key.index("]")])
                e = ax_key[ax_key.index("e=") + 2 : ax_key.index("(")]
                cl = ax_key[ax_key.index("(") + 1 : ax_key.index(")")]
                ax_morph = ax_morphs.get(n)
                if ax_morph is None:
                    # add shape plot
                    r = len(
                        ax_morphs
                    )  # may need to adjust how r is calculated if dict not ordered
                    ax_morphs[n] = fig.add_subplot(
                        gs_comb[
                            int(r1_c1e * 0.3) : int(r1_c1e * 0.95),
                            (width + single_pad) * r
                            + int(width * 0.3) : (width + single_pad) * r
                            + int(width * 0.95),
                        ]
                    )
                    copy_lines(ax, ax_morphs[n], linewidth=morph_lw)
                    adjust_spines(ax_morphs[n], [], 0)

                    cvals = df_shape[n].iloc[1:-1]
                    clim = [
                        cvals.min(),
                        cvals.max(),
                    ]  # get the min/max values for this time point
                    recolor_shapeplot2d(
                        ax_morphs[n],
                        cmap=plot_settings["relative"]["cmap"],
                        cvals=cvals,
                        clim=clim,
                    )

                    # annotate
                    underscore_idx = ax_key.rfind("_")
                    annotation_key = ax_key[:underscore_idx] + "_ANNOTATIONS"
                    annotations = plot_dict[str(e_offset)][2][annotation_key]
                    for annotation in annotations:
                        ann = copy.deepcopy(annotation)
                        # subtle change color of synapse according to reversal over time
                        _c = annotation["arrowprops"]["facecolor"]
                        _clim = plot_settings["EGABA"]["clim"]
                        egaba_t_loc = df_erev_loc[(n, e_offset, delta)].iloc[-1]
                        _egaba_perc_increase = (egaba_t_loc - _clim[0]) / (
                            _clim[1] - _clim[0]
                        )
                        # ann['arrowprops']['facecolor'] = settings.lighten_color(_c, 1 - _egaba_perc_increase)
                        ann["xytext"] = (ann["xytext"][0], morph_lw)
                        ax_morphs[n].annotate(**ann)

        # select from dataframes
        # plot EGABA and IL/IL_dff for sample_radial
        styles = dict(zorder=-99, alpha=1)
        egaba_style = dict(linestyle="--", lw=1, alpha=0.8, zorder=-99)

        sample_lines = []
        egaba_sample_lines = []
        egeba_cmap = settings.COLOR.truncate_colormap(
            settings.IL_config[settings.EGABA]["cmap"], 0.3, 0.8
        )
        egabas = np.linspace(egaba, vinit, round(vinit - egaba) + 1)

        def egaba_major_coords(radius):
            egabas_x = []
            egabas_y = []
            _line_colors = []
            df_egaba = df_erev_loc[radius].xs(delta, axis=1, level=1)
            for e in egabas:
                _df_ediff = abs(df_egaba - e)
                idxmin = _df_ediff.idxmin()
                if isinstance(idxmin, pd.Series):
                    idxmin = idxmin.mean()
                if _df_ediff.loc[idxmin].values < 0.01:
                    egabas_y.append(idxmin)
                egabas_x.append(idxmin)
                _line_colors.append(
                    egeba_cmap(int((e - egaba) * 255 / (vinit - egaba)))
                )
            return egabas_x, egabas_y, _line_colors

        for r, sample_radial in enumerate(radial_list):
            # EGABA and IL/ILdiff from just sample_radial
            select_erev = df_erev_loc[sample_radial].xs(delta, axis=1, level=1)
            select_erev_junc = df_erev_junction[sample_radial].xs(
                delta, axis=1, level=1
            )
            select_il = df_il[sample_radial].xs(delta, axis=1, level=1)
            select_il_junc = df_il_junction[sample_radial].xs(delta, axis=1, level=1)
            df_il_diff = (
                df_il[sample_radial, sample_e, "-"]
                - df_il[sample_radial, sample_e, delta]
            )
            df_il_diff_junction = (
                df_il_junction[sample_radial, sample_e, "-"]
                - df_il_junction[sample_radial, sample_e, delta]
            )

            cmap = settings.COLOR.truncate_colormap(
                settings.cmap_dict["num_synapses"][sample_radial], 0.3, 0.8
            )
            tmpEX = select_erev.plot(
                ax=ax_eXl[r], cmap=cmap, legend=False, ls="-", **styles
            )
            c = cmap(128)
            tmpE0 = select_erev_junc.plot(
                ax=ax_e0[r], c=c, legend=False, ls=":", **styles
            )
            tmpX = df_il_diff.plot(
                ax=ax_ilXl[r], cmap=cmap, legend=False, ls="-", **styles
            )
            tmp0 = df_il_diff_junction.plot(
                ax=ax_il0[r], c=c, legend=False, ls=":", **styles
            )

            # leg = ax_ilXl[r].legend([sample_radial], frameon=False, loc="lower right")
            # ax_ilXl[r].annotate(sample_radial, xy=(0.5, 0.2), xycoords="axes fraction", color=cmap(1), ha="center")
            ax_ilXl[r].set_title(
                sample_radial, color=cmap(1), fontsize="large", va="bottom"
            )
            ax_il0[r].spines["right"].set_visible(True)
            ax_il0[r].spines["right"].set_linestyle(":")
            # ax_il0[r].spines['right'].set_color(cmap(100))
            # ax_il0[r].spines['bottom'].set_color(cmap(1))
            # ax_ilXl[r].spines['left'].set_color(cmap(1))
            # ax_ilXl[r].spines['bottom'].set_color(cmap(1))
            # change tick line colors
            # ax_ilXl[r].tick_params(axis='y', which='major', colors=cmap(1))
            # ax_il0[r].tick_params(axis='y', which='major', colors=cmap(1))

            # change label size
            # ax_il0[r].tick_params(axis='both', which='major', labelsize='x-small')
            # ax_ilXl[r].tick_params(axis='both', which='major', labelsize='x-small')
            e_x, e_y, e_colors = egaba_major_coords(sample_radial)
            e_lines = ax_eXl[r].hlines(
                egabas, xmin=0, xmax=e_x, colors=e_colors, **egaba_style
            )
            ax_eXl[r].vlines(
                e_y, ymin=egaba, ymax=egabas[:len(e_y)], colors=e_colors, **egaba_style
            )
            if r == 0:
                sample_lines = list(flatten([tmpX.get_lines(), tmp0.get_lines()]))
                egaba_sample_lines = list(
                    flatten([tmpEX.get_lines(), tmpE0.get_lines(), e_lines])
                )

            # accumulation for all radial_list
        select_acc = df_accum.xs(sample_e, axis=1, level=1).xs(delta, axis=1, level=1)
        line_colors = []
        for n_dend in radial_list:
            cmap = settings.COLOR.truncate_colormap(
                settings.cmap_dict["num_synapses"][n_dend], 0.3, 0.8
            )
            line_colors.append(cmap(1))

        # plot accumulation for all radial_list
        select_acc.plot(
            ax=ax_accidx, color=line_colors, lw=4,
        )

        # LEGENDS for AccIdx plots
        ax_ilXl[0].legend(
            sample_lines,
            ["synapse", "junction"],
            ncol=2,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.9),
            fontsize="x-small",
            frameon=False,
        )
        ax_eXl[-1].legend(
            egaba_sample_lines[-3:],
            ["synapse", "junction", f"{settings.NABLA}(t)"],
            ncol=1,
            loc="upper left",
            bbox_to_anchor=(0.9, 0.9),
            borderpad=0,
            fontsize="x-small",
            title="EGABA",
            title_fontsize="small",
            frameon=False,
        )
        ax_accidx.legend(
            loc="lower left",
            bbox_to_anchor=(1, 0.0),
            fontsize="small",
            title="Branches",
            title_fontsize="small",
            frameon=False,
        )

        # labels and legends
        from matplotlib.ticker import AutoMinorLocator

        adjust_spines(ax_ilXl, ["left", "bottom"], 0)
        adjust_spines(ax_il0, ["right"], 0)
        for _ax in ax_il0:
            _ax.set_ylim(0)
            _ax.set_xlim(0, tstop)
            _ax.set_xticks(np.arange(0, t_points[-1], 50), minor=True)
            _ax.yaxis.set_major_locator(MaxNLocator(3, integer=True))

        for _ax in ax_ilXl:
            _ax.set_ylim(0)
            _ax.set_yticklabels(_ax.get_yticks())
            _ax.set_xlim(0, tstop)
            _ax.set_xticklabels([])
            _ax.set_xticks(np.arange(0, t_points[-1], 50), minor=True)
            _ax.yaxis.set_major_locator(MaxNLocator(3, integer=True))

        for _ax in ax_eXl:
            _ax.set_yticks(np.arange(egaba, vinit + 1, 1, dtype=int))
            _ax.yaxis.set_minor_locator(AutoMinorLocator())
            _ax.set_xticks(np.arange(0, t_points[-1], 50), minor=True)
            _ax.set_ylim(egaba)
            _ax.set_xlim(0, tstop)

        ax_il0[-1].set_ylabel(
            f"${settings.ILdiff.replace('$', '')}_{{\mathit{{junction}}}}$"
        )
        ax_ilXl[0].set_ylabel(
            f"${settings.ILdiff.replace('$', '')}_{{\mathit{{synapse}}}}$"
        )

        ax_accidx.set_xlim(0, tstop)

        # pandas plotting seems to hide x/y tick labels
        adjust_spines(ax_accidx, ["left", "bottom"], 0)
        ax_accidx.xaxis.label.set_visible(True)
        ax_accidx.set_xticks(np.arange(0, t_points[-1], 50), minor=True)

        for p in plots:
            adjust_spines(p[-1], ["left", "bottom", "top"], 0)  # -1 is ax_time
            p[-1].tick_params(
                axis="x", bottom=True, top=True, labelbottom=True, labeltop=False
            )
            p[-1].xaxis.label.set_visible(True)
            p[-1].yaxis.label.set_visible(True)

        ax_eXl[0].set_ylabel(f"{settings.EGABA}\n({settings.mV})")
        # ax_e0.set_ylabel(f"{settings.EGABA}\n({settings.mV})")
        ax_accidx.set(xlabel="Time (ms)", ylabel=f"{settings.ACC_IDX_FULL}")

    data = []
    plots = []
    for N, gridspec in zip(radial_morph_list, [gs0, gs1]):
        _data, _plots = cl_time(N, gridspec, labels=(N == radial_morph_list[0]))
        data.append(data)
        plots.append(_plots)
    accidx_v_time()

    letter_axes(
        plots[0][0][0][0],
        subscript="A",
        xy=(0.0, 0.97),
        xycoords="figure fraction",
        ha="left",
    )
    letter_axes(
        plots[1][0][0][0],
        start=2,
        subscript="A",
        repeat_subscript=True,
        xy=(0.52, 0.97),
        xycoords="figure fraction",
    )
    letter_axes(
        ax_ilXl[0], start="B", xy=(0.0, 0.75), xycoords="figure fraction", ha="left"
    )
    letter_axes(
        ax_ilXl[0], start="C", xy=(0.0, 0.53), xycoords="figure fraction", ha="left"
    )
    letter_axes(
        ax_eXl[0], start="D", xy=(0.0, 0.37), xycoords="figure fraction", ha="left"
    )
    letter_axes(
        ax_accidx, start="E", xy=(0.0, 0.24), xycoords="figure fraction", ha="left"
    )

    if settings.SAVE_FIGURES:
        plot_save("output/figure_dynamic_time.png", figs=[fig], close=False)
        plot_save("output/figure_dynamic_time.svg", figs=[fig], close=False)
        plot_save("output/figure_dynamic_time.pdf")
    else:
        import shared

        shared.show_n(1)


if __name__ == "__main__":
    # parse arguments
    import argparse
    from shared import INIT

    parser = argparse.ArgumentParser(description="")
    # add verbose
    parser.add_argument("-v", "--verbose", action="store_true")
    # add very verbose
    parser.add_argument("-vv", "--very_verbose", action="store_true")
    args = parser.parse_args()

    if args.very_verbose:
        INIT(reinit=True, log_level=logging.DEBUG)
    elif args.verbose:
        INIT(reinit=True, log_level=logging.INFO)
    else:
        INIT(reinit=True, log_level=logging.WARNING)

    # run
    figure_dynamic_il_time()