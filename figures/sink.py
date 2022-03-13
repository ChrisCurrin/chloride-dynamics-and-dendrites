import copy
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors, cm
from matplotlib.axes import Axes
from matplotlib.cbook import flatten
from matplotlib.colors import ListedColormap
from tqdm import tqdm

from utils import settings
from inhib_level.math import accumulation_index, lambda_d
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


def figure_sink():
    """ How a chloride sink affects optimal location


    """
    from matplotlib.gridspec import GridSpecFromSubplotSpec
    from main import run_inhib_level
    from shared import env_var
    import seaborn as sns

    logger.info(
        """
    ####################################################################################################################
    # CHLORIDE SINK and OPTIMAL LOCATION
    ####################################################################################################################
    """
    )
    logger.info("*" * 50 + "figure_sink" + "*" * 50)
    fig_settings = {"heatmap_location": "right"}

    fig, ax_dict = plt.subplot_mosaic([
        ['sink_a', 'IL_v_X', 'opt_X_vs_diam'],
        ['sink_b', 'IL_v_X', 'opt_X_vs_diam'],
        ['sink_c', 'IL_v_X', 'opt_X_vs_diam']],
        figsize=(settings.PAGE_W_FULL, settings.PAGE_H_FULL),
        )

    
    letter_axes(ax_dict["sink_a"], subscript="A", xy=(0.0, 1.1), ha="right")
    letter_axes(ax_dict["IL_v_X"], "B", xy=(-0.2, 1.0), ha="right", va="bottom")
    letter_axes(ax_dict["opt_X_vs_diam"], "C", xy=(-0.2, 1.0), ha="right")
 
    # local vars common between nested methods
    tstop = 500
    tm = 10
    e_offset = -5
    sample_N = 4
    IL_measure = "IL"
    plot_names = ["0", "i"]
    markers = ["o", "^"]
    measure_markers = dict(zip(plot_names, markers))


    def max_inhib_level(radials_diff=(2, 4, 6, 8), diams=(1., 0.5, 1.5, 2.), constant_L=True, kcc2="Y"):
        logger.info(
            "#" * 25
            + "\nCalculate and plot maximum IL for multiple numbers of branches\n"
            + "#" * 25
        )

        # default
        Rm = 20000
        Ra = 100
        L = lambda_d(diams[0], Rm, Ra)
        if constant_L:
            lengths = [L] * len(diams)
        else:
            other_lengths = [lambda_d(d, Rm, Ra) for d in diams[1:]]
            lengths = sorted([L] + other_lengths)

        dimensions = tuple(zip(diams, lengths))


        locs_diff = np.round(np.append(0.001, np.arange(0.01, 0.21, 0.01)), 5)
        loc_list = [0.0001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.9999]
        loc_list = np.round(loc_list, 6)
        full_loc_list = sorted(set(np.append(loc_list, locs_diff)))
        radials_diff_str = " ".join(str(r) for r in radials_diff)
        # line colors
        cs = [settings.cmap_dict["n_e"][n]["line"] for n in radials_diff]

        # check if data is already calculated
        file_name = "temp/sink_data.h5"
        egaba_file_name = "temp/sink_egaba_data.h5"

        # import os
        # if os.path.isfile(file_name):
        #     logger.info("Loading data from {}".format(file_name))
        #     df_il_r = pd.read_hdf(file_name, "data")
        #     egaba_dict = pd.read_hdf(egaba_file_name).to_dict(orient="records")
            
        df_il_r = pd.DataFrame(
            index=full_loc_list,
            columns=pd.MultiIndex.from_product(
                [["sink_i", "sink_AccIdx", "AccIdx", "i", "0"], diams, lengths, radials_diff], names=["Measure", "Diameter", "Length", "Branches"]
            ),
        ).sort_index(axis="columns")

        egaba_dict = {}

        logging.getLogger("inhib_level.plot").setLevel(logging.ERROR) # suppress warnings about plotting

        pbar = tqdm(reversed(list(itertools.product(radials_diff, full_loc_list, dimensions))))
        for num_branches, loc, dim in pbar:
            diam, l = dim
            sink_data = "{" + f"diam:{diam},l:{l}" + "}"
            txt = f"num_branches={num_branches} loc={loc} sink_data={sink_data}"
            logger.info(txt)
            pbar.set_description(txt)

            main_args = (
                f'--radial {num_branches} --loc={loc} '
                f"--e_offsets {e_offset} --synapse_dists=diffused_matched --kcc2={kcc2} "
                f'--plot_group_by=False '
                # f"--plot_group_by=num_dendrites --plot_color_by=num_synapses "
                f"--tstop={tstop} --tm={tm} --sink={sink_data} "
                f"--precise --sections radial_dends_1 sink_1 --nseg=267 "
                # f"--plot_shape"
            )
            plot_dict, sim_type, saved_args = run_inhib_level(main_args)
            continue
            il_dict = saved_args["il_dict"]
            ecl_dict = saved_args["ecl_dict"]
            index = il_dict["units"].index
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
                    df_il_r.loc[loc, ("i", diam, l, n)] = _df.loc[("radial_dends_1", tstop)].iloc[loc_idx]
                    df_il_r.loc[loc, ("sink_i", diam, l, n)] = _df.loc[("sink_1", tstop)].iloc[loc_idx]
                    df_il_r.loc[loc, ("0", diam, l, n)] = _df.loc[("soma", tstop)].iloc[0]
                    df_il_r.loc[loc, ("AccIdx", diam, l, n)] = (
                        df_il_r.loc[loc, ("0", diam, l, n)].values / df_il_r.loc[loc, ("i", diam, l, n)].values
                        )
                    df_il_r.loc[loc, ("sink_AccIdx", diam, l, n)] = (
                        df_il_r.loc[loc, ("0", diam, l, n)].values / df_il_r.loc[loc, ("sink_i", diam, l, n)].values
                        )
                    
                    ev = env_var()
                    pcl, ecl, phco3, ehco3, egaba, vinit = (
                        ev["pcl"],
                        ev["ecl"],
                        ev["phco3"],
                        ev["ehco3"],
                        ev["egaba"],
                        ev["v_init"],
                    )
                    egaba_dict[(n, diam, l, "radial", loc)] = (
                        pcl * ecl_dict[key]["radial_dends_1"].iloc[-1] + phco3 * ehco3
                    )
                    egaba_dict[(n, diam, l, "sink", loc)] = (
                        pcl * ecl_dict[key]["sink_1"].iloc[-1] + phco3 * ehco3
                    )
        # save data
        logger.info("Saving data to {}".format(file_name))
        df_il_r.to_hdf(file_name, "data")
        pd.DataFrame.from_dict(egaba_dict).to_hdf(egaba_file_name, "data")
        
        df_il_r.index = pd.Index(np.round(df_il_r.index, 2), name=settings.LOCATION_X_)

        max_il = {r: (-1, None, -1) for r in radials_diff}
        df_i = df_il_r["i"]
        df_i.name = "i"
        df_0 = df_il_r["0"]
        df_0.name = "0"

        _df_list = [
            _df
            for _df in [df_0, df_i,]
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

    max_inhib_level()

    if settings.SAVE_FIGURES:
        plot_save("output/sink.png", figs=[fig], close=False)
        plot_save("output/sink.pdf")
    else:
        import shared

        shared.show_n(1)
