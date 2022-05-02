import copy
import itertools
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors, cm
from matplotlib.axes import Axes
from matplotlib.cbook import flatten
from matplotlib.colors import ListedColormap
from tqdm import tqdm

import matplotlib

from utils.settings import IL, LOCATION_X_
matplotlib.use('pdf')

try:
    from utils import settings
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    # find the script's directory
    script_dir = Path(__file__).parent.parent.absolute()
    sys.path.insert(0, str(script_dir))
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


def figure_sink(**kwargs):
    """ How a chloride sink affects optimal location


    """
    from matplotlib.gridspec import GridSpecFromSubplotSpec
    from main import run_inhib_level
    from shared import env_var
    import seaborn as sns

    constant_L = kwargs.pop("constant_L", [True, False])

    if not np.iterable(constant_L):
        constant_L = [constant_L]

    logger.info(
        """
    ####################################################################################################################
    # CHLORIDE SINK and OPTIMAL LOCATION
    ####################################################################################################################
    """
    )
    logger.info("*" * 50 + "figure_sink" + "*" * 50)

    fig, ax_dict = plt.subplot_mosaic([
        ['sink_diam_2.0', 'sink_diam_0.5', 'IL_v_X', 'opt_X_vs_diam'],
        ['colorbar', 'colorbar', '.', 'opt_X_vs_diam'],
        ['sink_diam_2.0_1X', 'sink_diam_0.5_1X', 'IL_v_X_1X', 'opt_X_vs_diam'],
        ],
        figsize=(settings.PAGE_W_FULL, settings.PAGE_H_FULL/3),
        gridspec_kw={"width_ratios": [0.3, 0.3, 0.75, 1], "height_ratios":[1, 0.1, 1], 
                    "hspace": 0.4, "wspace": 0.5},
        )

    ax_diam_list = [ax for key, ax in ax_dict.items() if "sink_diam" in key]
    letter_axes(ax_diam_list, subscript="A", xy=(0.0, 1.), ha="right")
    letter_axes(ax_dict["IL_v_X"], "B", xy=(-0.3, 1.0), ha="right", va="bottom")
    letter_axes(ax_dict["opt_X_vs_diam"], "C", xy=(-0.4, 1.0), ha="right")
 
    # local vars common between nested methods
    tstop = 500
    tm = 10
    e_offset = -5
    sample_N = 4
    IL_measure = "IL"
    plot_names = ["0", "i"]
    markers = ["o", "^"]
    measure_markers = dict(zip(plot_names, markers))


    def max_inhib_level(radials_diff=(2, 4, 6, 8), diams=(1., 0.5, 1.5, 2.), constant_L=True, kcc2="Y", is_example=False):
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
            lengths = [L] + other_lengths

        dimensions = tuple(zip(diams, lengths))


        locs_diff = np.round(np.append(0.001, np.arange(0.01, 0.21, 0.01)), 5)
        loc_list = [0.0001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.9999]
        loc_list = np.round(loc_list, 6)
        full_loc_list = sorted(set(np.append(loc_list, locs_diff)))
        if is_example:
            full_loc_list = [0.2]
        radials_diff_str = " ".join(str(r) for r in radials_diff)
        # line colors
        cs = [settings.cmap_dict["n_e"][n]["line"] for n in radials_diff]

        # check if data is already calculated
        file_name = f"temp/sink_diam_{diams}_radials_diff_{radials_diff}_constant_L_{constant_L}_kcc2_{kcc2}".replace(".","dt")+".h5"
        egaba_file_name = file_name.replace(".h5", "_egaba.h5")
        plot_dict = {}
        il_dicts = {}

        import os
        if not os.path.isfile(file_name) or is_example:
            df_il_r = pd.DataFrame(
                index=full_loc_list,
                columns=pd.MultiIndex.from_product(
                    [["sink_i", "sink_AccIdx", "AccIdx", "i", "0"], diams, sorted(set(lengths)), radials_diff], names=["Measure", "Diameter", "Length", "Branches"]
                ),
            ).sort_index(axis="columns")

            egaba_dict = {}
            logging.getLogger("inhib_level.plot").setLevel(logging.ERROR) # suppress warnings about plotting

            pbar = tqdm(sorted(itertools.product(radials_diff, full_loc_list, dimensions), reverse=False))
            for num_branches, loc, dim in pbar:
                diam, l = dim
                sink_data = "{" + f"diam:{diam},l:{l}" + "}"
                txt = f"num_branches={num_branches} loc={loc} sink_data={sink_data}"
                logger.info(txt)
                pbar.set_description(txt)

                main_args = (
                    f'--radial {num_branches} --loc={loc} '
                    f"--e_offsets {e_offset} --synapse_dists=diffused_matched --kcc2={kcc2} "
                    # f'--plot_group_by=False '
                    f"--plot_group_by=num_dendrites --plot_color_by=num_synapses "
                    f"--tstop={tstop} --tm={tm} --sink={sink_data} "
                    f"--precise --sections radial_dends_1 sink_1 --nseg=267 "
                )
                if is_example:
                    main_args += f"--plot_shape"

                run_plot_dict, sim_type, saved_args = run_inhib_level(main_args)
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
                            df_il_r.loc[loc, ("0", diam, l, n)] / df_il_r.loc[loc, ("i", diam, l, n)]
                            )
                        df_il_r.loc[loc, ("sink_AccIdx", diam, l, n)] = (
                            df_il_r.loc[loc, ("0", diam, l, n)] / df_il_r.loc[loc, ("sink_i", diam, l, n)]
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
                if is_example:
                    plot_dict[(num_branches, diam)] = run_plot_dict[str(num_branches)]
                    # join il_dicts with il_dict
                    _dfs = [v for k, v in il_dict.items() if k != "units"]
                    assert len(_dfs) == 1, f"len(_dfs) = {len(_dfs)}"
                    il_dicts[(num_branches, loc, diam, constant_L)] = _dfs[0]
                else:
                    # close figures if not needed
                    for key, plots in run_plot_dict.items():
                        fig, *axes = plots
                        plt.close(fig)
            # save data
            logger.info("Saving data to {}".format(file_name))
            # suppress PerformanceWarning in context manager
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
                df_il_r.to_hdf(file_name, "data")
                pd.DataFrame.from_dict(egaba_dict).to_hdf(egaba_file_name, "data")
        
        logger.info("Loading data from {}".format(file_name))
        df_il_r = pd.read_hdf(file_name, "data")
        egaba_dict = pd.read_hdf(egaba_file_name).to_dict()
        
        df_il_r.index = pd.Index(np.round(df_il_r.index, 2), name=settings.LOCATION_X_)

        return df_il_r, egaba_dict, plot_dict, il_dicts
    # run examples
    arrowprops = None

    recolor_shape_dict = {}
    il_min, il_max = np.inf, -np.inf
    for i, c_L in enumerate(constant_L):
        
        df_il_r, egaba_dict, plot_dict, il_dicts = max_inhib_level(radials_diff=(4,), diams=(1., 0.5, 2.), constant_L=c_L, kcc2="Y", is_example=True)

        # plot constant_L
        for d in [2., 0.5]:
            ax_shape = ax_dict[f"sink_diam_{d:.1f}"] if c_L else ax_dict[f"sink_diam_{d:.1f}_1X"]
                
            shape_plots = {ax_key: _ax for ax_key, _ax in plot_dict[(4, d)][2].items() if ax_key.startswith("SHAPE")}
            shape_plot = [_ax for ax_key, _ax in shape_plots.items() if ax_key.endswith("_AX")][0]
            annotations = [_ax for ax_key, _ax in shape_plots.items() if ax_key.endswith("_ANNOTATIONS")][0]

            lines = copy_lines(shape_plot, ax_shape, rel_lw=2, alpha=0.01, clip_on=False)
            ax_shape.set_title(f"{d:.1f} $\mu$m", fontsize='small', va="top")
            _df = il_dicts[(4, 0.2, d, c_L)]
            _il_radial = _df.loc[("radial_dends_1", tstop)].iloc[1:-1]
            _il_sink = _df.loc[("sink_1", tstop)].iloc[1:-1]
            il_min = min(il_min, _il_radial.values.min(), _il_sink.values.min())
            il_max = max(il_max, _il_radial.values.max(), _il_sink.values.max())
            
            recolor_shape_dict[(d, c_L)] = {
                "lines": lines,
                "cvals": np.array(list(_il_radial)*4 + list(_il_sink)),
            }
            # plot default shape
            shape_plots = {ax_key: _ax for ax_key, _ax in plot_dict[(4, 1.0)][2].items() if ax_key.startswith("SHAPE")}
            shape_plot = [_ax for ax_key, _ax in shape_plots.items() if ax_key.endswith("_AX")][0]
            annotations = [_ax for ax_key, _ax in shape_plots.items() if ax_key.endswith("_ANNOTATIONS")][0]
            copy_lines(shape_plot, ax_shape, rel_lw=2, alpha=0.01, color='k', clip_on=False, zorder=-100000000)

            # add annotations
            for annotation in annotations:
                if arrowprops is None:
                    arrowprops = annotation["arrowprops"]
                ann = copy.deepcopy(annotation)
                ann["xytext"] = (0, 4)
                ax_shape.annotate(**ann)

    # change heatmap morphology to be different units and/or color
    cmap = settings.IL_config[settings.IL]["cmap"]
    clim = (il_min, il_max)
    for (d, c_L), lines_cvals_dict in recolor_shape_dict.items():
        ax_shape = ax_dict[f"sink_diam_{d:.1f}"] if c_L else ax_dict[f"sink_diam_{d:.1f}_1X"]
        lines = lines_cvals_dict["lines"]
        cvals = lines_cvals_dict["cvals"]
        recolor_shapeplot2d(lines, cmap=cmap, cvals=cvals, clim=clim)
    
    # heatmap
    norm = colors.Normalize(*clim)
    cb = fig.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax_dict["colorbar"],
        orientation="horizontal",
    )
    cb.set_label(
        settings.IL_config[settings.IL]["cb_label"],
        fontsize="x-small",
        rotation=0,
        ha="center",
        va="center_baseline",
    )
    # cb.set_ticks((0, 1))
    # cb.set_ticklabels((il_min, il_max), fontsize="xx-small")
    # cb.ax.tick_params(labelsize="xx-small")

    # set all shape plots to have the same x and y lims
    xlim = ax_diam_list[0].get_xlim()
    ylim = ax_diam_list[0].get_ylim()
    # get ax that have 'sink_diam' in their key
    # get max xlim and ylim
    for ax in ax_diam_list:
        xlim = (min(xlim[0], ax.get_xlim()[0]), max(xlim[1], ax.get_xlim()[1]))
        ylim = (min(ylim[0], ax.get_ylim()[0]), max(ylim[1], ax.get_ylim()[1]))
    # set all xlim and ylim
    for ax in ax_diam_list:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    # run full range
    radials_diff = kwargs.get("radials_diff", (2, 4, 6, 8))
    df_max_all = pd.DataFrame()

    for i, c_L in enumerate(constant_L):
        df_il_r, egaba_dict, plot_dict, il_dicts = max_inhib_level(constant_L=c_L, **kwargs)
        
        ax_il = ax_dict["IL_v_X"] if c_L else ax_dict["IL_v_X_1X"]

        df_plot = df_il_r.unstack().reset_index().rename(columns={0: IL})
        df_plot[IL] = df_plot[IL].astype(float)

        df_max = df_il_r.copy()
        df_max.iloc[:, :] = np.nan  # set all to nan

        max_idxs = df_il_r == df_il_r.max(axis=0)
        df_max[max_idxs] = df_il_r[max_idxs]  # assign only max values

        df_max = df_max.unstack().reset_index().rename(columns={0: IL})
        df_max[IL] = df_max[IL].astype(float)
        df_max = df_max.dropna() # drop rows with nan values
        df_max["1X"] = c_L
        df_max_all = pd.concat([df_max_all, df_max], ignore_index=True)

        # plot IL
        filtered_df = df_plot[(df_plot["Measure"].isin(["0"])) & (df_plot["Branches"]==4)]
        sns.lineplot(
            data=filtered_df, 
            x=LOCATION_X_,
            y=IL,
            # hue="Diameter",
            # palette=settings.n_branches_cmaps[4],
            color=settings.n_branches_cmap[4],
            size="Diameter",
            style='Diameter',
            style_order=[1.0] + sorted(set(df_plot["Diameter"].unique()) - {1.0}),
            ax=ax_il,
            legend=(i==0),
        )
        # plot max IL points bigger
        max_df = filtered_df.loc[filtered_df.groupby("Diameter")["IL"].idxmax()]
        sns.lineplot(
            data=max_df, 
            x=LOCATION_X_,
            y=IL,
            # hue="Diameter",
            # palette=settings.n_branches_cmaps[4],
            color=settings.n_branches_cmap[4],
            style="Diameter",
            legend=False,
            mec='k',
            marker=measure_markers['0'],
            ms=8,
            mew=1,
            linestyle='None',
            ax=ax_il,
            clip_on=False,
        )
        ax_il.set_ylim(0)
        ax_il.set_xlim(0, 0.2)
        ax_il.set_ylabel(f"${settings.IL}_{{0}}$")
        ax_il.set_xticks(np.arange(0, 0.2, 0.01), minor=True)
        

    io_df = df_max_all[df_max_all["Measure"].isin(["i", "0"])]
    df_max_io = io_df.loc[io_df.groupby(["1X", "Branches", "Diameter"])["IL"].idxmax()]
    sns.lineplot(
        data=df_max_io[df_max_io["1X"]==False],
        x="Diameter",
        y=LOCATION_X_,
        hue="Branches",
        palette=settings.n_branches_cmap,
        ax=ax_dict["opt_X_vs_diam"]
    )
    sns.lineplot(
        data=df_max_io[df_max_io["1X"]==False],
        x="Diameter",
        y=LOCATION_X_,
        hue="Branches",
        palette=settings.n_branches_cmap,
        style="Measure",
        markers=measure_markers,
        mec="k",
        ms=8,
        alpha=0.5,
        ax=ax_dict["opt_X_vs_diam"]
    )


    # clean up
    ax_dict["opt_X_vs_diam"].set(xlabel="Diameter of sink ($\\mu$m)", ylabel="Inhibitory synapse location that maximizes IL\n$argmax_{IL} i$ (X)")
    diams = kwargs.get("diams", [1., 0.5, 1.5, 2.])
    ax_dict["opt_X_vs_diam"].set_xticks(sorted(diams))

    # add * to each string value in radials_diff list
    star_radials_diff = [f"{r}*" for r in radials_diff]

    legend = ax_dict["opt_X_vs_diam"].legend(star_radials_diff, frameon=False, fontsize='small', loc=(0, 1), ncol=len(star_radials_diff))
    ax_dict["opt_X_vs_diam"].set_title("Branches\n*with 1 sink", fontsize="small")
    ax_dict["IL_v_X"].set(xlabel="", xticklabels=[], title="4 branches 1 sink")

    # add legend for markers in markers using custom Line2D objects
    handles = [plt.Line2D([0], [0], linestyle='None', marker=m, color='k', label=l, markersize=8, markeredgewidth=1) for l, m in measure_markers.items()]
    ax_dict["opt_X_vs_diam"].legend(handles=handles, frameon=False, fontsize='small')
    # re-add other legend
    ax_dict["opt_X_vs_diam"].add_artist(legend)


    for ax in ax_diam_list:
        # remove axes borders
        sns.despine(ax=ax, left=True, bottom=True)
        ax.set(xticks=[], yticks=[])

    if settings.SAVE_FIGURES:
        plot_save("output/sink.png", figs=[fig], close=False, bbox_inches="tight")
        plot_save("output/sink.svg", figs=[fig], close=False, bbox_inches="tight")
        # plot_save("output/sink.pdf", bbox_inches="tight")
    else:
        import shared

        shared.show_n(1)

if __name__ == "__main__":
    # parse arguments
    # with default radials_diff=(2, 4, 6, 8), diams=(1., 0.5, 1.5, 2.), constant_L=True, kcc2="Y"
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-r", "--radials_diff", type=int, nargs="*", default=[2, 4, 6, 8])
    parser.add_argument("-d", "--diams", type=float, nargs="*", default=[1., 0.5, 1.5, 2.])
    parser.add_argument("-c", "--constant_L", type=str, nargs="*", default=["N", "Y"])
    parser.add_argument("-k", "--kcc2", default="Y")
    # add verbose
    parser.add_argument("-v", "--verbose", action="store_true")
    # add very verbose
    parser.add_argument("-vv", "--very_verbose", action="store_true")
    args = parser.parse_args()

    if args.very_verbose:
        logging.basicConfig(level=logging.DEBUG, force=True)
    elif args.verbose:
        logging.basicConfig(level=logging.INFO, force=True)
    else:
        logging.basicConfig(level=logging.WARNING, force=True)

    # convert to int
    radials_diff = tuple([int(r) for r in args.radials_diff])
    # convert to float
    diams = tuple([float(d) for d in args.diams])
    # convert constant_L str to bool
    constant_L = list(set([c.lower() in ["y", "t", "true"] for c in args.constant_L]))
    
    kcc2 = args.kcc2
    logger.info(f"radials_diff={radials_diff}, diams={diams}, constant_L={constant_L}, kcc2={kcc2}")

    # run
    figure_sink(radials_diff=radials_diff, diams=diams, constant_L=constant_L, kcc2=kcc2)