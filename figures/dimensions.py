import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from utils import settings
from inhib_level.math import accumulation_index, lambda_d
from utils.plot_utils import letter_axes, copy_lines, plot_save

import logging

logger = logging.getLogger()


def radius_length():
    from main import run_inhib_level

    logger.info("*" * 50 + "radius_length" + "*" * 50)

    fig, (ax_A, ax_B) = plt.subplots(nrows=2, ncols=1)

    letter_axes(ax_A, ax_B)
    X = 0.2
    loc_list = [X]
    loc_list_str = " ".join(f"[{_loc}]" for _loc in loc_list)
    radial_list = [2, 4, 8]
    radial_list_str = " ".join(str(r) for r in radial_list)
    e_offset_list = [-2.0, -1.0, 0.0]
    e_offsets = " ".join(str(e) for e in e_offset_list)
    diam_list = [1, 0.5, 2]
    diam_list_str = " ".join(str(d) for d in diam_list)

    logger.info(
        """
    ####################################################################################################################
    # RADIUS and LENGTH 
    ####################################################################################################################
    """
    )
    tstop = 500
    tm = 10
    #  Change diameter AND length constant
    main_args = (
        f"--radial {radial_list_str} --loc {loc_list} --e_offsets {e_offsets}"
        f" --synapse_dists=diffused_matched --kcc2=Y --tstop={tstop} --tm={tm}"
        " --plot_group_by=num_dendrites_arr --plot_color_by=e_offsets"
        " --sections=radial_dends_1"
        f" --diams {diam_list_str}"
        " --precise"
    )
    plot_dict, sim_type, saved_args = run_inhib_level(main_args)
    il_dict = saved_args["il_dict"]
    df_accum = pd.DataFrame(
        index=diam_list,
        columns=pd.MultiIndex.from_product([radial_list, e_offset_list]),
    )
    for key, df in il_dict.items():
        if key == "units":
            continue
        n = int(key[key.index("n=") + 2 : key.index("/")])
        e = float(key[key.index("e=") + 2 : key.index("(")])
        d = float(key[key.index("d=") + 2 : key.index(" l")])
        l = float(key[key.index("l=") + 2 :])
        accum = accumulation_index(df, loc_list)
        df_accum.loc[d, (int(n), float(e))] = accum[X][tstop]
    for d in diam_list:
        for n in radial_list:
            df_accum.loc[d, n].plot(ax=ax_A, cmap=settings.cmap_dict["num_synapses"][n])

    # Change diameter but keep length the same as first diam (1)
    plot_dict, sim_type, saved_args = run_inhib_level(main_args + " --constant_L")
    il_dict = saved_args["il_dict"]
    df_accum = pd.DataFrame(
        index=diam_list,
        columns=pd.MultiIndex.from_product([radial_list, e_offset_list]),
    )
    for key, df in il_dict.items():
        if key == "units":
            continue
        n = int(key[key.index("n=") + 2 : key.index("/")])
        e = float(key[key.index("e=") + 2 : key.index("(")])
        d = float(key[key.index("d=") + 2 : key.index(" l")])
        l = float(key[key.index("l=") + 2 :])
        accum = accumulation_index(df, loc_list)
        df_accum.loc[d, (int(n), float(e))] = accum[X][tstop]
    for d in diam_list:
        for n in radial_list:
            df_accum.loc[d, n].plot(ax=ax_B, cmap=settings.cmap_dict["num_synapses"][n])
    if settings.SAVE_FIGURES:
        plot_save("output/figure_dimensions.png", figs=[fig], close=False)
        plot_save("output/figure_dimensions.pdf")
    else:
        import shared

        shared.show_n(1)


def cl_radius_length(diam=1.0, sample_N=1):
    from main import run_inhib_level
    from shared import env_var

    fig, (ax_time_lr, ax_cl_lr) = plt.subplots(ncols=2)
    # default
    Rm = 20000
    Ra = 100
    L = lambda_d(diam, Rm, Ra)
    L1 = lambda_d(1, Rm, Ra)
    L2 = lambda_d(2, Rm, Ra)
    lengths = list(reversed(sorted({L, L1, L2})))
    tstop = 500
    tm = 10
    loc = 0.1
    units = settings.um
    h_center = True

    leg_lines_l = []
    vinit = -65.0

    for length, ls, c in zip(
        lengths,
        ["-", "--", ":", "-."],
        [settings.COLOR.K, settings.COLOR.Pu, settings.COLOR.Cy, settings.COLOR.B],
    ):

        main_args = (
            f"--radial {sample_N} --loc {loc} --e_offsets -5"
            " --synapse_dists=diffused_matched --kcc2=C"
            " --plot_group_by=num_dendrites_arr --plot_color_by=num_synapses"
            " --sections=radial_dends_1"
            f" --tstop={tstop} --tm={tm} --with-v-trace"
            f" --diams 1.0 2.0 --constant_L={length}"
            " --precise"
        )
        plot_dict, func_name, saved_args = run_inhib_level(main_args)
        il_dict = saved_args["il_dict"]
        ecl_dict = saved_args["ecl_dict"]
        ev = env_var()

        pcl, ecl, phco3, ehco3, egaba, vinit = (
            ev["pcl"],
            ev["ecl"],
            ev["phco3"],
            ev["ehco3"],
            ev["egaba"],
            ev["v_init"],
        )

        lines = plot_dict[f"{sample_N}"][2][settings.Vm].get_lines()

        copy_lines(lines, ax_time_lr, rel_lw=1.0, alpha=0.3, c=c)
        for line in ax_time_lr.get_lines():
            line.set_ydata(line.get_ydata() + vinit)
        # get units for reindex/renaming of index
        df_units = il_dict["units"]
        dend = df_units.index.levels[0][0]
        if h_center and units == settings.um:
            index_vals = df_units.loc[dend, settings.um] - length * loc
        elif h_center:
            index_vals = df_units.loc[dend].index - loc
        else:
            index_vals = df_units.loc[dend, units]

        for k, _df in il_dict.items():
            if k == "units":
                continue
            n = int(k[k.index("n=") + 2 : k.index("/")])
            d = float(k[k.index("d=") + 2 : k.index("l=")])
            lw = 1.3 * d
            cl = False if "(-)" in k else True
            loc_idx = abs(loc - _df.index.levels[1]).argmin()
            _s_t = _df.loc[dend].iloc[loc_idx]
            _s_t = pcl * ecl_dict[k].T.loc[dend].iloc[loc_idx] + phco3 * ehco3
            _s_il = _df.loc[dend].iloc[:, -1]
            _s_il = _s_il.rename(index_vals)
            # plot time series
            _s_t.plot(ax=ax_time_lr, c=c, ls=ls, lw=lw, label=f"{length}{d}")
            # plot inhib level vs distance
            _s_il.plot(ax=ax_cl_lr, c=c, ls=ls, lw=lw, label=f"{length}{d}")

            if length == L and d == diam and n == sample_N:
                _c = c
                # add markers at x and y
                x = index_vals.iloc[loc_idx]
                y = _s_il.loc[x]
                ax_time_lr.plot(
                    tstop, _s_t.iloc[-1], marker="^", c=_c, markeredgecolor="k"
                )
                if h_center:
                    x = 0
                ax_cl_lr.plot(
                    x,
                    y,
                    marker="^",
                    c=_c,
                    markersize=2 + 8 * length / L,
                    markeredgecolor="k",
                    markeredgewidth=d,
                )

        leg_lines_l.append(Line2D([], [], color=c, ls=ls))

    l2s = []
    for line in leg_lines_l:
        l2 = Line2D(
            [],
            [],
            color=line.get_color(),
            ls=line.get_linestyle(),
            lw=line.get_linewidth() * 2,
        )
        l2s.append(l2)
    leg_lines_l = l2s + leg_lines_l
    leg = ax_cl_lr.legend(
        leg_lines_l,
        [""] * len(lengths) + [f"{l:>3.0f} {settings.um}" for l in lengths],
        title=f"  $\mathbf{{Diameter}}$ \n  2 {settings.um}  1 {settings.um}   "
        f"$\mathbf{{Length}}$",
        loc="upper right",
        labelspacing=0.0,
        handletextpad=1.0,
        bbox_to_anchor=(1.2, 1.1),
        ncol=2,
        frameon=True,
        facecolor="w",
        columnspacing=0.0,
        edgecolor="None",
        fontsize="x-small",
        title_fontsize="x-small",
        borderpad=0.0,
    )
    ax_time_lr.set(
        xlabel=f"{settings.TIME} ({settings.ms})",
        ylabel=f"{settings.EGABA} ({settings.mV})",
    )
    ax_time_lr.axhline(
        vinit,
        xmax=tstop,
        linestyle="--",
        color=settings.COLOR.K,
        alpha=0.3,
        linewidth=1.0,
    )
    ax_cl_lr.set(xlabel=f"Distance from synapse ({units})", ylabel=settings.IL)
    ax_cl_lr.spines["left"].set_position(("data", 0))
    ax_cl_lr.spines["left"].set_zorder(-10)
    ax_cl_lr.locator_params(nbins=4, axis="y")
    # ax_cl_lr.set_ylim(-.1, 1.5)
    # ax_cl_lr.set_yticks([0.5, 1.5])
    if settings.SAVE_FIGURES:
        plot_save("output/figure_dimensions_cl.png", figs=[fig], close=False)
        plot_save("output/figure_dimensions_cl.pdf")
    else:
        import shared

        shared.show_n(1)
