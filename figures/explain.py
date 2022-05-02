import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, cm, colors
from matplotlib.lines import Line2D
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from utils import settings
from utils.settings import gi, Rd, Rdi, Ri, Aid, Adi, Vd, Vdi, ILd
from inhib_level.math import lambda_d, dvv
from model.morphology import SingleDend
from model.nrn_simulation import hoc_run
from utils.plot_utils import (
    adjust_spines,
    plot_save,
    opacity,
    new_gridspec,
    letter_axes,
)

import logging

logger = logging.getLogger()


def figure_explain():
    """ Compare different methods for calculating ILd

         |      B
    A    | ----------
         |      C

    A: different traces for analytical, semi-numerical, and numerical voltage

    B: Voltage difference

    C: example structure (soma at left)

    """
    from neuron import h
    from inhib_level.simulation import add_synapse_type
    from utils.plot_utils import shapeplot2d, mark_locations2d

    if settings.SAVE_FIGURES:
        import matplotlib

        matplotlib.use("agg")  # so hatches can be saved

    # setup
    logger.info("*" * 50 + "figure_explain" + "*" * 50)
    saved_args = locals()
    func_name = "analytical"
    plot_dict = {}
    from shared import INIT

    INIT()

    import matplotlib

    matplotlib.use("PDF")  # for hatches

    # local settings
    figsize = (settings.PAGE_W_FULL, settings.PAGE_H_half)
    g_i = 2e-3  # convert from uS to S
    Rm = 20000
    Ra = 100
    loc = [0.4] # inhibitory synapses location i
    tstop = 200
    tm = 5
    diam = 1.0
    L = lambda_d(diam, Rm, Ra)
    # visual settings
    base = -65
    e_off = 0
    _d = 0.3 # excitatory input location example 1
    _d_2 = 0.7 # excitatory input location example 2
    d_locs = [_d, _d_2]
    plot_colors = ["#D394D6", "#D6967E", "#88D69A"]
    d_colors = [settings.COLOR.R, settings.COLOR.O]
    d_cmaps = [
        settings.COLOR.truncate_colormap(plt.get_cmap("Reds"), 0.2, 0.7),
        settings.COLOR.truncate_colormap(plt.get_cmap("Oranges"), 0.2, 0.7),
    ]
    h_alpha_marker = 1.0
    h_alpha_line = 0.3
    h_alpha_other = 0.1
    i_props = dict(
        markersize=8, markeredgecolor="k", color=settings.COLOR.I, markeredgewidth=0.5,
    )
    # create objects
    neuron = SingleDend(dend_L=L, dend_diam=diam, dend_nseg=81, Rm=Rm, Ra=Ra)

    inh_synapses, l1, l2 = add_synapse_type(
        neuron, "inhfluct_simple", [(neuron.dend, loc)], e=base+e_off, gmax=0, netstim_args={}
    )
    exc_synapses, l1, l2 = add_synapse_type(
        neuron, "IClamp", [(neuron.dend, loc)], e=0, netstim_args={}
    )
    exc = exc_synapses[0]["object"]
    exc.amp = 1e-3
    hoc_run(v_init=-65, tstop=0)

    # run
    df_il, df_v, df_v_star, input_events, df_sl, df_ecl = dvv(
        neuron,
        -65,
        inh_synapses,
        exc_synapses,
        vary="exc",
        tstop=tstop,
        tm=tm,
        g_i=g_i,
        e_offset=-0.0,
        calc_shunt=True,
    )
    df_il = df_il.loc[:, range(50, 201, 50)]
    # Make the results look pretty
    h_r = [3, 3, 10]
    w_r = [1, 8, 10, 8]
    fig, gs = new_gridspec(
        len(h_r),
        len(w_r),
        figsize=figsize,
        grid_kwargs={"height_ratios": h_r, "width_ratios": w_r},
    )
    ax_time = fig.add_subplot(gs[1, 1])
    ax_time2 = fig.add_subplot(gs[1, -1], sharex=ax_time)
    ax_shape = fig.add_subplot(gs[0, 1:])
    ax_cbar = fig.add_subplot(gs[0, 0])
    ax_sl = fig.add_subplot(gs[-1, 1:])
    adjust_spines(ax_shape, [], 0)
    # letter_axes(ax_cbar, ax_time, ax_sl)
    letter_axes(ax_cbar, start="A", xy=(0.0, 1.0), xycoords="figure fraction", va="top")
    letter_axes(ax_time, start="B", xy=(0.0, 0.72), xycoords="figure fraction")
    letter_axes(ax_sl, start="C", xy=(0.0, 0.45), xycoords="figure fraction")
    # put DataFrame values in same order as sections for plotting method
    #   weird double transpose as ordering seems to only work for columns
    sec_names = ["dend_1"]
    select_il = df_il.T[sec_names].T[tstop]
    # remove 0.0 and 1.0 as these were added in with `dummy_seg`
    _zero = select_il.xs(0.0, level=1, drop_level=False)
    _one = select_il.xs(1.0, level=1, drop_level=False)
    select_il = select_il.loc[
        ~select_il.index.isin(_zero.index) & ~select_il.index.isin(_one.index)
    ]
    il_values = select_il.values.flatten()
    lines = shapeplot2d(
        h, ax=ax_shape, sections=neuron.dend, cvals=il_values, linewidth=diam * 5
    )

    cb = fig.colorbar(
        cm.ScalarMappable(
            norm=colors.Normalize(0, 1), cmap=settings.IL_config[settings.IL]["cmap"]
        ),
        cax=ax_cbar,
        ax=ax_shape,
        fraction=0.005,
        pad=0.0,
        aspect=10,
    )
    cb.set_label("Inhibitory Level")
    cb.set_ticks([0, 1])
    cb.set_ticklabels(["min", "max"])
    # cb.set_ticklabels(["$IL_{min}$", "$IL_{max}$"])
    cb.ax.tick_params(labelsize="x-small")
    # cb.ax.set_title("Inhibitory Level", fontsize='small')
    cb.ax.yaxis.set_ticks_position("left")
    cb.ax.yaxis.set_label_position("left")
    # adjust ax_A so axes are still aligned

    ax_shape.set_xlim(neuron.soma.L, L)
    ax_shape.set_ylim(-1, 1)
    idx = abs(_d - df_v_star["dend_1"].columns).argmin()
    idx2 = abs(_d_2 - df_v_star["dend_1"].columns).argmin()
    real_d = df_v_star["dend_1"].columns[idx]  # more accurate location
    real_d2 = df_v_star["dend_1"].columns[idx2]  # more accurate location
    real_i = df_v_star["dend_1"].columns[
        abs(loc[0] - df_v_star["dend_1"].columns).argmin()
    ]
    # shape plot annotations
    ax_shape.set_axis_off()
    # exc loc
    mark_locations2d(
        h,
        neuron.dend[0],
        locs=[_d - 3 / L],
        ax=ax_shape,
        markspec=".",
        zorder=100,
        markersize=10,
        color=d_colors[0],
        alpha=h_alpha_marker,
    )
    mark_locations2d(
        h,
        neuron.dend[0],
        locs=[_d_2 - 9 / L],
        ax=ax_shape,
        markspec=".",
        zorder=100,
        markersize=10,
        color=d_colors[1],
        alpha=h_alpha_marker,
    )
    ax_shape.annotate(
        "d",
        xy=(real_d * L + 11, 0),
        xytext=(0, 20),
        fontsize="medium",
        textcoords="offset points",
        color=settings.COLOR.R,
        ha="center",
        arrowprops=dict(arrowstyle="-", connectionstyle="arc3,rad=0"),
    )
    # left/right arrows
    ax_shape.annotate(
        "",
        xy=(real_d * L - 24, 0),
        xytext=(+36, 0),
        fontsize="medium",
        textcoords="offset points",
        color=settings.COLOR.R,
        ha="center",
        arrowprops=dict(arrowstyle="<->", connectionstyle="arc3,rad=0"),
    )
    # soma
    mark_locations2d(
        h,
        neuron.dend[0],
        locs=[0.0],
        ax=ax_shape,
        markspec="o",
        markersize=15,
        color=lines[0].get_color(),
    )

    # inhib loc
    xy_i = (real_i * L + 7.5, 0.2)
    ax_shape.annotate(
        "i",
        xy=xy_i,
        xytext=(0, 20),
        fontsize="medium",
        textcoords="offset points",
        color=settings.COLOR.I,
        ha="center",
        arrowprops=dict(
            arrowstyle="|-|,widthA=0.0,widthB=.3", connectionstyle="arc3,rad=0"
        ),
    )
    mark_locations2d(
        h, neuron.dend[0], locs=[loc[0] - 7 / L], ax=ax_shape, markspec="v", **i_props
    )
    # arr_img = plt.imread("", format='png')
    #
    # imagebox = OffsetImage(arr_img, zoom=1)
    # imagebox.image.axes = ax_shape
    #
    # ab = AnnotationBbox(imagebox, xy_i,
    #                     xybox=(0, 1),
    #                     boxcoords="offset points",
    #                     pad=0.,
    #                     bboxprops=dict(facecolor='none', edgecolor='none'),
    #                     arrowprops=dict(arrowstyle="-")
    #                     )
    # ax_shape.add_artist(ab)

    # IL plot
    result_tm1 = dvv(
        neuron,
        -65,
        inh_synapses,
        exc_synapses,
        vary="exc",
        tstop=tstop,
        tm=5,
        g_i=g_i,
        e_offset=-0.0,
        calc_shunt=True,
    )
    tm_1_lines = ax_sl.plot(
        result_tm1[0].loc["dend_1"],
        linestyle=":",
        linewidth=1.0,
        color=plot_colors[-1],
        alpha=h_alpha_other,
        label=5,
    )

    tm_2_lines = ax_sl.plot(
        df_il.loc["dend_1"],
        # linestyle=":",
        linewidth=1.0,
        color='k',
        alpha=h_alpha_line,
        label=tm,
    )

    R_soma = df_sl["Rd"].loc["soma", 0.5]
    R_i = df_sl["Ri"].loc["dend_1"].values[0]
    R_inf = R_i * np.tanh(1.0)
    rho = R_soma / R_inf

    g_i_Ri = g_i * df_sl["Ri"]
    gir = g_i_Ri / (1 + g_i_Ri) * df_sl["Aid"] * df_sl["Adi"]
    sl_g = ax_sl.plot(
        gir.loc["dend_1"],
        label=r"$\frac{{{giri}}}{{{giri_bottom}}}$ {aid} {adi}".format(
            giri=(gi + "  " + Ri).replace("$", ""),
            giri_bottom="1 + " + (gi + "  " + Ri).replace("$", ""),
            aid=Aid,
            adi=Adi,
        ),
        color='k',
        alpha=1.0,
        lw=4,
    )
    drr = (df_sl["Rd"] - df_sl["Rd*"]) / df_sl["Rd"]
    sl_r = ax_sl.plot(
        drr.loc["dend_1"],
        label="$\\frac{{{rtop}}}{{{rbottom}}}$".format(
            rtop=f"{Rd} - {Rdi}".replace("$", ""), rbottom=f"{Rd}".replace("$", "")
        ),
        # linestyle="--",
        color='k',
        alpha=0.5,
        lw=2,
    )
    sl_v = ax_sl.plot(
        df_il.loc["dend_1"].iloc[:, -1],
        label=r"$\frac{{{vtop}}}{{{vbottom}}}$".format(
            vtop=f"{Vd} - {Vdi}".replace("$", ""), vbottom=f"{Vd}".replace("$", "")
        ),
        # linestyle=":",
        color='k',
        alpha=0.2,
        lw=1,
    )
    # inhibitory synapse location
    df_imark = df_il.loc["dend_1"]
    i_idx = abs(loc[0] - df_imark.index).argmin()
    i_line = ax_sl.plot(
        df_imark.index[i_idx],
        df_imark.iloc[i_idx, -1],
        linestyle="None",
        marker="v",
        **i_props,
    )
    # ax[1].legend(title='$t$')
    ax_sl.set_ylabel("Inhibitory Level [$IL$]")
    ax_sl.set_xlabel("Distance from soma (X)")
    ax_sl.set_xticks(d_locs, minor=True)
    # plot df_sl over time on same axis

    # plot points of IL @ d at different time points
    sl_d_time = df_il.loc["dend_1"].iloc[idx, :]
    sl_d2_time = df_il.loc["dend_1"].iloc[idx2, :]
    _x = [real_d] * sl_d_time.size
    _x2 = [real_d2] * sl_d2_time.size
    df_sl_d_time = pd.DataFrame(sl_d_time, columns=[real_d])
    df_sl_d2_time = pd.DataFrame(sl_d2_time, columns=[real_d2])
    df_sl_d_time.T.plot(
        ax=ax_sl,
        legend=False,
        linestyle="None",
        marker=".",
        alpha=h_alpha_marker,
        cmap=d_cmaps[0],
    )
    df_sl_d2_time.T.plot(
        ax=ax_sl,
        legend=False,
        linestyle="None",
        marker=".",
        alpha=h_alpha_marker,
        cmap=d_cmaps[1],
    )

    lines, labels = ax_sl.get_legend_handles_labels()
    line_colors = [_l.get_color() for _l in lines if _l.get_marker() == "."]
    values = [float(_l) for _l in labels if "$" not in _l]

    d_dv = df_v - df_v_star
    v_t = df_v["dend_1"].iloc[:, idx]
    v_t2 = df_v["dend_1"].iloc[:, idx2]
    v_star_t = df_v_star["dend_1"].iloc[:, idx]
    v_star_t2 = df_v_star["dend_1"].iloc[:, idx2]
    # swap index (time) with values (V/IL)
    v_t_s = pd.Series(dict((v, k) for k, v in v_t.items()))
    v_star_t_s = pd.Series(dict((v, k) for k, v in v_star_t.items()))
    sl_d_time_s = pd.Series(dict((v, k) for k, v in sl_d_time.items()))

    v_t += base
    v_star_t += base
    v_t2 += base
    v_star_t2 += base

    for _d_loc, _real_d, _ax, _df, _v, _v_star in zip(
        d_locs,
        [real_d, real_d2],
        [ax_time, ax_time2],
        [sl_d_time, sl_d2_time],
        [v_t, v_t2],
        [v_star_t, v_star_t2],
    ):
        _d_idx = d_locs.index(_d_loc)
        _v_line = _ax.plot(
            _v, linestyle="-", c="k", label=f"{settings.Vm} - excitation (d) only"
        )
        _v_fill = _ax.fill_between(
            _v.index,
            y1=_v,
            y2=[base] * len(_v.index),
            color="k",
            alpha=0.2,
            lw=0,
            label=Vd,
        )
        _v_star_line = _ax.plot(
            _v_star,
            linestyle="--",
            c="k",
            label=f"{settings.Vm} - excitation (d) and inhibition (i)",
        )
        _v_star_fill = _ax.fill_between(
            _v_star.index,
            y1=_v_star,
            y2=[base] * len(_v_star.index),
            color="k",
            alpha=0.2,
            lw=0,
            hatch="//",
            rasterized=True,
            label=Vdi,
        )
        integration_window = _v.loc[145:150]

        # _ax.fill_between(integration_window.index,
        #                  y1=list(integration_window),
        #                  y2=[base]*len(integration_window),
        #                  color=d_colors[d_locs.index(_d_loc)],
        #                  alpha=0.8, lw=0,
        #                  label="$\Delta t = 5 ms$")

        ymax = _ax.get_ylim()[1]

        _ax_sl = _ax.twinx()
        _ax_sl.set_clip_on(False)
        # for i, (idx_v, v) in enumerate(_df.loc[::-1].iteritems()):
        #     c = line_colors[len(line_colors)//(len(d_locs) - d_locs.index(_d_loc)) - i - 1]
        #     _ax_sl.plot(idx_v, v,
        #                 linestyle='None', marker='.',
        #                 alpha=h_alpha_marker, c=c)
        # _ax_sl.axvline(v, base, np.ceil(ymax), c=c,
        #                alpha=h_alpha_line, linestyle='-', linewidth=1.)
        _ax.set_xticks(np.append(0, _df.index.values))
        _ax.set_xticks(np.arange(0, _df.index.values.max() + 5, 5), minor=True)
        # _ax.vlines(values, base, np.ceil(ymax), colors=[settings.COLOR.R]*len(values),
        #            alpha=h_alpha_other, linestyles=['-']*len(values), linewidth=1.)
        df = result_tm1[0].loc["dend_1", _real_d].to_frame()
        diag = np.diag([1] * len(df))
        square_df = pd.DataFrame(
            np.tile(df.values, len(df.index)),
            index=df.index.values,
            columns=df.index.values,
        )
        square_df = square_df.mask(~diag.astype(bool))  # only keep diag
        square_df.plot(
            ax=_ax_sl, marker=".", linestyle="None", ms=4, cmap=d_cmaps[_d_idx]
        )
        square_df[_df.index].plot(
            ax=_ax_sl, marker=".", linestyle="None", ms=8, cmap=d_cmaps[_d_idx]
        )
        # _ax.set_ylim(base, base + 1)
        _ax_sl.set_ylim(0, ax_sl.get_ylim()[1])
        _ax.set_xlim(0, tstop)
        _ax_sl.set_xlim(0, tstop)
        # use_scalebar(ax=ax_time,
        #              matchx=True, matchy=True, hidex=True, hidey=True,
        #              loc=4,
        #              labely=f"{settings.ms}",
        #              labelx=f"{settings.mV}"
        #              )
        if base == 0:
            _ax.set_ylabel(
                f"Voltage\ndeflection\n({settings.mV})",
                fontdict={"fontsize": "x-small"},
            )
        else:
            _ax.set_ylabel(
                f"Membrane\npotential\n({settings.mV})",
                fontdict={"fontsize": "x-small"},
            )
        _ax_sl.set_ylabel(f"Inhibitory Level")
        adjust_spines(_ax, ["top", "bottom", "left", "right"], position=0)
        _ax.yaxis.tick_right()
        _ax.yaxis.set_label_position("right")
        _ax_sl.yaxis.tick_left()
        _ax_sl.yaxis.set_label_position("left")

        adjust_spines(_ax_sl, ["left", "bottom"], position=0)
        _ax.set_xlabel(
            f"{settings.TIME} {settings.UNITS(settings.ms)}",
            fontdict={"fontsize": "small"},
        )

        _v_line = _v_line if _d_loc == _d else []
        _v_star_line = _v_star_line if _d_loc == _d else []
        lines = _v_line + _v_star_line + [_v_fill] + [_v_star_fill]
        labels = [l.get_label() for l in lines]
        if _d_loc == _d:
            lines = _v_line + _v_star_line
            labels = [l.get_label() for l in lines]
            leg = _ax.legend(
                lines,
                labels,
                fontsize="x-small",
                loc="lower left",
                bbox_to_anchor=(1, 1.5),
                ncol=1,
                borderaxespad=0.0,
                columnspacing=1.0,
                labelspacing=0.5,
                frameon=False,
            )
            _ax.add_artist(leg)
            lines = [_v_fill] + [_v_star_fill]
            labels = [l.get_label() for l in lines]
            leg = _ax.legend(
                lines,
                labels,
                fontsize="small",
                loc="upper left",
                bbox_to_anchor=(1.55, 1.1),
                ncol=1,
                borderaxespad=0.0,
                columnspacing=1.0,
                labelspacing=0.5,
                frameon=False,
            )
        leg = _ax_sl.legend(
            [_ax_sl.get_lines()[-1]],
            [ILd.replace("d", f"d={_d_loc}")],
            fontsize="small",
            loc="lower center",
            bbox_to_anchor=(0.5, 1),
            borderaxespad=0.0,
            labelspacing=0.1,
            handlelength=0.8,
            frameon=False,
        )
    # pandas hides xlabel and xticklabels so bring ithem back
    adjust_spines(ax_time, ["top", "bottom", "left", "right"], position=0)
    ax_time.tick_params(labelbottom=True)
    ax_time.xaxis.get_label().set_visible(True)
    ax_time.yaxis.tick_right()
    ax_time.yaxis.set_label_position("right")
    # get ymin and ymax for ax_time and ax_time2
    ymin = np.min([ax_time.get_ylim()[0], ax_time2.get_ylim()[0]])
    ymax = np.max([ax_time.get_ylim()[1], ax_time2.get_ylim()[1]])
    ax_time.set_ylim(ymin, ymax)
    ax_time2.set_ylim(ymin, ymax)

    ax_sl.set_xlim(0, 1)
    # l = Line2D([], [], linestyle='None',
    #            marker='v', ms=8,
    #            color=settings.COLOR.I, alpha=0.3, markeredgecolor='k')
    math_lines = sl_g + sl_r + sl_v
    labels = [l.get_label() for l in math_lines]
    math_lines += tm_2_lines[0:1]
    labels += ["IL every 5 ms"]
    legend = ax_sl.legend(
        math_lines,
        labels,
        loc="lower left",
        bbox_to_anchor=(1.0, 0.0),
        ncol=1,
        mode=None,
        borderaxespad=0.0,
        frameon=False,
        fontsize="small",
        title="IL calculation",
        title_fontsize="small",
    )
    ax_sl.add_artist(legend)
    # ax_sl.legend(i_line,
    #              [ILd.replace("d", f"d=i={loc[0]}")],
    #              loc='lower center', bbox_to_anchor=(0.5, 1), ncol=1,
    #              borderaxespad=0., frameon=False, fontsize='x-small'
    #              )
    x, y = i_line[0].get_data()
    ax_sl.annotate(
        ILd.replace("d", f"d=i={loc[0]}"),
        xy=(x[0], y[0]),
        xytext=(0, 5),
        textcoords="offset points",
        fontsize="x-small",
    )
    gs.update(left=0.1, hspace=0.8, wspace=0.2, top=0.93, right=0.8)

    if settings.SAVE_FIGURES:
        plot_save("output/figure_explain.png", figs=[fig], close=False)
        plot_save("output/figure_explain.pdf")
    return saved_args, func_name, plot_dict
