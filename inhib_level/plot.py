import logging

import numpy as np
import pandas as pd
from collections import OrderedDict
from cycler import cycler
from matplotlib import pyplot as plt, colors, colorbar, cycler, ticker
from matplotlib.lines import Line2D
from neuron import h

from utils import settings
from utils.plot_utils import opacity, adjust_spines
from utils.plot_utils import plot_input_events

logger = logging.getLogger(__name__)


def create_figure(
    new_once=True, voltage=False, attenuation=False, time=None, shape=False
):
    def fig_type():
        # plot
        fig = plt.figure()
        named_axes = {}
        if time:
            size = (2, 10)
            named_axes[settings.TIME] = plt.subplot2grid(
                size, (0, 0), colspan=size[1], fig=fig
            )
        elif voltage:
            size = (2, 10)
            named_axes[settings.Vm] = plt.subplot2grid(
                size, (0, 0), colspan=size[1], fig=fig
            )
        else:
            size = (1, 10)

        named_axes[settings.INHIB_LEVEL] = plt.subplot2grid(
            size, (size[0] - 1, 0), colspan=size[1], fig=fig
        )

        if attenuation:
            named_axes[settings.ILid] = named_axes[
                settings.INHIB_LEVEL
            ].twinx()  # add second axis to this axis
            # Set ax's patch invisible
            named_axes[settings.INHIB_LEVEL].patch.set_visible(False)
            named_axes[settings.ILid].patch.set_visible(False)
            # place axes[1] in front of axes[2]
            named_axes[settings.INHIB_LEVEL].set_zorder(
                named_axes[settings.ILid].get_zorder() + 1
            )

        axes = []
        order = [settings.TIME, settings.Vm, settings.INHIB_LEVEL, settings.ILid]
        for name in order:
            if name in named_axes:
                axes.append(named_axes[name])
        fig.set_constrained_layout(True)
        return fig, axes, named_axes

    if new_once:
        # call now
        return fig_type()
    # call later (and repeatably)
    return fig_type


def inhib_level_plot(
    figure,
    at_hotspot=False,
    tm=None,
    t=None,
    neuron=None,
    inhib_syn_type=None,
    exc_syn_type=None,
    df_v=None,
    df_v_star=None,
    df_il=None,
    df_il_attenuation=None,
    df_ecl=None,
    x_units=None,
    hotspot_record_locations=None,
    inhib_synapse_locations=None,
    inhib_n_loc_insert_actual_short=None,
    colormap=None,
    show_input=None,
    input_events=None,
    iter_label=None,
    time=None,
    draw_sections=False,
    plot_shape=False,
):
    # graphs
    if type(figure) is tuple:
        fig, axes, named_axes = figure
    else:
        fig, axes, named_axes = figure()

    # x_units = 'X'
    # because of the floating point mantissa, need to adjust the locations to the indexed values
    # Note this is deprecated in favour of formatting locations to 5 decimal points
    # for i, loc in enumerate(hotspot_loc):
    #     hotspot_loc[i] = loc + min(abs(df_il.index.values - loc))
    # for i, loc in enumerate(inhib_loc):
    #     inhib_loc[i] = loc + min(abs(loc - df_il.index.values))

    col_samples = None
    if settings.Vm in named_axes:
        logger.info(
            """
                                                ##########################
                                                # VOLTAGE PLOT
                                                ##########################"""
        )
        # choose samples to plot
        section_sample = inhib_n_loc_insert_actual_short[0][0]
        if type(section_sample) is list:
            section_sample = section_sample[0]
        if type(section_sample) is str:
            if "[" in section_sample:
                section_sample, index = section_sample.split("[")
                index = int(index[:-1]) + 1  # ignore ']'
            else:
                index = "1"
            section_sample_name = "{}_{}".format(section_sample, index)
            exists = getattr(neuron, section_sample_name, False)
            if not exists:
                section_sample_name = section_sample
        else:
            section_sample_name = section_sample.name()
        cols = df_v[section_sample_name].columns
        mid = int(len(cols) / 2)
        quart = int(mid / 2)
        col_samples = cols[[0, quart, mid, mid + quart, -1]].values
        v_samples = [(section_sample_name, v_sample) for v_sample in col_samples]

        # also include places of input
        hot_locs = [h_loc for h_sec, h_loc in hotspot_record_locations]
        col_samples = np.sort(
            list(set(np.append(col_samples, hot_locs + inhib_synapse_locations)))
        )

        Vm_color_cycle = []
        for i, column in enumerate(df_v[section_sample_name].columns):
            if at_hotspot and column in hot_locs:
                Vm_color_cycle.append(opacity(100, "d62728"))
            elif not at_hotspot and column in inhib_synapse_locations:
                Vm_color_cycle.append(opacity(100, "1f77b4"))
            else:
                Vm_color_cycle.append(opacity(10, "7f7f7f"))
        color_cycle = cycler("color", Vm_color_cycle)
        named_axes[settings.Vm].set_prop_cycle(color_cycle)

        # plot
        named_axes[settings.Vm].plot(df_v, linestyle="-")
        # plot input events
        if show_input:
            plot_input_events(
                named_axes[settings.Vm], input_events, y_offset=df_v.max().max()
            )

        # named_axes[settings.Vm].plot(df_v_star.loc[:, v_samples], linestyle='--')
        named_axes[settings.Vm].plot(df_v_star, linestyle="--")

        vd_legend = named_axes[settings.Vm].get_legend()
        if vd_legend is not None:
            labels_Vd = [
                text.get_text() for text in vd_legend.texts
            ]  # contains Text objects
            labels_Vd = list(filter(lambda x: not x.startswith("("), labels_Vd))
        else:
            labels_Vd = []
        # remove generated ones from dataframe
        labels_kv = {}
        for prev_label_vd in labels_Vd:
            d_start = prev_label_vd.find("{") + 1
            if prev_label_vd[d_start : d_start + 4] == "soma":
                prev_label_loc = 0.00
            else:
                prev_label_loc = float(prev_label_vd[d_start : d_start + 4])
            labels_kv[prev_label_loc] = prev_label_vd
        # lines,labels_Vd_star = named_axes[settings.Vm].get_legend_handles_labels()
        for i, sample in enumerate(v_samples):
            sec_name = sample[0]
            loc = sample[1]
            extra = ""
            if loc in hot_locs:
                extra = "(hotspot)"
            elif loc in inhib_synapse_locations:
                extra = "(g_i)"
            elif loc == min(col_samples):
                loc = (
                    np.nan
                )  # a ".replace('nan','')" hack is required below to remove the 'nan' as .2f requires a
                # float
                extra = "soma"
            vd_label = settings.Vd.replace(
                "d", "{:.2f}{}".format(loc, extra).replace("nan", "")
            )
            label = vd_label.replace("V_", "")
            if vd_label not in labels_Vd:
                labels_Vd.append(vd_label)
            if loc == np.nan:
                loc = 0.00
            if loc not in labels_kv or len(label) > len(labels_kv[loc]):
                labels_kv[loc] = label
            # labels_Vd_star.append(settings.Vdstar.replace('d', "{:.2f}{}".format(loc, extra).replace('nan', '')))

        # created ordered tuple on labels_kv keys
        ordered_loc_labels = OrderedDict(sorted(labels_kv.items()))
        ordered_labels = list(ordered_loc_labels.values())
        ordered_locs = list(ordered_loc_labels.keys())
        if np.nan in ordered_locs:
            index = ordered_locs.index(np.nan)
            ordered_locs[index] = 0.00

        # legend of voltage trace
        # we now have all labels from all sims
        rgb_colors = plt.get_cmap("Greys", len(labels_Vd) + 1)
        custom_lines = []
        for i, label in enumerate(ordered_labels):
            if "hotspot" in label or "g_i" in label:
                color = opacity(50, colors.rgb2hex(rgb_colors(i + 1)[:3]))
            else:
                color = opacity(50, colors.rgb2hex(rgb_colors(i + 1)[:3]))
            custom_lines.append(Line2D([], [], color=color, linestyle="-"))
        l = named_axes[settings.Vm].legend(
            custom_lines,
            ordered_labels,
            bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
            loc=4,
            ncol=4,
            mode=None,
            borderaxespad=0.0,
            fontsize="xx-small",
        )
        if len(named_axes) >= 5:
            # colorbar as legend
            full_grey_cmap = plt.get_cmap("Greys")

            # Get the colormap colors
            my_cmap = full_grey_cmap(np.arange(full_grey_cmap.N))
            # Set alpha
            my_cmap[:, -1] = 1
            # Create new colormap
            my_cmap = colors.ListedColormap(my_cmap)
            # bounds = v_samples
            # norm = colors.BoundaryNorm(v_sample_locs, full_grey_cmap.N)
            cb2 = colorbar.ColorbarBase(
                named_axes[settings.Vm],
                cmap=my_cmap,
                # norm=norm,
                ticks=ordered_locs,  # optional
                spacing="proportional",
                orientation="vertical",
            )
            cb2.ax.tick_params(axis="y", which="minor", colors="blue")
            cb2.set_ticklabels(ordered_labels)
            cb2.set_label("Location, d (X)")

        # plt.colorbar(rgb_colors,ax=named_axes[settings.Vm])
        l.set_title(
            "{} \u2500  \t {} \u254C".format(settings.Vd, settings.Vdstar),
            prop={
                "family": "serif",
                "stretch": "normal",
                "weight": "bold",
                "size": "large",
            },
        )

    if settings.TIME in named_axes:
        logger.info(
            """
                                                    ##########################
                                                    # TIME PLOT
                                                    ##########################"""
        )
        time_loc = time
        # get non-soma section
        sec_name = None
        for section_name in df_il.index.levels[0]:
            if section_name != "soma" and section_name != "axon":
                sec_name = section_name
        sec_values = df_il.loc[sec_name]

        actual_d_index = np.argmin(np.abs(time - sec_values.index.values))
        sl_d = sec_values.iloc[actual_d_index]
        if colormap is not None:
            rgb_colors = plt.get_cmap(
                colormap, 2 + len(df_il.index.levels[0]) * 2 * t / tm
            )
            axes1_color_cycle = [
                colors.rgb2hex(rgb_colors(i)[:3])
                for i in range(int(rgb_colors.N / 2), rgb_colors.N)
            ]
        else:
            axes1_color_cycle = None
        # plot all locations at low opacity
        named_axes[settings.TIME].plot(
            sec_values.T, color=opacity(10, axes1_color_cycle[0])
        )
        # plot specific line
        sl_d.plot(
            ax=named_axes[settings.TIME], color=axes1_color_cycle[0], label=time_loc
        )
        named_axes[settings.TIME].legend()

    ##########################
    # DISTANCE CALCULATION
    ##########################
    # SET DISTANCE ORIGIN
    where = 0
    distance_sec = neuron.soma
    hotspot_loc = 0
    if at_hotspot and len(hotspot_record_locations) == 1:
        logger.debug("setting distance measurement from hotspot")
        # set loc origin to hotspot location
        hotspot_sec_name, hotspot_loc = hotspot_record_locations[0]
        for sec in neuron.sections:
            if sec.name() == hotspot_sec_name:
                hotspot_sec = sec
                break
        else:
            raise Exception("section not found {}".format(hotspot_sec_name))
        where = hotspot_loc
        distance_sec = hotspot_sec
    else:
        logger.debug("setting distance measurement from soma")
        # set loc origin to soma
        if "radial" in neuron.soma.name():
            # no actual soma - soma is an equal arm
            where = 1
    # set distance origin at 'where' of 'distance_sec' - 0 is for initialisation
    h.distance(0, where, sec=distance_sec)

    # skip the 'axon' section
    excluded = "axon"
    if excluded in df_il.index.levels[0]:
        indices = df_il.index.levels[0].difference([excluded])
        # indx = pd.IndexSlice[:, indices.values]
        df_il = df_il.loc[indices, :]

    logger.debug("distance units for inhib level is {}".format(x_units))
    df_distance = pd.DataFrame(columns=[settings.um, "X"])

    for sec in neuron.sections:
        sec_name = sec.name()
        if sec_name == "axon":
            continue
        d = h.distance(0.0, sec=sec)
        x = relative_x = float("{:.5f}".format(0.0))
        if at_hotspot:
            if sec_name != "soma":
                relative_x = x - hotspot_loc
            d *= -1
        df_temp_um = pd.DataFrame(
            {settings.um: d, "X": relative_x}, index=[(sec_name, x)]
        )
        df_distance = df_distance.append(df_temp_um)
        for seg in sec:
            d = h.distance(seg.x, sec=sec)
            x = relative_x = float("{:.5f}".format(seg.x))
            if at_hotspot:
                relative_x = x - hotspot_loc
                if sec_name == "soma" or seg.x < hotspot_loc:
                    d *= -1
            df_temp_um = pd.DataFrame(
                {settings.um: d, "X": relative_x}, index=[(sec_name, x)]
            )
            df_distance = df_distance.append(df_temp_um)
        d = h.distance(1.0, sec=sec)
        x = relative_x = float("{:.5f}".format(1.0))
        if at_hotspot:
            if sec_name == "soma":
                d *= -1
            else:
                relative_x = x - hotspot_loc
        df_temp_um = pd.DataFrame(
            {settings.um: d, "X": relative_x}, index=[(sec_name, x)]
        )
        df_distance = df_distance.append(df_temp_um)

    # create a MultiIndex
    df_distance = df_distance.reindex(
        pd.MultiIndex.from_tuples(df_distance.index)
    ).sort_index()
    # create an index for IL dataframe with chosen 'x' units
    alt_index = pd.MultiIndex.from_tuples(
        zip(df_distance[x_units].index.droplevel(level=1), df_distance[x_units]),
        names=["compartment_name", x_units],
    )

    # SET COLORS
    section_names = df_il.index.levels[0]
    if colormap is None:
        axes1_color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    else:
        rgb_colors = plt.get_cmap(colormap, 2 + len(section_names) * 2 * t / tm)
        # convert to hex
        axes1_color_cycle = [
            colors.rgb2hex(rgb_colors(i)[:3])
            for i in range(int(rgb_colors.N / 2), rgb_colors.N)
        ]

    grey_rgb_colors_sections = plt.get_cmap("Greys", 2 + len(section_names) * 2)
    grey_color_cycle_sections = [
        colors.rgb2hex(grey_rgb_colors_sections(i)[:3])
        for i in range(int(grey_rgb_colors_sections.N / 2), grey_rgb_colors_sections.N)
    ]

    named_axes[settings.INHIB_LEVEL].set_prop_cycle(cycler("color", axes1_color_cycle))

    if settings.ILid in named_axes:
        named_axes[settings.INHIB_LEVEL_ATTENUATION].set_prop_cycle(
            cycler("color", axes1_color_cycle)
        )

    logger.info(
        """
                                            ##########################
                                            # PLOT INHIBITORY LEVEL
                                            ##########################"""
    )
    for section_name in section_names:
        if section_name == "axon":
            continue
        if x_units == "X" and section_name == "soma":
            # can cause ugly discontinuities
            continue
        named_axes[settings.INHIB_LEVEL].plot(
            df_distance.loc[section_name, x_units][df_il.loc[section_name].index],
            df_il.loc[section_name],
            linestyle="-",
            # color=default_color_cycle[hotspot_color_index],
            label=section_name,
        )

    logger.info("formatting figure")

    if tm < t:
        logger.debug("adding tm next to each trace")
        for t_i, tm_value in enumerate(df_il):
            if tm_value not in [df_il.columns[0], df_il.columns[-1]]:
                continue
            # y = df_il.max()[tm_value]
            # x = df_il.loc[:, tm_value][df_il.loc[:, tm_value] == y].index.values
            x = df_distance.loc[df_il.loc[:, tm_value].index.values[-4], x_units]
            y = df_il.loc[:, tm_value].values[-4]

            named_axes[settings.INHIB_LEVEL].text(
                x,
                y,
                "{:>5.0f}".format(tm_value),
                rotation=0,
                ha="left",
                va="center",
                color=opacity(80, axes1_color_cycle[t_i]),
                # bbox=dict(boxstyle="square",
                #           ec=(1, 1, 1, 0),
                #           fc=opacity(80,axes1_color_cycle[t_i]), # white = (1, 1,
                #           1, 0.8)
                #           )
            )

    logger.debug("placing markers")
    if len(inhib_n_loc_insert_actual_short) > 0 and not at_hotspot:
        for inh_section_name, inh_section_loc in inhib_n_loc_insert_actual_short:
            if x_units == "X" and inh_section_name == "soma":
                # can cause ugly discontinuities
                continue
            try:
                y_points = df_il.loc[(inh_section_name, inh_section_loc)].values
            except (KeyError, TypeError) as err:
                logger.warning(
                    f"if specific sections/segments weren't specified, investigate this {err}"
                )
                continue

            for y in y_points:
                named_axes[settings.INHIB_LEVEL].plot(
                    df_distance.loc[(inh_section_name, inh_section_loc), x_units],
                    y,
                    linestyle="none",
                    color=opacity(100, axes1_color_cycle[0]),
                    marker="v",
                    markeredgecolor="k",
                    label=None,
                )
                # if 'radial' in inh_section_name:
                #     # check if accumulation
                #     named_axes[settings.INHIB_LEVEL].axhline(y,
                #                                              xmax=2, lw=0.5,
                #                                              linestyle=':', color=opacity(10, '#000000'))
    if len(hotspot_record_locations) > 0 and at_hotspot:
        for hp_section_name, hp_section_loc in hotspot_record_locations:
            if x_units == "X" and hp_section_name == "soma":
                # can cause ugly discontinuities
                continue
            y_points = df_il.loc[(hp_section_name, hp_section_loc)].values
            for y in y_points:
                named_axes[settings.INHIB_LEVEL].plot(
                    df_distance.loc[(hp_section_name, hp_section_loc), x_units],
                    y,
                    linestyle="none",
                    color=opacity(100, axes1_color_cycle[0]),
                    marker="o",
                    label=None,
                )

    if settings.ILid in named_axes:
        logger.info("plotting inhib level attenuation")
        for sec_name in df_il_attenuation.index.levels[0]:
            if sec_name in df_il_attenuation.index and sec_name in df_distance.index:
                named_axes[settings.INHIB_LEVEL_ATTENUATION].plot(
                    df_distance.loc[sec_name, x_units],
                    df_il_attenuation.loc[sec_name, :],
                    linestyle=":",
                    # color=default_color_cycle[hotspot_color_index],
                    label="{}.{}".format(sec_name, df_distance.loc[sec_name, "X"]),
                )

    # DRAW Sections
    if x_units == settings.um and draw_sections:
        logger.debug("adding sections to figure")
        upper_bound_secs = {}
        for sec_name in df_il.index.levels[0]:
            if sec_name not in df_distance.index:
                continue
            upper_bound_sec = df_distance.loc[sec_name, x_units].max()
            if upper_bound_sec in upper_bound_secs:
                upper_bound_secs[upper_bound_sec] += 1
            else:
                upper_bound_secs[upper_bound_sec] = 1
            if sec_name[-1].isdigit():
                sec_name = sec_name[:-2]
                if upper_bound_secs[upper_bound_sec] > 1:
                    # section is same distance away from soma (default), skip plotting
                    continue
            named_axes[settings.INHIB_LEVEL].axvline(
                upper_bound_sec,
                ymax=2,
                lw=0.5,
                linestyle="-",
                color=opacity(10, "#000000"),
            )
            named_axes[settings.INHIB_LEVEL].text(
                upper_bound_sec,
                0.1,
                "{}".format(sec_name),
                rotation=90,
                ha="right",
                va="center",
                color=opacity(10, "#000000"),
                fontsize="xx-small",
            )
        for sec_loc, num_repeats in upper_bound_secs.items():
            if num_repeats > 1:
                named_axes[settings.INHIB_LEVEL].text(
                    sec_loc + 1,
                    0.1,
                    "[ x {}]".format(num_repeats),
                    withdash=True,
                    rotation=90,
                    ha="left",
                    va="center",
                    color=opacity(10, "#000000"),
                    fontsize="xx-small",
                )
    logger.debug("customising figure legends")

    # FIGURE SETTINGS

    labels = [settings.gi, "'Hotspot'"]
    custom_lines = [
        Line2D(
            [],
            [],
            color=opacity(100, grey_color_cycle_sections[0]),
            linestyle="none",
            marker="v",
            markeredgecolor="k",
        ),
        Line2D(
            [],
            [],
            color=opacity(100, grey_color_cycle_sections[-1]),
            linestyle="--",
            marker="o",
        ),
    ]
    if len(hotspot_record_locations) == 0:
        labels = labels[:1]
        custom_lines = custom_lines[:1]
    # just put g_i and Hotspot in labels
    named_axes[settings.INHIB_LEVEL].legend(
        custom_lines,
        labels,
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
        loc=4,
        ncol=4,
        mode=None,
        borderaxespad=0.0,
    )

    logger.debug("setting labels")
    if settings.Vm in named_axes:
        named_axes[settings.Vm].set_ylabel(
            " ".join([settings.RELATIVE_VOLTAGE_SYMBOL, settings.UNITS(settings.mV)])
        )
        named_axes[settings.Vm].set_xlabel(
            " ".join([settings.TIME, settings.UNITS(settings.ms)])
        )
        named_axes[settings.Vm].set_ylim(auto=True)
        named_axes[settings.Vm].set_xlim(0, t)

    if settings.TIME in named_axes:
        y_label = []
        if at_hotspot:
            y_label.append(settings.ILh)
        else:
            y_label.append(settings.ILd)
        named_axes[settings.TIME].set_ylabel(" ".join(y_label))
        named_axes[settings.TIME].set_xlabel(
            " ".join([settings.TIME, settings.UNITS(settings.ms)])
        )
        named_axes[settings.TIME].set_ylim(auto=True)
        named_axes[settings.TIME].set_xlim(0, t)

    if at_hotspot:
        y_label = '{} at "hotspot" ({})'.format(settings.IL, settings.ILh)
        named_axes[settings.INHIB_LEVEL].set_ylabel(
            ylabel=y_label, labelpad=-30, rotation=270, va="bottom"
        )
        x_label = 'Distance of {} from "hotspot" ({})'.format(settings.gi, x_units)
        named_axes[settings.INHIB_LEVEL].spines["left"].set_position(("data", 0))
    else:
        y_label = settings.IL
        named_axes[settings.INHIB_LEVEL].set_ylabel(y_label)
        x_label = "Distance from junction ({})".format(x_units)

    named_axes[settings.INHIB_LEVEL].set_xlabel(x_label)
    named_axes[settings.INHIB_LEVEL].autoscale(True)
    ax_sl_ylim = named_axes[settings.INHIB_LEVEL].get_ylim()
    # named_axes[settings.INHIB_LEVEL].set_ylim(bottom=min(0, ax_sl_ylim[0]))

    # yticks = np.linspace(0, ax_sl_ylim[1], 7)
    # yticks = named_axes[settings.INHIB_LEVEL].get_yticks()
    # yticks_major = yticks[0::2]
    # if at_hotspot:
    #     yticks_major = yticks_major[1:]
    # named_axes[settings.INHIB_LEVEL].set_yticks(yticks_major, minor=False)
    # named_axes[settings.INHIB_LEVEL].set_yticks(yticks, minor=True)

    if not df_il_attenuation.empty and settings.INHIB_LEVEL_ATTENUATION in named_axes:
        named_axes[settings.INHIB_LEVEL_ATTENUATION].set_ylabel(
            settings.INHIB_LEVEL_ATTENUATION + "\nat 'Hotspot'"
        )
        named_axes[settings.INHIB_LEVEL_ATTENUATION].set_xlabel(
            "$d$ ({})".format(named_axes[settings.INHIB_LEVEL_ATTENUATION].get_xlabel())
        )
        named_axes[settings.INHIB_LEVEL_ATTENUATION].set_ylim(auto=True)
        named_axes[settings.INHIB_LEVEL_ATTENUATION].set_ylim(
            bottom=min(
                [
                    0,
                    named_axes[settings.INHIB_LEVEL_ATTENUATION].get_ylim()[0],
                    df_il_attenuation.min().values[0],
                ]
            ),
            top=max(
                [
                    1.01,
                    named_axes[settings.INHIB_LEVEL_ATTENUATION].get_ylim()[1],
                    df_il_attenuation.max().values[0],
                ]
            ),
        )

    # named_axes[settings.INHIB_LEVEL].set_xlim(0, 1 if x_units)
    # named_axes[settings.INHIB_LEVEL_ATTENUATION].set_xlim(0, 1)

    # add extra legend on first plot for different iterations of this method
    if iter_label is not None:
        if iter_label == "auto":
            iter_label = "{}{}{}".format(
                settings.SYNAPSE_MAP[exc_syn_type] if exc_syn_type is not None else "",
                " + "
                if (exc_syn_type is not None and inhib_syn_type is not None)
                else "",
                settings.SYNAPSE_MAP[inhib_syn_type]
                if inhib_syn_type is not None
                else "",
            )
        logger.debug(f"iter_label = {iter_label}")

        inh_section_name, inh_section_loc = inhib_n_loc_insert_actual_short[0]
        try:
            df_values = df_il.loc[(inh_section_name, inh_section_loc)].values
        except (KeyError, TypeError) as err:
            df_values = []
        for df_value in df_values:
            named_axes[settings.INHIB_LEVEL].plot(
                df_distance.loc[(inh_section_name, inh_section_loc), x_units],
                df_value,
                linestyle="-",
                color=opacity(100, axes1_color_cycle[0]),
                label=iter_label,
            )
        # just put g_i and Hotspot in labels
        named_axes[settings.INHIB_LEVEL].legend()
        lines, labels = named_axes[settings.INHIB_LEVEL].get_legend_handles_labels()
        keep_lines = []
        keep_labels = []
        for line, label in zip(lines, labels):
            if label in section_names:
                continue
            if label in keep_labels:
                continue
            keep_lines.append(line)
            keep_labels.append(label)
        named_axes[settings.INHIB_LEVEL].legend(
            keep_lines,
            keep_labels,
            title="Iteration",
            bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
            loc=3,
            ncol=4,
            mode=None,
            borderaxespad=0.0,
            fontsize="xx-small",
        )

    if plot_shape:
        logger.info(
            """
                                            ##########################
                                            # SHAPE PLOT
                                            ##########################"""
        )
        cmap = plot_shape if type(plot_shape) is str else None
        f_shape, ax_shape, annotations = plot_shape_values(
            neuron, df_il, inhib_n_loc_insert_actual_short, axes1_color_cycle, cmap=cmap
        )
        # remove x and y
        if neuron.kcc2_inserted and (
            inhib_n_loc_insert_actual_short[0][1] < 0.01
            or inhib_n_loc_insert_actual_short[0][1] > 0.99
        ):
            plot_shape_values(
                neuron,
                df_ecl.T,
                inhib_n_loc_insert_actual_short,
                axes1_color_cycle,
                cmap=plt.get_cmap("winter"),
            )
        adjust_spines(ax_shape, [])
        ax_shape.set_title(iter_label, fontsize="x-small")
        f_shape.tight_layout()
        named_axes[f"SHAPE_{iter_label}_AX"] = ax_shape
        named_axes[
            f"SHAPE_{iter_label}_ANNOTATIONS"
        ] = annotations  # store args in memory for easy re-plotting
        axes.append(ax_shape)

    # fig.tight_layout()

    # ADD SCALEBARS
    # if settings.Vm in named_axes:
    #     use_scalebar(ax=named_axes[settings.Vm],
    #             matchx=True, matchy=True, hidex=True, hidey=True,
    #             loc=5,
    #             labelx=" ".join([settings.TIME, settings.UNITS(settings.ms)]),
    #             labely=" ".join([settings.MEMBRANE_POTENTIAL, settings.UNITS(settings.mV)]),
    #     )
    # use_scalebar(ax=named_axes[settings.INHIB_LEVEL],
    #         matchx=True, matchy=True, hidex=True, hidey=True,
    #
    #         loc=4,
    #         labelx="Location ({})".format(x_units) if x_units != settings.um
    #                 else "Distance from junction ({})".format(settings.um),
    #         labely=settings.INHIB_LEVEL
    # )
    # df_il.index = alt_index
    return df_il, df_distance, fig, named_axes


def plot_shape_values(
    neuron,
    df,
    inhib_n_loc_insert_actual_short,
    axes1_color_cycle,
    ax=None,
    cmap=None,
    show_soma=False,
    t_point=-1,
):
    from utils.plot_utils import shapeplot2d, mark_locations2d
    from PyNeuronToolbox.morphology import get_section_path, interpolate_jagged

    if ax is None:
        f_shape, ax_shape = plt.subplots(1, 1, figsize=(2, 2))
    else:
        f_shape = ax.figure
        ax_shape = ax
    plot_sections = neuron.sections
    if not show_soma:
        for sec in plot_sections:
            if sec.name() == "soma":
                plot_sections.remove(sec)
                break
    sec_names = [sec.name() for sec in neuron.sections]

    if t_point == -1 or type(t_point) is int:
        df = df.iloc[:, t_point]
    else:
        df = df.loc[:, t_point]
    # put DataFrame values in same order as sections list for plotting method
    #   weird double transpose as ordering seems to only work for columns
    try:
        select_il = df.T[sec_names].T
    except KeyError as ke:
        logger.warning(
            f"KeyError: {ke} was raised, suppressing and adjusting. If the shapeplot looks strange, "
            f"run again without `sections` arg"
        )
        plot_sections = (
            neuron.radial_dends
        )  # skip the soma so il_values are repeated evenly
        select_il = df.T[["radial_dends_1"]].T
    # remove 0.0 and 1.0 as these were added in with `dummy_seg`
    _zero = select_il.xs(0.0, level=1, drop_level=False)
    _one = select_il.xs(1.0, level=1, drop_level=False)
    select_il = select_il.loc[
        ~select_il.index.isin(_zero.index) & ~select_il.index.isin(_one.index)
    ]
    il_values = select_il.values.flatten()
    if cmap:
        shapeplot2d(h, ax_shape, sections=plot_sections, cvals=il_values, cmap=cmap)
    else:
        shapeplot2d(h, ax_shape, sections=plot_sections, cvals=il_values)
    # annotate inhib placement
    n_xy = {}
    annotations = []
    x_off = 0
    for s_i, sec in enumerate(plot_sections):
        if "radial_dends_2" in sec_names:
            x_off = 2.0 * (len(plot_sections) - 1 - s_i)
        xyz = get_section_path(h, sec)
        seg_paths = interpolate_jagged(xyz, sec.nseg)
        for j, seg in enumerate(sec):
            x = round(seg.x, 5)
            if (sec.name(), x) in inhib_n_loc_insert_actual_short:
                path = seg_paths[j]
                if x < 0.5:
                    xy = [path[0, 0], path[0, 1]]
                else:
                    xy = [path[-1, 0], path[-1, 1]]
                n_xy = inhib_n_loc_insert_actual_short.count((sec.name(), x))
                neg, dx, dy = -1, 0, 0
                for xy_i in range(n_xy):
                    neg *= -1
                    if xy_i % 2 == 1:
                        dx = 4 * (xy_i + 1) / 2
                        dy = 4 * (xy_i + 1) / 2
                    xytext = (0 + neg * dx, 6 - dy)
                    # mark_locations2d(h, sec, x, ax=ax_shape, markspec='v',
                    #                  markerfacecolor=opacity(100, axes1_color_cycle[0]), markeredgecolor='k',
                    #                  markersize=4,
                    #                  )
                    annotation = dict(
                        text="",
                        xy=xy,
                        fontsize="medium",
                        xytext=xytext,
                        textcoords="offset points",
                        zorder=99,
                        arrowprops=dict(
                            arrowstyle="-|>",
                            connectionstyle="arc3,rad=0",
                            facecolor=opacity(100, axes1_color_cycle[0]),
                            ec="k",
                            lw=0.8,
                        ),
                    )
                    ax_shape.annotate(**annotation)
                    annotations.append(annotation)
    return f_shape, ax_shape, annotations
