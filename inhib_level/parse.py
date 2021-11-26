import ast
from typing import Tuple

import numpy as np

from utils import settings
import shared
from inhib_level.math import lambda_d, offset_index
from inhib_level.plot import create_figure
from inhib_level.simulation import run_sim
from model.morphology import SingleDend, MultiDend

import logging

logger = logging.getLogger("parse")


def parse_and_run(
    hotspots: list = None,
    num_dendrites_arr: list = None,
    loc: list = None,
    synapse_dists: list = None,
    e_offsets: list = None,
    diams: list = None,
    constant_L: bool = False,
    voltage: bool = False,
    time: bool = None,
    plot_group_by: str = None,
    kcc2: str = None,
    tstop: float = 100.0,
    tm: float = 0.0,
    timing: dict = None,
    quick: bool = True,
    plot_color_by: str = None,
    dvv_kwargs: dict = None,
    plot_shape: bool = False,
    nseg=None,
) -> Tuple[dict, str, dict]:
    """
    Run simulation to calculate IL using N number of dendrites. Multiple Ns can be specified to compare effects.
    :param hotspots: Calculate IL at hotspot. Multiple locations can be specified.
    :param num_dendrites_arr: Number of dendrites
    :param loc: Locations to place synapses (can be list of lists). Inner strings are converted to floats or lists.
    :param synapse_dists: Distribution of synapses
    :param e_offsets: Reversal potential of EGABA, offset from Vrest.
    :param diams: Change diameter, electrotonic distance will stay constant by changing L
    :param constant_L: keep L constant, electrotonic distance will vary with diameter
    :param voltage: Show voltage plot
    :param time: Show time plot
    :param plot_group_by: Group traces by 'num_dendrites_arr' or 'e_offsets' instead of plotting everything on a
    single axis
    :param kcc2: Run simulation with chloride dynamics ('Y') or static chloride ('N). 'C' compares both.
    :param tstop: Time to run each NEURON simulation for (to calculate a single ILd or ILh).
    :param tm: Time integration window, to see progression of IL over time. 0 is integrate entire tstop window.
    :param timing: Dict of 'hz', 'offset' and 'noise' to use GABAa synapses and adjust their relative timings.
    :param quick: Perform a quick (True) or a longer, more precise with constant dt (False), NEURON simulation
    :param plot_color_by: Provide which rule to determine line coloring
    :param dvv_kwargs: Arguments to be provided to `inhibi_level.math.dvv`
    :param plot_shape: Plot shape of the neuron (False)
    :param nseg: Number of segments per section of neuron dendrite (default is 8*LENGTH + 1; minimum of 89)
    :return: plot_dict, func_name, saved_args
    """
    shared.INIT()
    if hotspots is None:
        hotspots = [None]  # empty
    elif hotspots:
        if "[" in hotspots[0]:
            # loc is list of lists
            for i, l in enumerate(hotspots):
                if "*" in l:
                    star_idx = l.find("*")
                    hotspots[i] = ast.literal_eval(l[:star_idx]) * int(
                        float(l[star_idx + 1 :])
                    )
                else:
                    hotspots[i] = ast.literal_eval(l)
    if num_dendrites_arr is None:
        num_dendrites_arr = [4]
    if loc is None:
        # place inhibition at 0.2 on each dendrite
        # loc = ["[{}]*{}".format(0.2, n) for n in num_dendrites_arr]
        loc = ["0.2"]
    if synapse_dists is None:
        synapse_dists = ["diffused_matched"]
    if e_offsets is None:
        e_offsets = [0]
    if diams is None:
        diams = [settings.DIAM]
    else:
        diams = [float(d) for d in diams]

    il_dict = {}
    ecl_dict = {}
    saved_args = locals()
    func_name = "hotspot" if hotspots[0] else "radial"
    plot_dict = {}

    # SYNAPSES
    if type(synapse_dists) is not list:
        synapse_dists = [synapse_dists]

    if "[" in loc[0]:
        # loc is list of lists
        for i, l in enumerate(loc):
            if "*" in l:
                star_idx = l.find("*")
                loc[i] = ast.literal_eval(l[:star_idx]) * int(float(l[star_idx + 1 :]))
            else:
                loc[i] = ast.literal_eval(l)
    else:
        loc = [loc]

    logger.debug("number of different locs is '{}'".format(len(loc)))
    logger.debug("synapse distribution is '{}'".format(synapse_dists))
    inhib_syn_type = "inhfluct"
    inh_netstim_args = dict(
        hz=20,
        start=0,
        noise=0,
        duration=tstop,  # defaults to tstop
        weight=1,
        n_sources=0,
        own_stream=False,
    )
    if timing is not None:
        assert type(timing) is dict
        inhib_syn_type = "GABAa"
        logger.debug("synapse is now of type 'GABAa'")
        inh_netstim_args["hz"] = timing["hz"]
        inh_netstim_args["noise"] = timing["noise"]

    exc_syn_type = "IClamp"
    total_exc_synapses = 1
    exc_synapse_location = 0.6
    exc_netstim_args = dict(
        hz=100,
        start=0,
        noise=1,
        duration=None,  # defaults to tstop
        weight=1,
        n_sources=0,
        own_stream=False,
    )
    # NEURON
    x_units = settings.X_UNITS
    Rm = 20000
    Ra = 100  # default
    neuron_kwargs = dict(axon=False, call_geom_nseg=False, Rm=Rm, Ra=Ra,)

    kcc2_options = []
    if kcc2.lower() in ["no", "n", "compare", "c", "both", "b"]:
        kcc2_options.append(False)
    if kcc2.lower() in ["only", "yes", "y", "compare", "c", "both", "b"]:
        kcc2_options.append(True)

    len_k = len(kcc2_options)
    len_h = len(hotspots)
    len_n = len(num_dendrites_arr)
    len_l = len(loc)
    len_e = len(e_offsets)
    len_d = len(diams)
    len_s = len(synapse_dists)
    len_hz = len(timing["hz"])
    len_off = len(timing["offset"])
    len_noise = len(timing["noise"])

    fig, named_axes = None, None

    sequential_cmaps = (
        settings.cmaps["Sequential"]
        + settings.cmaps["Sequential (2)"]
        + settings.cmaps["Perceptually " "Uniform " "Sequential"]
    )
    if len_k * len_h * len_n * len_e * len_s * len_hz * len_off * len_noise > len(
        sequential_cmaps
    ):
        replace = True
    else:
        replace = False
    cmaps = np.random.choice(
        sequential_cmaps,
        size=(len_k, len_h, len_n, len_e, len_s, len_hz, len_off, len_noise),
        replace=replace,
    )

    dict_key = 0
    if len(plot_dict) == 0:
        if plot_group_by == "e_offsets":
            for e, e_offset in enumerate(e_offsets):
                plot_dict[e_offset] = create_figure(
                    new_once=True, voltage=voltage, attenuation=False, time=time
                )
        elif plot_group_by == "num_dendrites_arr" or plot_group_by == "num_dendrites":
            for n, num_dendrites in enumerate(num_dendrites_arr):
                plot_dict[num_dendrites] = create_figure(
                    new_once=True, voltage=voltage, attenuation=False, time=time
                )
        elif plot_group_by == "False":
            plot_dict[dict_key] = False
        else:
            plot_dict[dict_key] = create_figure(
                new_once=True, voltage=voltage, attenuation=False, time=time
            )
    # nested for loops (variable lengths low enough to be done without needing efficient itertools)
    arg_product = (
        (hot, k, n, l, e, d, s, hz_i, off_i, noise_i)
        for hot in range(len_h)
        for k in range(len_k)
        for n in range(len_n)
        for l in range(len_l)
        for e in range(len_e)
        for d in range(len_d)
        for s in range(len_s)
        for hz_i in range(len_hz)
        for off_i in range(len_off)
        for noise_i in range(len_noise)
    )
    for (hot, k, n, l, e, d, s, hz_i, off_i, noise_i) in arg_product:
        hot_loc = hotspots[hot]
        at_hotspot: bool = hot_loc is not None
        kcc2_option = kcc2_options[k]
        num_dendrites = num_dendrites_arr[n]
        sub_loc = loc[l]
        e_offset = e_offsets[e]
        diam = diams[d]
        synapse_dist = synapse_dists[s]
        hz = timing["hz"][hz_i]
        t_offset = timing["offset"][off_i]
        noise = timing["noise"][noise_i]

        if plot_group_by == "e_offsets":
            dict_key = e_offset
        elif plot_group_by == "num_dendrites_arr" or plot_group_by == "num_dendrites":
            dict_key = num_dendrites
        else:
            dict_key = 0
        plot = plot_dict[dict_key]

        e_offset_num = float(e_offset)

        if type(constant_L) is float:
            length = constant_L
        else:
            length = (
                lambda_d(diams[0], Rm, Ra) if constant_L else lambda_d(diam, Rm, Ra)
            )

        logger.debug(neuron_kwargs)

        if at_hotspot:
            inhib_n_loc_insert = None
            exc_n_loc_insert = [float(hot_loc)]
            iter_label = "h={}".format(hot_loc)
            num_syn = 1
            neuron_class = SingleDend
            neuron_kw_pre = "dend"
        else:
            neuron_class = MultiDend
            neuron_kw_pre = "radial"
            neuron_kwargs["num_dendrites"] = int(num_dendrites)
            inhib_n_loc_insert = _calc_inhib_loc(
                neuron_kwargs["num_dendrites"], sub_loc, str(synapse_dist)
            )
            logger.debug("inhib_n_loc_insert={}".format(inhib_n_loc_insert))
            exc_n_loc_insert = None
            sub_loc_len = len(sub_loc)
            num_syn = 0
            if type(inhib_n_loc_insert[0][1]) is int:
                for loc_insert in inhib_n_loc_insert:
                    num_syn += loc_insert[1]
            else:
                for loc_insert in inhib_n_loc_insert:
                    num_syn += len(loc_insert[1])
                if type(inhib_n_loc_insert[0]) is float:
                    num_syn = neuron_kwargs["num_dendrites"]
            iter_label = "n={:g}/{:g}\n{:s}".format(
                num_syn, neuron_kwargs["num_dendrites"], synapse_dist
            )
            iter_label += f" x={sub_loc}"

        neuron_kwargs[f"{neuron_kw_pre}_l"] = length
        neuron_kwargs[f"{neuron_kw_pre}_diam"] = diam
        neuron_kwargs[f"{neuron_kw_pre}_nseg"] = (
            2 * int(neuron_kwargs[f"{neuron_kw_pre}_l"] / 16) + 1
        )
        if neuron_kwargs[f"{neuron_kw_pre}_nseg"] < 89:
            neuron_kwargs[f"{neuron_kw_pre}_nseg"] = 89
        neuron_kwargs[f"{neuron_kw_pre}_nseg"] = (
            nseg or neuron_kwargs[f"{neuron_kw_pre}_nseg"]
        )  # assign from arg
        neuron_kwargs["add_kcc2"] = kcc2_option

        show_input = False

        # inhib input type
        inh_netstim_args_iter = inh_netstim_args
        if hz > 0:
            inhib_syn_type = "GABAa"
            quick = False
            logger.debug("synapse is now of type 'GABAa'")
            inh_netstim_args["hz"] = hz
            inh_netstim_args["noise"] = noise
            if t_offset > 0:
                # create dict of netstim args for each offset such that they start 'offset' time apart
                inh_netstim_args_iter = dict()
                for _s_i in range(num_syn):
                    inh_netstim_args_iter["t{}".format(_s_i)] = dict(
                        inh_netstim_args, start=t_offset * _s_i
                    )
                logger.debug(
                    "multiple offsets used. inh_netstim_args_iter = {}".format(
                        inh_netstim_args_iter
                    )
                )
                show_input = True
        else:
            inhib_syn_type = "inhfluct"
            logger.debug("synapse is now of type 'inhfluct'")
        iter_label += "\ne={:.1f}({})".format(
            e_offset_num, settings.DELTA if kcc2_option else "-"
        )
        iter_label += f"\nd={diam:.1f} l={length:.2f}"
        logger.info("running {}".format(iter_label.replace("\n", "\t")))

        # choose cmap
        cmap = None
        if plot_color_by is not None:
            if plot_color_by in settings.cmap_dict:
                if plot_color_by == "num_synapses":
                    cmap = settings.cmap_dict[plot_color_by].get(num_syn, None)
                elif plot_color_by == "e_offsets":
                    cmap = settings.cmap_dict[plot_color_by].get(e_offset_num, None)
                elif plot_color_by == "kcc2":
                    cmap = settings.cmap_dict[plot_color_by].get(kcc2_option, None)
                elif plot_color_by == "t_offset":
                    cmap = settings.cmap_dict[plot_color_by].get(t_offset, None)
                elif plot_color_by == "synapse_dist":
                    cmap = settings.cmap_dict[plot_color_by].get(synapse_dist, None)
        if cmap is None:
            cmap = cmaps[k][hot][n][e][s][hz_i][off_i][noise_i] + "_r"

        il_sim = run_sim(
            plot,
            at_hotspot=at_hotspot,
            inhib_syn_type=inhib_syn_type,
            inhib_n_loc_insert=inhib_n_loc_insert,
            inh_netstim_args=inh_netstim_args_iter,
            e_offset=e_offset_num,
            exc_syn_type=exc_syn_type,
            exc_netstim_args=exc_netstim_args,
            exc_n_loc_insert=exc_n_loc_insert,
            colormap=cmap,
            neuron=neuron_class,
            neuron_args=neuron_kwargs,
            iter_label=iter_label,
            x_units=x_units,
            quick=quick,
            t=tstop,
            tm=tm,
            time=time,
            show_input=show_input,
            dvv_kwargs=dvv_kwargs,
            plot_shape=plot_shape,
        )
        df_il, df_units, fig, named_axes, df_ecl = il_sim
        il_dict[iter_label] = df_il
        ecl_dict[iter_label] = df_ecl
        if "units" not in il_dict or len(df_units.index.levels[0]) > len(
            il_dict["units"].index.levels[0]
        ):
            il_dict["units"] = df_units
        logger.info(
            "{:.2f}% | {:.2f}% | {:.2f}% | {:.2f}%".format(
                100 * (k + 1) / len_k,
                100 * (n + 1) / len_n,
                100 * (e + 1) / len_e,
                100 * (s + 1) / len_s,
            )
        )
    logger.debug(il_dict.keys())
    if plot_group_by == "e_offsets_norm":
        offset_dict = offset_index(il_dict, num_dendrites_arr)
        fig, ax, named_axes = create_figure(
            new_once=True, voltage=voltage, attenuation=False
        )

        for il_key, df_il in offset_dict.items():
            section_names = df_il.index.levels[0]
            # df_il.unstack().plot()
            for section_name in df_il.index.levels[0]:
                if section_name == "axon":
                    continue
                elif section_name == "soma":
                    label = il_key
                else:
                    label = None
                line = named_axes[settings.INHIB_LEVEL].plot(
                    il_dict["units"].loc[section_name, x_units],
                    df_il.loc[section_name],
                    linestyle="-",
                    label=label,
                )
        named_axes[settings.INHIB_LEVEL].legend(
            bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
            loc=4,
            ncol=4,
            mode=None,
            borderaxespad=0.0,
        )
        named_axes[settings.INHIB_LEVEL].set_ylabel(
            "Normalised {}".format(settings.INHIB_LEVEL)
        )
        named_axes[settings.INHIB_LEVEL].set_xlabel(
            "Distance from soma {}".format(settings.UNITS(settings.um))
        )
        fig.tight_layout()
        plot_dict["norm"] = (fig, ax, named_axes)
    return plot_dict, func_name, saved_args


def _calc_inhib_loc(num_dendrites: int, loc_xs: list, synapse_dist: str):
    """
    Determine the list syntax that `run_sim` wants from a list of dendrite numbers, locations,
    and dependent
    on the distribution of synapses
    :param num_dendrites: Number of dendrites
    :param loc_xs: List of locations in 'X' units
    :param synapse_dist: Distribution of synapses.
                        diffused_matched - loc list used on every dendrite
                        diffused - place a synapse at locations in loc on each dendrite according to `num_dendrites`.
                                    when len(loc) == num_dendrites, then each dendrite will have 1 synapse
                        clustered - place all synapses in loc on a single dendrite (try having all values in loc be
                                    the same!)
                        clustered_n - place all synapses on a single dendrite, evenly distributed (ignores value of loc)

    :return: list of tuples of ('section_name',[locations to place synapse])
    """
    if synapse_dist == "diffused_matched":
        # loc_xs used on every dendrite
        inhib_n_loc_insert = []
        for i in range(num_dendrites):
            inhib_n_loc_insert.append(
                ("radial_dends[{}]".format(i), [float(l) for l in loc_xs])
            )
    elif synapse_dist == "diffused":
        inhib_n_loc_insert = {}
        i = 0
        for loc_x in loc_xs:
            loc_x = float(loc_x)
            if loc_x > 1.0:
                raise Exception("'loc_xs' values be be E[0;1] for 'diffused")
            if i >= num_dendrites:
                # 'rollover'
                i = 0
            sec_name = "radial_dends[{}]".format(i)
            if sec_name in inhib_n_loc_insert:
                inhib_n_loc_insert[sec_name].append(loc_x)
            else:
                inhib_n_loc_insert[sec_name] = [loc_x]
            i += 1
        # convert from dict to list of tuples
        inhib_n_loc_insert = list(inhib_n_loc_insert.items())
    elif synapse_dist == "clustered":
        # note that if loc_xs is an integer, then synapses will be evenly distributed.
        # this is similar to clustered_n, but with the explicit value of loc_xs
        if len(loc_xs) == 1 and float(loc_xs[0]) == int(float(loc_xs[0])):
            inhib_n_loc_insert = [("radial_dends[0]", int(loc_xs[0]))]
        else:
            inhib_n_loc_insert = [("radial_dends[0]", [float(l) for l in loc_xs])]
    elif synapse_dist == "clustered_n":
        # shortcut method to ignore the values of loc_xs and just use the LENGTH of
        # loc_xs to evenly distribute the synapses
        inhib_n_loc_insert = [("radial_dends[0]", len(loc_xs))]
    else:
        raise Exception(
            "arg 'dendrite_dist' must be one of ['diffused','clustered','clustered_n','diffused_matched']"
        )
    return inhib_n_loc_insert
