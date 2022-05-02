# coding=utf-8
""""
Main file for running Inhibitory Level simulations.
"""
from __future__ import print_function, division

import logging

from matplotlib import pyplot as plt
import sys
import yaml
import shared
from figures.dimensions import cl_radius_length, radius_length
from figures.optimal_location import figure_optimal_loc

try:
    from utils import settings
except ModuleNotFoundError:
    sys.path.insert(0, "../../utils")
    from utils import settings
from inhib_level import parse
from figures import (
    figure_basic,
    figure_input_structure_eff,
    figure_input_structure_loc_dist,
    figure_dynamic_il_time,
    figure_dynamic_il_loc,
    figure_explain,
    figure_sink,
)
from shared import INIT

logger = logging.getLogger("run_inhib_level")
shape_window = None


def parse_args(cmd_args=None):
    """Argument Parser for the spatiotemporal program"""

    def loc_check(loc):
        """Check that value is a float in range [0,1]"""
        value = float(loc)
        if not 0 <= value <= 1:
            msg = "{} must be in the range [0,1]".format(loc)
            raise argparse.ArgumentTypeError(msg)
        return value

    import argparse

    parser = argparse.ArgumentParser(
        description="""
        Python program for running Inhibitory Level simulations.
        
        Note that locations are specified in NEURON section units 'X'∈ [0,1]. In some cases (such as --hotspot), 
        '0' and '1' are not valid locations because synapses with ion dependencies (e.g. GABAa) cannot be placed there.
        """
    )
    parser.add_argument(
        "-v", dest="debug", action="store_true", help="show debug output"
    )
    group = parser.add_argument_group("Simulation type").add_mutually_exclusive_group()
    group.add_argument(
        "--compare",
        dest="compare",
        action="store_true",
        help="compare analytical vs synapses",
    )
    group.add_argument(
        "--hotspot",
        nargs="*",
        metavar="X",
        default=None,
        help="locations for excitatory synapses",
    )
    group.add_argument(
        "--radial",
        nargs="*",
        metavar="N",
        default=None,
        help='radial inhibitory level for "N" number of dendrites',
    )

    e_args = parser.add_argument_group("Reversal potential of inhibition")
    e_args.add_argument(
        "--e_offsets",
        nargs="*",
        metavar="E",
        default=None,
        help="offset of EGABA from Vrest",
    )
    e_args.add_argument(
        "--kcc2",
        nargs="?",
        const="compare",
        default="N",
        choices=["Y", "N", "C"],
        help="""using dynamic EGABA based on chloride dynamics, with bicarbonate ions. 
                        Yes (Y), No (N), or Compare (C)
                        [N]""",
    )

    radial_args = parser.add_argument_group("Radial")
    radial_args.add_argument(
        "--loc",
        nargs="*",
        metavar="X",
        default=None,
        help="locations for inhibitory synapses",
    )
    radial_args.add_argument(
        "--synapse_dists",
        nargs="*",
        metavar="D",
        default=None,
        choices=["diffused", "clustered", "clustered_n", "diffused_matched"],
        help="""how synapses should be distributed in the neuron.
                             *diffused_matched* - loc list used on every dendrite (default)
                             *diffused* - place a synapse at locations in loc on each dendrite according to 
                             `num_dendrites`.
                                         when len(loc) == num_dendrites, then each dendrite will have 1 synapse
                             *clustered* - place all synapses in loc on a single dendrite (try having all values in 
                             loc be the same!)
                             *clustered_n* - place all synapses on a single dendrite, evenly distributed (ignores 
                             value of loc)
                             """,
    )
    radial_args.add_argument(
        "--nseg", type=int, default=None, help="number of segments per branch"
    )

    morph_group = parser.add_argument_group("Morphology")
    morph_group.add_argument(
        "--diams",
        nargs="*",
        metavar="d",
        default=[settings.DIAM],
        help=f"change diameter (default is {settings.DIAM} um), "
        "electrotonic distance will stay constant by changing L",
    )
    morph_group.add_argument(
        "--constant_L",
        type=float,
        nargs="?",
        const=True,
        default=False,
        metavar="L",
        help="keep L constant (first value of diam if no arg provided), "
        "and hence electrotonic distance will vary with diameter",
    )
    morph_group.add_argument(
        "--sink",
        type=str,
        default=None,
        help="add a sink to the soma, which doesn't have any input and can have its properties varied. "
        "Notably, keyword arguments are: l, diam, num, and nseg",
    )

    time_group = parser.add_argument_group("Timing")
    time_group.add_argument(
        "--tstop",
        type=float,
        metavar="T",
        default=settings.TSTOP,
        help="length of simulation (in ms)",
    )
    time_group.add_argument(
        "--tm",
        type=float,
        default=0,
        help="time integration windows (default is tstop)",
    )

    quick_group = time_group.add_mutually_exclusive_group()
    quick_group.add_argument(
        "--quick",
        action="store_true",
        default=settings.NEURON_QUICK_DEFAULT,
        help="quick simulations, less precise "
        "(note that default is True but tm<tstop changes to precise simulations)",
    )
    quick_group.add_argument(
        "--precise",
        dest="quick",
        action="store_false",
        help="long simulations, more precise",
    )

    loc_group = parser.add_argument_group("DVV location")
    loc_group.add_argument(
        "--sections",
        type=str or int,
        nargs="*",
        metavar="sec",
        default=None,
        help="which sections (indexes or actual name) to use for the IL calculation",
    )
    loc_group.add_argument(
        "--segments",
        nargs="*",
        metavar="X",
        default=None,
        help='which segments in "X" [0;1] to use for the IL calculation',
    )

    syn_group = parser.add_argument_group(
        "Synapses",
        description="""
                        Parameters for investigating frequency-dependent synapses (GABAa) instead of the default 
                        fluctuating conductance synapses (inhfluct). 
                        """,
    )
    syn_group.add_argument(
        "--hz",
        type=float,
        nargs="*",
        default=[0],
        help="frequency of synapses (sets synapse type to GABAa)",
    )
    syn_group.add_argument(
        "--offset",
        type=float,
        nargs="*",
        metavar="t",
        default=[0],
        help="timing offset between each GABAa synapse (in ms)",
    )
    syn_group.add_argument(
        "--noise",
        type=loc_check,
        nargs="*",
        metavar="ε",
        default=[0],
        help="noise of the synapses ε ∈ [0,1]",
    )

    figure_args = parser.add_argument_group("Figure")
    figure_args.add_argument(
        "--plot_group_by",
        type=str,
        default=None,
        choices=["e_offsets", "num_dendrites", "num_dendrites_arr", "False"],
        help="""how to group plots""",
    )
    figure_args.add_argument(
        "--plot_color_by",
        type=str,
        default="num_synapses",
        choices=["num_synapses", "e_offsets", "kcc2", "t_offset", "synapse_dist"],
        help="""How to differentially color the graphs. Default is color by number of synapses.
                             """,
    )
    figure_args.add_argument(
        "--save-fig",
        dest="save_fig",
        action="store_true",
        default=False,
        help='save the figure in "output/"',
    )

    trace_group = parser.add_argument_group("Traces").add_mutually_exclusive_group()
    trace_group.add_argument(
        "--with-v-trace", dest="v_trace", action="store_true", help="show voltage trace"
    )
    trace_group.add_argument(
        "--with-t-trace",
        dest="t_trace",
        type=loc_check,
        nargs="?",
        const=0.2,
        metavar="X",
        default=None,
        help='trace of IL over time windows "tm" at given location "X" (e.g. 0.2)',
    )
    parser.add_argument(
        "--plot_shape",
        type=str,
        nargs="?",
        const=True,
        default=False,
        metavar="cmap",
        help="show a ShapePlot with IL heatmap.",
    )
    if cmd_args:
        # explicitly parsed
        return parser.parse_args(cmd_args.split())
    # use sys.argv
    return parser.parse_args()


def run_inhib_level(cmd_args=None, **kwargs):
    """Main method to initialise, run, and display Inhibitory Level simulations."""
    logger.info("run_inhib_level")
    INIT()
    args = parse_args(cmd_args)

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.debug("args = {}".format(args))

    result = [{}, "", {}]  # [plot_dict, func_name, saved_args]

    if args.t_trace is not None and not 0 < args.tm < args.tstop:
        raise Exception("t_trace requires a better tm")

    timing = {"hz": args.hz, "offset": args.offset, "noise": args.noise}

    if args.segments is not None:
        segs = []
        for loc in args.segments:
            if type(loc) is str:
                loc = loc.replace(",", "").replace("[", "").replace("]", "")
            loc = float(loc)
            segs.append(loc)
        args.segments = segs
    dvv_kwargs = {"sections": args.sections, "select_segs": args.segments}

    if args.sink is not None:
        # add back spaces
        sink = yaml.safe_load(args.sink.replace(":", ": ").replace(",", ", "))
    else:
        sink=None

    if args.compare:
        result = figure_explain()
    else:
        result = parse.parse_and_run(
            hotspots=args.hotspot,
            num_dendrites_arr=args.radial,
            loc=args.loc,
            synapse_dists=args.synapse_dists,
            e_offsets=args.e_offsets,
            diams=args.diams,
            constant_L=args.constant_L,
            sink=sink,
            plot_group_by=args.plot_group_by,
            voltage=args.v_trace,
            time=args.t_trace,
            kcc2=args.kcc2,
            tstop=args.tstop,
            tm=args.tm,
            quick=args.quick,
            timing=timing,
            plot_color_by=args.plot_color_by,
            dvv_kwargs=dvv_kwargs,
            plot_shape=args.plot_shape,
            nseg=args.nseg,
        )

    if settings.NEURON_GUI:
        from neuron import h

        # PLOT SHAPE (REMOVES PREVIOUS ONES)
        shape_window = h.PlotShape()
        shape_window.exec_menu("Show Diam")
        shape_window.exec_menu("Shape Plot")
        shape_window.exec_menu("View = plot")
        # """'baseattr', 'begin', 'beginline', 'cas', 'color', 'color_all', 'color_list', 'colormap', 'erase',
        # 'erase_all',
        # 'exec_menu', 'fastflush', 'flush', 'gif', 'hinton', 'hname', 'hocobjptr', 'label', 'len_scale', 'line',
        # 'mark',
        # 'menu_action', 'menu_tool', 'nearest', 'next', 'observe', 'printfile', 'push_selected', 'ref', 'rotate',
        # 'same',
        # 'save_name', 'scale', 'setpointer', 'show', 'size', 'unmap', 'variable', 'view', 'view_count' """
        shape_window.variable("v")
        shape_window.scale(-80, -40)
        h.topology()
    plot_dict, func_name, saved_args = result
    if args.save_fig:
        from utils.plot_utils import save_fig

        save_fig(plot_dict, cmd_args)
    logger.debug("DONE: run_inhib_level {}".format(cmd_args))
    return result


def examples():
    """Examples"""
    #######################################################################################
    # Basic ones (num dendrites == num synapses)
    #######################################################################################
    run_inhib_level("--radial 1 2 4 8 16")  # from Gidon & Segev 2012
    run_inhib_level("--radial 4 --e_offsets 0 -5 -10")
    run_inhib_level("--radial 4 8 --e_offsets -1 0 1")
    run_inhib_level("--radial 2 5 10 --e_offsets -2 -1 0 1 2")

    #######################################################################################
    # Hotspot
    #######################################################################################
    # place hotspot at 0.6
    run_inhib_level("--hotspot 0.6")
    # compare hotspot locations
    run_inhib_level("--hotspot 0.6 0.8")
    # compare reversals
    run_inhib_level("--hotspot 0.6 --e_offsets 0 -5")

    #######################################################################################
    # Effect of offsets and number of dendrites (num dendrites == num synapses)
    #######################################################################################
    # all on one plot
    run_inhib_level("--radial 1 4 8 --e_offsets 0 -0.5 -1 -2 -5 -10")
    # group by num_dendrites to compare offsets
    run_inhib_level(
        "--radial 1 4 8 --e_offsets 0 -0.5 -1 -2 -5 -10 --plot_group_by num_dendrites_arr"
    )
    # group by offsets to compare dendrites
    run_inhib_level(
        "--radial 1 4 8 --e_offsets 0 -0.5 -1 -2 -5 -10 --plot_group_by e_offsets"
    )
    # normalised
    run_inhib_level(
        "--radial 1 4 8 --e_offsets 0 -1 -2 -5 --loc 0.2 --synapse_dists diffused --plot_group_by e_offsets_norm"
    )

    #######################################################################################
    # Effective branches - some branches will NOT have synapses
    #######################################################################################
    #
    # # 1/1, 16/4, 64/8
    # locs = ["[{}]*{}".format(loc, l) for l in [base, 4, 8]]
    # run_inhib_level('--radial 1 4 8 --e_offsets 0 -1 --loc {} --synapse_dists diffused'.format(" ".join(locs)))

    # 4/4, 4/6, 4/8
    base = 4
    loc = ["0.2"] * base
    radials = [str(int(b)) for b in [base, base * 1.5, base * 2]]
    run_inhib_level(
        "--radial {} --e_offsets 0 --loc {} --synapse_dists diffused".format(
            " ".join(radials), " ".join(loc)
        )
    )

    # 6/6, 6/9, 6/12
    base = 6
    loc = ["0.2"] * base
    radials = [str(int(b)) for b in [base, base * 1.5, base * 2]]
    run_inhib_level(
        "--radial {} --e_offsets 0 --loc {} --synapse_dists diffused".format(
            " ".join(radials), " ".join(loc)
        )
    )

    # 4/8, 6/8, 8/8
    base = 4
    loc = "0.2"
    locs = ["[{}]*{}".format(loc, l) for l in [base, base * 1.5, base * 2]]
    run_inhib_level(
        "--radial {} --e_offsets 0 -1 --loc {} --synapse_dists diffused --plot_group_by e_offsets".format(
            base * 2, " ".join(locs)
        )
    )
    # # 1/8 4/8, 8/8 16/8
    base = 8
    loc = "0.2"
    locs = ["[{}]*{}".format(loc, l) for l in [1, int(base / 2), base, base * 2]]
    run_inhib_level(
        "--radial {} --e_offsets 0 -1 --loc {} --synapse_dists diffused --plot_group_by e_offsets".format(
            base, " ".join(locs)
        )
    )

    # 1/1, 1/4, 1/8
    #   EFFECT OF OFFSETS when there is only ever ONE synapse
    # 1 synapse with multiple dendrites needs to be treated slightly differently (clustered)
    run_inhib_level(
        "--radial 1 4 8 --e_offsets 0 -1 -2 --loc 0.2 --synapse_dists clustered --plot_group_by e_offsets"
    )
    # 1/1, 4/4, 8/8 (num dendrites == num synapses)
    run_inhib_level(
        "--radial 1 4 8 --e_offsets 0 -1 -2 --loc 0.2 --synapse_dists diffused_matched --plot_group_by e_offsets"
    )

    #######################################################################################
    # Clustered vs Diffused
    #######################################################################################
    base = 4
    loc = ["0.2"] * base
    # diffused synapses
    run_inhib_level(
        "--radial {} --e_offsets 0 --loc {} --synapse_dists diffused".format(
            base, " ".join(loc)
        )
    )
    # clustered synapses all at one loc
    run_inhib_level(
        "--radial {} --e_offsets 0 --loc {} --synapse_dists clustered".format(
            base, " ".join(loc)
        )
    )
    # clustered synapses evenly distributed
    run_inhib_level(
        "--radial {} --e_offsets 0 --loc {} --synapse_dists clustered".format(
            base, base
        )
    )
    # equivalent to
    run_inhib_level(
        "--radial {} --e_offsets 0 --loc {} --synapse_dists clustered_n".format(
            base, " ".join(loc)
        )
    )

    # all of the above in one
    run_inhib_level(
        "--radial {} --e_offsets 0 --loc {} "
        "--synapse_dists diffused clustered clustered_n".format(base, " ".join(loc))
    )

    # compare clustering based on num dendrites and offsets
    for base in [4, 8]:
        for offset in "0 -1 -2".split():
            run_inhib_level(
                "--radial {}  --e_offsets {} --loc {} "
                "--synapse_dists diffused clustered clustered_n".format(
                    base, offset, " ".join(["0.2"] * base)
                )
            )

    #######################################################################################
    # Timing
    #######################################################################################
    # add in time plot
    run_inhib_level("--radial 1 4 --tm=5 --with-t-trace")
    # time plot at location 0.3 (in X units)
    run_inhib_level("--radial 1 4 --tm=5 --with-t-trace 0.3")
    # timing for different offsets
    run_inhib_level("--radial 1 4 --e_offsets 0 -5 --tm=5 --with-t-trace")

    # IL over time at hotspot (0.6)
    run_inhib_level("--hotspot 0.6 --e_offsets 0 -5 --tm=5 --with-t-trace 0.6")

    # use GABAA synapses
    run_inhib_level(
        "--radial 1 2 4 8 16 --hz 20 --tstop 500"
    )  # may be useful to increase time of simulation
    # set synapses to be the most noisy
    run_inhib_level("--radial 1 4 --hz 100 --noise 1")
    # predictable input, with each synapse offset by 20 ms
    run_inhib_level("--radial 1 4 --hz 100 --offset 20")
    run_inhib_level("--radial 1 4 --hz 100 --offset 20 --noise 1")

    # different reversal offset, length of simulation, integration time window. Display over time.
    run_inhib_level(
        "--radial 1 4 --e_offsets -2 --hz 100             --noise 1 --tm=50 --tstop=500 --with-t-trace"
    )
    run_inhib_level(
        "--radial 1 4 --e_offsets -2 --hz 100 --offset 20           --tm=50 --tstop=500 --with-t-trace"
    )
    run_inhib_level(
        "--radial 1 4 --e_offsets -2 --hz 100 --offset 20 --noise 1 --tm=50 --tstop=500 --with-t-trace"
    )

    #######################################################################################
    # Chloride
    #######################################################################################
    # Over time
    run_inhib_level(
        "--radial 1 --e_offsets -2 --synapse_dists diffused_matched --kcc2 Y --tm=10 --with-t-trace"
    )
    # Over time, comparing with and without chloride dynamics
    run_inhib_level(
        "--radial 1 --e_offsets -2 --synapse_dists diffused_matched --kcc2 C --tm=10 --with-t-trace"
    )


if __name__ == "__main__":
    import time

    start = time.time()

    if len(sys.argv) > 1:
        run_inhib_level()  # will get arguments from sys.argv automatically
    else:
        figure_explain()
        figure_basic()
        figure_input_structure_eff()
        figure_input_structure_loc_dist()
        figure_dynamic_il_time()
        figure_dynamic_il_loc()
        figure_optimal_loc()
        # radius_length()
        figure_sink()
        cl_radius_length(diam=1, sample_N=4)
        # cl_radius_length(diam=1, sample_N=1)
    logger.info(
        "COMPLETED\n{}\n** run_inhib_level took {:.2f}s\n{}".format(
            "*" * 50, time.time() - start, "*" * 50
        )
    )
    if settings.SAVE_FIGURES:
        shared.show_n(6)
    plt.show()
