# coding=utf-8
import logging
from typing import Tuple, Union

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from neuron import h

from inhib_level import math
from utils import settings
from inhib_level.plot import inhib_level_plot
from model.base import BaseNeuron
from model.morphology import SingleDend
from model.nrn_simulation import set_global_cli
from model.synapse import get_synapse_args

logger = logging.getLogger(__name__)


def add_synapse_type(
    neuron: BaseNeuron,
    syn_type: str,
    n_loc_insert: list,
    e: float,
    netstim_args: dict,
    gmax: float = 0.0,
    **kwargs
):
    # noinspection SpellCheckingInspection
    """Add synapses of type `syn_type` to `BaseNeuron` neuron object, given parameters.

        Most of the work is done by `BaseNeuron.add_n_synapses`, this is a wrapping function to

        - insert only a known syn_type (and ignore None `syn_type`)
        - retrieve default values of syn_type from `neuronsynapse.get_synapse_args`
        - insert synapses
        - call BaseNeuron.netstim method (with different netstim_args if there are timing offsets)
        - handle IClamp injection
        - get actual locations using **NEURON**'s get_loc method

        :param neuron: BaseNeuron to add synapses on
        :param syn_type: Type of synapse to add
        :param n_loc_insert: Locations to insert synapses
        :param e: Reversal potential of synapse
        :param netstim_args: Parameters for netstim objects to stimulate synapses with NET_RECEIVE NMODL block
        :param gmax: Max conductance
        :param kwargs: Other parameters to be passed to `neuronsynapse.get_synapse_args` to assign other parameters
                according to the synapse

        :return:  tuple of
                1) list of synapse dicts for each synapse,
                2) synapse locations in 'X' coords,
                3) synapses locations in list of tuples with fromat ('section name', location in X)
        """
    synapses = []  # list of dicts. see add_synapses in BaseNeuron
    synapse_locations = []
    n_loc_insert_actual_short = []
    if syn_type is not None:
        logger.info("placing {} on {}".format(syn_type, neuron.name))
        logger.debug("n_loc_insert = {}".format(n_loc_insert))
        if "std_i" not in kwargs:
            kwargs["std_i"] = 0
        syn_args = get_synapse_args(syn_type, gmax=gmax, e=e, **kwargs)

        syn_label_extra = ""
        if "t0" in netstim_args:
            # different sets of synapses will have different offsets
            for ((sections, n_or_locations), (key, netstim_args_tn)) in zip(
                n_loc_insert, netstim_args.items()
            ):
                new_syn = neuron.add_n_synapses(
                    syn_type,
                    sections=sections,
                    locations=n_or_locations,
                    syn_label_extra=key,
                    **syn_args
                )
                neuron.netstim(synapses=new_syn, **netstim_args_tn)
                synapses += new_syn
        else:
            for sections, n_or_locations in n_loc_insert:
                new_syn = neuron.add_n_synapses(
                    syn_type,
                    sections=sections,
                    locations=n_or_locations,
                    syn_label_extra=syn_label_extra,
                    **syn_args
                )
                synapses += new_syn
        # set random seeds for repeatable simulations (netstim method uses settings.RANDOM_SEED)
        if "fluct" in syn_type:
            for synapse in synapses:
                synapse["object"].new_seed(settings.RANDOM_SEED)
        elif "IClamp" in syn_type:
            iclamp = synapses[0]["object"]
            iclamp.delay = 0
            iclamp.amp = 1e-3  # nA
        else:
            if "t0" not in netstim_args:
                neuron.netstim(synapses=synapses, **netstim_args)

        for i, syn in enumerate(synapses):
            x = float("{:.5f}".format(syn["object"].get_loc()))
            h.pop_section()  # important after get_loc() to prevent stack overflow
            sec_name = syn["sec"].name()
            n_loc_insert_actual_short.append((sec_name, x))
            synapse_locations.append(x)

    return synapses, synapse_locations, n_loc_insert_actual_short


def run_sim(
    figure: Tuple[Figure, Axes] = None,  # Tuple of (fig,axes)
    at_hotspot=False,
    v_init=-65.0,  # mV
    t=100.0,  # ms
    tm=0.0,  # integration window (ms)
    g_i=0.001,  # uS
    e_offset=-0.0,  # v_init + e_offset
    g_e=0.001,
    inhib_syn_type="inhfluct",
    inh_netstim_args=None,
    inhib_n_loc_insert=None,
    exc_syn_type="IClamp",
    exc_netstim_args=None,
    exc_n_loc_insert=None,
    show_input=False,
    # v_sample_show_section=False,
    neuron: Union[type, BaseNeuron] = None,
    neuron_args: dict = None,
    colormap: str = None,
    iter_label: str = None,
    x_units: str = settings.um,  # settings.um or 'X'
    quick=True,
    time: bool = None,
    dvv_kwargs=None,
    plot_shape: bool = False,
):
    """Set up the neuron to run Inhibitory Level given the parameters. Results can then be plotted on `figure`"""
    if neuron is None:
        neuron = SingleDend(L=707, diam=1, nseg=273)
    if neuron_args is None:
        neuron_args = dict()

    if isinstance(neuron, BaseNeuron):
        neuron = neuron
    else:
        # neuron is type, instantiate it
        neuron: BaseNeuron = neuron(**neuron_args)

    if inhib_n_loc_insert is None:
        inhib_n_loc_insert = [(neuron.dend, [0.2])]
    elif type(inhib_n_loc_insert[0]) is not tuple:
        inhib_n_loc_insert = [(neuron.dend, inhib_n_loc_insert)]

    if exc_n_loc_insert is None:
        exc_n_loc_insert = [(neuron.soma, [0.5])]
    elif type(exc_n_loc_insert[0]) is not tuple:
        exc_n_loc_insert = [(neuron.dend, exc_n_loc_insert)]

    if inh_netstim_args is None:
        inh_netstim_args = dict(
            hz=10,
            start=20,
            noise=0.5,
            duration=t,
            weight=1,  # weight is number of channels activated at the synapse
            n_sources=False,
            own_stream=False,
        )

    assert x_units == settings.um or x_units == "X"

    # sim params
    if tm <= 0 or tm > t:
        tm = t  # time integration window
    if dvv_kwargs is None:
        dvv_kwargs = {}

    # add inhibitory synapses
    result_tup = add_synapse_type(
        neuron,
        inhib_syn_type,
        inhib_n_loc_insert,
        e=v_init + e_offset,
        gmax=g_i,
        std_i=g_i * settings.STD_i,
        netstim_args=inh_netstim_args,
    )
    (
        inhib_synapses,
        inhib_synapse_locations,
        inhib_n_loc_insert_actual_short,
    ) = result_tup
    logger.debug("actual inh locations: {}".format(inhib_n_loc_insert_actual_short))
    if inhib_syn_type == "inhfluct" or inhib_syn_type == "GABAa":
        logger.debug(
            "cli before setting EGABA to {:.2f} : {:.5f}".format(
                v_init + e_offset, h.cli0_cl_ion
            )
        )
        set_global_cli(inhib_synapses[0], egaba=v_init + e_offset)
        neuron.ions["cli"] = h.cli0_cl_ion
        logger.debug(
            "cli after setting EGABA to {:.2f} : {:.5f}".format(
                v_init + e_offset, h.cli0_cl_ion
            )
        )

    # add excitatory synapses
    result_tup = add_synapse_type(
        neuron,
        exc_syn_type,
        exc_n_loc_insert,
        e=0,
        gmax=g_e,
        std_e=g_e * settings.STD_e, 
        netstim_args=exc_netstim_args,
    )
    exc_synapses, exc_synapse_locations, exc_n_loc_insert_actual_short = result_tup
    logger.debug("actual exc locations: {}".format(exc_n_loc_insert_actual_short))

    if at_hotspot:
        vary = "inhib"
    else:
        vary = "exc"

    # Run simulations
    df_il, df_v, df_v_star, input_events, df_sl, df_ecl = math.dvv(
        neuron,
        v_init,
        inhib=inhib_synapses,
        exc=exc_synapses,
        vary=vary,
        tstop=t,
        tm=tm,
        quick=quick,
        g_i=g_i,
        e_offset=e_offset,
        **dvv_kwargs
    )

    df_sl_attenuation = pd.DataFrame()
    if at_hotspot:
        logger.debug("calculating attenuation")
        attenuation_calculation_locations = set(exc_n_loc_insert_actual_short)
        # # convert from [('sec',[loc1,loc2])] to [('sec',loc1),('sec',loc2)]
        # for h_sec, h_loc in exc_n_loc_insert_actual_short:
        #     for h_loc_x in h_loc:
        #         attenuation_calculation_locations.add((h_sec, h_loc_x))
        column_index = pd.MultiIndex.from_product(
            [attenuation_calculation_locations, df_il.columns.values],
            names=["hotspot_loc", "tm_value"],
        )
        df_sl_attenuation = pd.DataFrame(columns=column_index, index=df_il.index)
        for hotspot_sec_name, hotspot_loc in attenuation_calculation_locations:
            values = df_il.loc[(hotspot_sec_name, hotspot_loc)].values
            initial_attenuation_calc = np.divide(df_il, values)
            for tm_value in initial_attenuation_calc:
                step_2_calc = initial_attenuation_calc[tm_value]
                # adjust relative to location
                greater_than_hotspot_idx = step_2_calc > 1.0
                step_3 = step_2_calc
                # DataFrame.update only applies to non-nan values and does so in-place
                # https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.update.html
                step_3.update(1.0 / step_2_calc[greater_than_hotspot_idx])
                # assign
                df_sl_attenuation.loc[
                    :, ((hotspot_sec_name, hotspot_loc), tm_value)
                ] = step_3

    if figure is False:
        return df_il, df_sl_attenuation, None, None, df_ecl
    else:
        df_il, df_distance, fig, named_axes = inhib_level_plot(
            figure,
            at_hotspot,
            tm,
            t,
            neuron,
            inhib_syn_type,
            exc_syn_type,
            df_v,
            df_v_star,
            df_il,
            df_sl_attenuation,
            df_ecl,
            x_units,
            exc_n_loc_insert_actual_short,
            inhib_synapse_locations,
            inhib_n_loc_insert_actual_short,
            colormap,
            show_input,
            input_events,
            iter_label,
            time,
            plot_shape=plot_shape,
        )
        return df_il, df_distance, fig, named_axes, df_ecl
