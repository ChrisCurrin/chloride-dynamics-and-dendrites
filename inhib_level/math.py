# coding=utf-8
import logging
import time
from collections import namedtuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from neuron import h

from utils import settings
from shared import t_vec, is_dvv_saved, save_dvv, load_dvv, env_var

logger = logging.getLogger("inhib_level_math")

zz = h.Impedance()


# noinspection PyPep8Naming
def lambda_d(d, Rm, Ra):
    """Calculate lambda - length/space constant, in um, which is electrotonic length (L) of 1
    :param d - diameter (um)
    :param Rm - membrane resistivity (set in pas channel)
    :param Ra - axial resistivity (set in neuron)
    :return length constant (um)
    """

    return np.sqrt(d * Rm / Ra / 4) * 100


def calc_z(sec, where, FREQ=0):
    h.distance(0, where, sec=sec)
    zz.loc(where, sec=sec)
    zz.compute(FREQ, 1, sec=sec)


def dvv(
    neuron,
    v_init,
    inhib,
    exc,
    vary="exc",
    tstop=10,
    tm=0,
    quick=True,
    g_i=0,
    e_offset=-0.0,
    sections=None,
    select_segs=None,
    calc_shunt=False,
):
    """
    o = soma
    - = dendrite branch
    i = inhibition
    x = excitation
    v = voltage recording

    Vary the location of excitation while keeping inhibition the same
    i.e.
                i
    o--------------------------------
      x x x x x x x x x x x x x x x x
      v v v v v v v v v v v v v v v v

     Inhibition level is therefore IL(X), given i
    IL_d

    Vary the location of inhibition while keeping excitation the same
    i.e.
          i i i i i i i i i i i i i i i i
        o--------------------------------
                        x
                        v

    Inhibition level is therefore IL(i), given X ["IL at hotspot"]. This is best expressed in attenuation
    IL_i,h

    Shunt Level Calculation from
    Gidon, A., & Segev, I. (2012). Principles Governing the Operation of Synaptic Inhibition in Dendrites. Neuron,
        75(2), 330–341. https://doi.org/10.1016/j.neuron.2012.05.015

    https://www.neuron.yale.edu/neuron/static/new_doc/analysis/programmatic/impedance.html

    Symbols according to table in paper
    X,(Xi)  Electrotonic distance (in units of the space constant, l) from origin (to location i); (dimensionless).
    L 		Electrotonic length (in units of l) of a dendritic branch; (dimensionless).
    V 		Steady membrane potentials, as a deviation from the resting potential; (volt).
    Ri  	Input resistance at location i;(U).
    DRi 	Change in Ri due to synaptic conductance perturbation; (U).
    gi 		Steady synaptic conductance perturbation at location i; (S).
    IL 		Shunt level; (0%IL%1; dimensionless).
    SLi 	Shunt level DRi / Ri due to activation of single or multiple conductance perturbations; (0%IL%1;
            dimensionless).
    Ri,j  	Transfer resistance between location i and location j;(U).
    SLi,j	Attenuation of IL (SLj/ SLi) for a single conductance perturbation at location i;(0%SLi,j%1;
            dimensionless).
    Ai,j	Voltage attenuation, Vj/Vi, for current perturbation at location i;(0%Ai,j%1; dimensionless).
    p 		Dendritic-to-somatic conductance ratio; (G dendrite/G soma; dimensionless).
    RN  	Input resistance at X = 0 for a semi-infinite cable; (U).
    B 		Cable boundary condition; (G dendrite/GN; dimensionless)

    In equations, i = input location and d = other location
    DRd = Rd - Rd* = (gi x (Ri,d)^2)/(1+ gi x Ri)       Equation 4
    Ri,d = Rd,i = Ri x Ai,d = Rd x Ad,i                 Equation 5
    ILd = DRd / Rd = [(gi x Ri)/(1 + gi x Ri)] x Ai,d x Ad,i
    SLi,d = ILd,i = Ai,d x Ad,i

    """
    if quick:
        # force change to precise if tm < tstop (values of 0 and tstop are fine)
        quick = not 0 < tm < tstop
    saved_args = dict(locals())
    del saved_args[
        "tm"
    ]  # this isn't important for inhib_level *simulation*, only integration
    del saved_args["neuron"]
    del saved_args["calc_shunt"]
    if sections is None:
        del saved_args["sections"]
    if select_segs is None:
        del saved_args["select_segs"]
    assert vary in ["exc", "inhib"]
    # remove HocObjects from saved_args
    saved_args["exc"] = [
        {k: v for k, v in syn_i.items() if k != "object"} for syn_i in exc
    ]
    saved_args["inhib"] = [
        {k: v for k, v in syn_i.items() if k != "object"} for syn_i in inhib
    ]

    h.tstop = tstop
    str_saved_args = ",".join(
        "{}={}".format(key, value) for (key, value) in saved_args.items()
    )
    saved, fname = is_dvv_saved(neuron, str_saved_args)
    df_sl = None
    env_var(v_init=v_init)
    if saved:
        df_v, df_v_star, input_events, df_sl, df_ecl = load_dvv(fname)
    else:
        # stim
        stim = None
        if vary == "exc":
            # change exc (e.g. iclamp) location to stim at d (as per definition of IL)
            stim = exc
        elif vary == "inhib":
            stim = inhib

        stim_sec = stim[0]["sec"]
        stim_obj = stim[0]["object"]
        if "IClamp" in exc[0]["label"]:
            exc[0]["object"].dur = tstop
        if "IClamp" in inhib[0]["label"]:
            inhib[0]["object"].dur = tstop
            assert (
                inhib[0]["object"] < 0
            ), "inhib iclamp does not have negative amplitude (amp={})".format(
                inhib[0]["object"].amp
            )
        logger.debug("varying the location of {}".format(stim))
        exc_sec = exc[0]["sec"]
        exc_loc = exc[0]["loc"]
        inhib_sec = inhib[0]["sec"]
        inhib_loc = inhib[0]["loc"]

        # sim args
        h.v_init = v_init
        if quick:
            logger.debug("using cvode")
            h.useCV()
        else:
            h.disableCV()

        # initially will have every time point in the dataframe.
        from shared import t_vec

        t_vec.record(h.__getattribute__("_ref_t"))
        v_vec = h.Vector()
        ecl_vec = h.Vector()
        df_v = pd.DataFrame()
        df_v_star = pd.DataFrame()
        df_ecl = pd.DataFrame()
        df_sl = pd.DataFrame(index=["Rd", "Rd*", "Ri", "Aid", "Adi"])

        # record input events
        input_events = {}
        netcons = h.List()  # keep in memory
        dummy_ampanmda = h.AmpaNmda(
            0.5
        )  # // used as target in scheme to record spikes from stims
        for key, stim in neuron.netstims.items():
            event_vec = h.Vector()
            # // last arg 0 keeps sim unchanged
            # // source, target, thresh, delay, weight
            dummy_netcon = h.NetCon(stim, dummy_ampanmda, 0, 0, 0)
            dummy_netcon.record(event_vec)
            input_events[key] = event_vec
            netcons.append(dummy_netcon)

        # method to be re-used per new location
        def init_and_run(_df, _columns, _df_ecl=None):
            reset_seeds(neuron)

            h.finitialize()
            h.run()
            if t_vec.size() != v_vec.size() and quick:
                logger.warning(
                    f"Time vec and Voltage vec don't match ({t_vec.size()}) !=  ({v_vec.size()}). "
                    f"Running again with cvode off."
                )
                h.disableCV()
                h.finitialize()
                h.run()
                h.useCV()

            t_index = np.array(t_vec)  # get time h.Vector as numpy index
            #  note we can get away with accessing the parent method's v_vec because the recording location changes, not
            # the object itself
            np_v = v_vec.as_numpy()  # numpy reference to hoc Vector
            # set area under the curve for integration to be between Vm and v_init
            np_v = np_v - v_init  # subtract resting potential
            _df_v_t = pd.DataFrame(
                data=np_v.copy(), index=t_index, columns=_columns
            )  # copies data
            if _df.shape[1] > 0:
                _df = _df.join(_df_v_t, how="outer")
            else:
                _df = _df_v_t  # assign it the first time so MultiIndex is used for the columns
            if _df_ecl is not None:
                _ecl_t = pd.DataFrame(
                    data=np.array(ecl_vec), index=t_index, columns=_columns
                )  # copies data
                _df_ecl = (
                    _df_ecl.join(_ecl_t, how="outer")
                    if _df_ecl.shape[1] > 0
                    else _ecl_t
                )
                return _df, _df_ecl
            return _df

        if settings.NEURON_GUI:
            g = h.Graph()
            # g.size(0,5,-80,40)
            #        label      var    col brush    section
            g.addvar("simple_neuron.dend[0].v(.5)", "v(0.5)", 2, 2, sec=neuron.dend[0])
            g.addvar("simple_neuron.dend[0].v(.1)", "v(0.1)", 2, 3, sec=neuron.dend[0])
            g.addvar("simple_neuron.dend[0].v(.9)", "v(0.9)", 2, 4, sec=neuron.dend[0])
            g.addvar("soma.v(.5)", "v(0.5)", 3, 2, sec=neuron.soma)
            # g.addvar("simple_neuron.dend[0].ecl(.5)", 'ecl(0.5)', 4, 2, sec=neuron.dend[0])
            # g.addvar("simple_neuron.dend[0].ecl(.1)", 'ecl(0.1)', 4, 3, sec=neuron.dend[0])
            # g.addvar("simple_neuron.dend[0].ecl(.9)", 'ecl(0.9)', 4, 4, sec=neuron.dend[0])
            # g.addvar("simple_neuron.dend[0].egaba(.2)", 'inhfluct[0].egaba', 6, 4)
            # g.addvar("some.ecl(.5)", 'ecl(0.5)', 5, 2, sec=neuron.soma)
            # # |     Plot view   |       Window size     |
            # x1, y1, dx, dy    | x0, y0, dx, dy
            g.view(0, -80, tstop, 100, 400, 400, 3000, 1000)
            g.exec_menu("Keep Lines")
            if g not in h.graphList[0]:
                h.graphList[0].append(g)

        precision = 1
        if select_segs is not None:
            for seg in select_segs:
                _prec = len(str(seg)) - 2
                if _prec > precision:
                    precision = _prec
        if sections is not None and inhib_sec.name() not in sections:
            sections.append(inhib_sec.name())
        if select_segs is not None and inhib_loc not in select_segs:
            select_segs = list(select_segs)  # copy arg
            select_segs.append(round(inhib_loc, 5))

        logger.info("running dvv with quick={}".format(quick))
        start = time.time()
        dummy_seg = namedtuple("seg", ["x"])
        
        if sections is None:
            N = (
                neuron.total_nseg + len(neuron.sections) * 2
            )  # included 0.0 and 1.0 for each section
        else:
            # count the number of iterations based on the specified sections (and the soma)
            N = 0
            for i, sec in enumerate(neuron.sections):
                if (sec.name() != "soma") and (
                    (type(sections[0]) is str and sec.name() not in sections) or (
                        type(sections[0]) is int and i not in sections
                    )):
                        continue
                N += sec.nseg + 2 # included 0.0 and 1.0 for each section

        pbar = tqdm(total=N, desc=f"dvv for {neuron.name}")
        for i, sec in enumerate(neuron.sections):
            if sections is not None and sec.name() != "soma":
                if (type(sections[0]) is str and sec.name() not in sections) or (
                    type(sections[0]) is int and i not in sections
                ):
                    logger.debug(
                        f"skipping {sec.name()} as it is not in 'sections' arg [{sections}]"
                    )
                    continue
            segments = (
                [dummy_seg(0.0)] + [seg for seg in sec] + [dummy_seg(1.0)]
            )  # add 0.0 and 1.0
            for j, seg in enumerate(segments):
                iter_start = time.time()
                x = float(f"{seg.x:.5f}")
                if select_segs is not None and sec.name() != "soma":
                    rounded_x = round(x, precision)
                    if rounded_x not in select_segs:
                        logger.debug(
                            f"skipping {rounded_x} (rounded from {x}) as it is not in 'segments' arg "
                            f"[{select_segs}]"
                        )
                        pbar.update()
                        continue

                columns = pd.MultiIndex.from_product(
                    [[sec.name()], [x]], names=["compartment_name", "seg.x"]
                )
                # record from d if varying excitation location
                if vary == "exc":
                    v_vec.record(sec(seg.x)._ref_v)
                    ecl_vec.record(sec(seg.x)._ref_ecl)
                elif vary == "inhib":
                    # record at 'hotspot' when varying inhibition
                    v_vec.record(exc_sec(exc_loc)._ref_v)
                    ecl_vec.record(exc_sec(exc_loc)._ref_ecl)
                    # You can not locate a point process at
                    #  position 0 or 1 if it needs an ion
                    # NEURON: cl_ion can't be inserted in this node
                    if seg.x == 0.0 or seg.x == 1.0:
                        df_v_t = pd.DataFrame(
                            data=np.nan, index=np.array(t_vec), columns=columns
                        )
                        df_v_star_t = pd.DataFrame(
                            data=np.nan, index=np.array(t_vec), columns=columns
                        )
                        if df_v.shape[1] > 0:
                            df_v = pd.concat([df_v, df_v_t], axis=1).fillna(
                                method="ffill", axis=0
                            )
                            df_v_star = pd.concat(
                                [df_v_star, df_v_star_t], axis=1
                            ).fillna(method="ffill", axis=0)
                        else:
                            df_v = df_v_t  # assign it the first time so MultiIndex is used for the columns
                            df_v_star = df_v_star_t  # assign it the first time so MultiIndex is used for the columns

                        logger.debug(
                            "{}({:.5f}) \t nan (end segment)".format(sec.name(), x)
                        )
                        pbar.update()
                        continue
                # change object (e.g. iclamp) location to stim at d (as per definition of IL)
                # if object is inhib, this is IL @ hotspot
                stim_obj.loc(sec(seg.x))

                deactivate_inhibition(neuron)
                df_v = init_and_run(df_v, columns)
                if calc_shunt:
                    # analytical solution for shunt level
                    calc_z(sec, seg.x)
                    r_d = zz.input(seg.x, sec=sec)
                    a_di = zz.ratio(inhib_loc, sec=inhib_sec)

                    calc_z(inhib_sec, inhib_loc)
                    r_i = zz.input(inhib_loc, sec=inhib_sec)
                    a_id = zz.ratio(seg.x, sec=sec)

                # run again with inhibition
                activate_inhibition(neuron, g_i)
                df_v_star, df_ecl = init_and_run(df_v_star, columns, _df_ecl=df_ecl)
                if calc_shunt:
                    calc_z(sec, seg.x)
                    r_d_star = zz.input(seg.x, sec=sec)

                if logger.getEffectiveLevel() == logging.DEBUG:
                    logger.debug(
                        f"{sec.name()}({x:.5f}) \t {time.time() - iter_start:.2f}s "
                        f"\t IL~{-((df_v_star.iloc[-1, -1] - df_v.iloc[-1, -1]) / df_v.iloc[-1, -1]):.5f} "
                        f"\t df sizes: V={df_v.size} V*={df_v_star.size}"
                    )
                if calc_shunt:
                    if "IClamp" in exc[0]["label"]:
                        rd_v = df_v.iloc[-1, -1] / stim_obj.amp
                        rd_v_star = df_v_star.iloc[-1, -1] / stim_obj.amp
                        logger.debug(f"DR/R (from V) {-(rd_v_star-rd_v)/rd_v:.5f}")
                    logger.debug(f"DR/R          {-(r_d_star-r_d)/r_d:.5f}")
                    logger.debug(
                        f"analytical    {((g_i * r_i) / (1 + g_i * r_i)) * a_id * a_di:.5f}"
                    )
                    _df_sl = pd.DataFrame(
                        data=[r_d, r_d_star, r_i, a_id, a_di],
                        index=df_sl.index,
                        columns=columns,
                    )
                    if df_sl.size == 0:
                        df_sl = _df_sl
                    else:
                        df_sl = df_sl.join(_df_sl)
                pbar.update()

        logger.info("running took {:.2f}s".format(time.time() - start))

        if quick:
            logger.debug("disable cvode")
            h.disableCV()
            if vary == "inhib":
                # note that interpolation probably won't work for radial dendrites as there will be
                # discontinuities in the DataFrame
                logger.debug("interpolate")
                # for sec in df_v.columns.levels[0].values:
                #     df_v[sec] = df_v[sec].interpolate('krogh', axis=1, limit_direction='both')
                #     df_v_star[sec] = df_v_star[sec].interpolate('krogh', axis=1, limit_direction='both')
        logger.debug("align indices of v and v*")
        # because each run may have different time indices (due to cvode), forward-fill non-aligned values of the
        # dataframe
        # combine both into one to align indices
        df_v_v_star = pd.concat([df_v, df_v_star], axis=1, keys=["v", "v*"]).fillna(
            method="ffill"
        )
        df_v = df_v_v_star["v"]
        df_v_star = df_v_v_star["v*"]
        # convert input_events to dataframe (via series for dataframe index alignment)
        se_input_events = dict([(k, pd.Series(v)) for k, v in input_events.items()])
        input_events = pd.DataFrame.from_dict(se_input_events)
        save_dvv(neuron, str_saved_args, df_v, df_v_star, input_events, df_sl, df_ecl)

    if (
        (tm <= 0 or tm >= tstop)
        and "IClamp" in exc[0]["label"]
        and "inhfluct" in inhib[0]["label"]
    ):
        # just take last (steady-state) value
        df_il = -((df_v_star.iloc[-1:] - df_v.iloc[-1:]) / df_v.iloc[-1:])
        df_il.index = [tstop]
    else:
        if tm == 0:
            tm = tstop
        logger.debug("integrating Vd with t={} tm={}".format(tstop, tm))
        start = time.time()
        integral_v_d = integrate(df_v, window=tm)
        logger.debug("integrating took {:.2f}s".format(time.time() - start))

        logger.debug("integrating Vd* with t={} tm={}".format(tstop, tm))
        start = time.time()
        integral_v_d_star = integrate(df_v_star, window=tm)
        logger.debug("integrating took {:.2f}s".format(time.time() - start))

        logger.info("calculating inhib level")

        df_il = -1.0 * ((integral_v_d_star - integral_v_d) / integral_v_d)
    # re-orientate
    df_il = df_il.T.sort_index()
    return df_il, df_v, df_v_star, input_events, df_sl.T, df_ecl


def activate_inhibition(neuron, g_i):
    """

    :param neuron: Neuron object with inhibitory synapses ('inhfluct' or 'GABAa')  inserted to activate
    :param g_i: amount of conductance to assign to each synapse

    """
    inhib_synapses = neuron._synapses["inh"]

    # activate conductances of inhibitory synapse(s)
    # initialise to prevent issues with parameter assignment mid-way through simulations
    h.finitialize()
    if "inhfluct" in inhib_synapses[0]["label"]:
        params = {"g_i0": g_i, "std_i": g_i * settings.STD_i}
    else:
        params = {"gmax": g_i}

    for i, inh_syn in enumerate(inhib_synapses):
        for key, value in params.items():
            try:
                setattr(inh_syn["object"], key, value)
            except LookupError:
                if key == "gmax":
                    # try alternative gmax param name
                    setattr(inh_syn["object"], "g_max", value)


def deactivate_inhibition(*args, **kwargs):
    activate_inhibition(g_i=0, *args, **kwargs)


def reset_seeds(neuron):
    for key, stim in neuron.netstims.items():
        stim.seed(settings.RANDOM_SEED)
    # for synapse in neuron.synapses:
    #     try:
    #         synapse['object'].new_seed(settings.RANDOM_SEED)
    #     except AttributeError:
    #         pass


def integrate(df, how="trapz", window=20.0, rolling=False):
    """Numerically integrate the time series.

    :type df: pd.DataFrame
    :param df: dataframe to integrate (note that integration will be relative to 1st data point, v_init)
    :param how: the method to use (trapz)
    :param window: the integration window, tm, in ms (20)
    :param rolling: integrate on a rolling window basis (False)

    :return

    Available methods:
     * trapz - trapezoidal
     * cumtrapz - cumulative trapezoidal
     * simps - Simpson's rule
     * romb - Romberger's rule

    See http://docs.scipy.org/doc/scipy/reference/integrate.html for the method details.
    or the source code
    https://github.com/scipy/scipy/blob/master/scipy/integrate/quadrature.py
    """
    from scipy import integrate

    available_rules = set(["trapz", "cumtrapz", "simps", "romb"])
    if how in available_rules:
        rule = integrate.__getattribute__(how)
    else:
        print("Unsupported integration rule: %s" % (how))
        print(
            "Expecting one of these sample-based integration rules: %s"
            % (str(list(available_rules)))
        )
        raise AttributeError

    if rolling:
        rolling_window = df.rolling(window=int(window / h.dt))
        result = rolling_window.apply(rule)  # integrate along the index (time)
    else:
        if df.last_valid_index() > h.tstop:
            n_windows = int(df.last_valid_index() / window)
        else:
            n_windows = int(df.last_valid_index() / window) + 1
        t_points = [i * window for i in range(1, n_windows + 1)]
        result = pd.DataFrame(columns=df.columns, index=t_points)
        prev_t_point = 0
        for t_point in t_points:
            result.loc[t_point] = df[prev_t_point:t_point].apply(rule)
            prev_t_point = t_point
    return result


def get_df_loc_idx(df: pd.DataFrame, loc: list, dend="radial_dends_1"):
    index = df.loc[dend].index
    if type(loc) is not list:
        loc = [loc]
    loc_idx = np.ndarray(shape=len(loc), dtype=int)
    for li, x in enumerate(loc):
        loc_idx[li] = abs(x - index).argmin()
    return loc_idx


def accumulation_index(df_il: pd.DataFrame, loc: list, dend="radial_dends_1"):
    """Calculate the 'Accumulation Index' of an IL dataframe for each location in loc.
    The Accumulation Index (AccIdx) is the ratio of the IL at the soma to the IL at a location l.
    i.e. if a neuron has an IL at the soma(.5) of 0.3 and an IL at radial_dends_1(0.2) of 0.6
        then the AccIdx is 0.3/0.6 = 0.5

    :rtype: dict[float, pd.Series]
    """
    soma_val = df_il.loc["soma", 0.5]
    accum = {}
    index = df_il.loc[dend].index
    for li in loc:
        loc_idx = abs(li - index).argmin()
        inhib_point_val = df_il.loc[dend].iloc[loc_idx]
        accum[li] = soma_val / inhib_point_val
    return accum


def offset_index(il_dict, num_dendrites_arr):
    """Calculate the 'Offset Index' of all IL curves which have the same number of branches.
    For a given number of branches, the Offset Index is calculated as percentage point increase in IL due to more
    negative reversal potential from baseline.
    E.g. for an arbor of 4 branches and simulations with e_offsets of 0 (required) and -1 mV
            the increase in IL at the soma due to a more negative offset is
                            [IL @ soma for e=-1] / [IL @ soma for e=0]
                        =           1.5          /         0.30
                        =   5.0 times greater IL
            This is done for each location.

    :param il_dict: Dictionary of IL dataframes from the `dvv` simulation.
    :type il_dict: dict[str, pd.DataFrame]
    :param num_dendrites_arr: List of number of dendrites used in a simulation. To separate # branches.
    :type num_dendrites_arr: list[str or int]
    :return: Offset Index
    :rtype: dict[str, pd.DataFrame]
    """
    offset_dict = {}
    keys = il_dict.keys()
    for n_i, n_d in enumerate(num_dendrites_arr):
        group_keys = []
        idx = 0
        for key in keys:
            if "n={}".format(n_d) in key:
                if "e=0.0" in key:
                    idx = len(group_keys)
                group_keys.append(key)
        base_il_key = group_keys.pop(idx)
        base_il = il_dict[base_il_key]

        for i, other_il_key in enumerate(group_keys):
            offset_dict[other_il_key] = il_dict[other_il_key] / base_il
    return offset_dict


def inv_nernst(e, c_out): 
    """Inverse of Nernst equation

    :math: `E = RTF/z x ln(c_out / c_in)`
    
    for z = -1 
    :math: `c_in = c_out x exp(E/RTF)`

    :param e: voltage in mV
    :param c_out: concentration in mM

    :return: concentration in mM

    """
    RTF = (h.R * (h.celsius+273.15)) / h.FARADAY
    return np.exp((1/RTF) * e/1000) * c_out


def ghk(C_outs, C_ins, ps, zs):
    """Calculate the GHK current for a given set of concentrations, proportions, and valences"""
    dividend = 0
    divisor = 0
    RTF = (h.R * (h.celsius+273.15)) / h.FARADAY
    for cin, cout, p, z  in zip(C_ins, C_outs, ps, zs):
        assert abs(z) == 1, "only monovalent ions supported"
        if z>0:
            dividend += p*cout
            divisor += p*cin
        else:
            dividend += p*cin
            divisor += p*cout
    return RTF*np.log(dividend/divisor)*1000