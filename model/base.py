# coding=utf-8
from __future__ import print_function, division

import logging

import numpy as np
from nrnutils import Mechanism, stringify

from utils import settings
from utils.plot_utils import plot_v, plot_cli
from shared import hashable
from neuron import h

logger = logging.getLogger("baseneuron")


# noinspection PyAttributeOutsideInit,PyPep8Naming
@hashable
@stringify
class BaseNeuron(object):
    geom_nseg_called = False

    __strpropsignore__ = ["synapses"]

    def __init__(
        self, name="BaseNeuron", call_geom_nseg=True, add_kcc2=False, *args, **kwargs
    ):
        self.name = name
        self.kcc2_inserted = False
        self.mechanisms = []
        self.ions = {}
        self.sections = []
        self.synapses = []
        self._synapses = {
            "all": self.synapses,
            "exc": [],
            "inh": [],
            "exc_labels": [],
            "inh_labels": [],
        }
        self.netcons = {}
        self.netstims = {}
        self.netstims_ranstreams = {}
        self.vec_hoc = {}
        # self.vec_np = pd.DataFrame()
        self.apc = None
        # NOTE that the NEURON+PYTHON docs use these steps. Because nrnutils is used, there are changes
        # self.create_sections()
        # self.build_topology()
        # self.build_subsets()
        # self.define_geometry()
        # self.define_biophysics()
        self.create_mechanisms(**kwargs)
        self.soma = None
        self.axon = None
        self.dend = []
        self.build_sections(**kwargs)
        self.build_subsets()
        h.define_shape()
        if add_kcc2:
            self.add_kcc2(**kwargs)
        self.set_ions(**kwargs)

        # logger.debug("P sections: {}".format([h.psection(sec=sec) for sec in self.sections]))
        total_nseg = 0
        for sec in self.sections:
            total_nseg += sec.nseg
        if call_geom_nseg:
            logger.debug("total nseg BEFORE geom_nseg: {}".format(total_nseg))
            self.geom_nseg()
            total_nseg = 0
            for sec in self.sections:
                total_nseg += sec.nseg
        self.total_nseg = total_nseg
        logger.debug("total nseg = {}".format(total_nseg))
        logger.info(f"Neuron {name} built")

    def geom_nseg(
        self,
        freq=100,  # Hz, frequency at which AC length constant will be computed
        d_lambda=0.1,
    ):
        if self.geom_nseg_called:
            logger.debug("'geom_nseg' method should only be called once")
        # h.geom_nseg()
        for sec in self.sections:
            sec.nseg = (
                int((sec.L / (d_lambda * h.lambda_f(freq, sec=sec)) + 0.9) / 2) * 2 + 1
            )
        self.geom_nseg_called = True

    def create_mechanisms(self, **kwargs):
        raise NotImplementedError

    def build_sections(self, Ra=160, cm=1, *args, **kwargs):
        """ Geometry and topology goes here
        Membrane properties across the cell
        :param Ra:  cytoplasmic resistivity         Ohm/cm
        :param cm:  specific membrane capacitance   uF/cm^2
        """
        from model.morphology import create_soma

        self.Ra = Ra
        self.cm = cm
        create_soma(self, **kwargs)

    def build_subsets(self):
        """Build subset lists. For now we define 'all'."""
        self.all = h.SectionList()
        self.all.wholetree(sec=self.soma)
        # self.sections = list(h.allsec())

    def insert_spike_count(self, section=None, location=1):
        if section is None:
            section = self.soma
        self.apc = section.add_synapse("apc", "APCount1", locations=[location])

    @property
    def num_spikes(self):
        if self.apc is None:
            raise Exception(
                "APCount1 not initialised. Call BaseNeuron.insert_spike_count(section,location)"
            )
        return self.apc.n

    def set_ions(
        self,
        sections=None,
        ki=140,
        ko=5,
        nai=8,
        nao=145,
        cli=5,
        clo=135,
        hco3i=12,
        hco3o=23,
        **kwargs,
    ):
        if sections is None:
            sections = self.sections
        if not self.kcc2_inserted:
            # add cl_ion anyway for consistency
            for sec in sections:
                try:
                    sec.insert("cl_ion")
                except ValueError:
                    h.ion_register("cl", -1)
                    sec.insert("cl_ion")
                # ion_style("name_ion", c_style, e_style, einit, eadvance, cinit)
                # cinit = 1 is for assigning from global (we are)
                h.ion_style("cl_ion", 3, 2, 1, 0, 1, sec=sec)
        # set HCO3 now because synapses will be added later and we don't want to set hco3 after adding synapses
        #   normally just need to set HCO3 at site of (inhib) synapse, but we add the ion to every section and then
        #   have it assigned from the global value using ion_style and h.hco3i0_hco3_ion
        for sec in sections:
            try:
                sec.insert("hco3_ion")
            except ValueError:
                h.ion_register("hco3", -1)
                sec.insert("hco3_ion")
            h.ion_style("hco3_ion", 3, 2, 1, 0, 1, sec=sec)
        # else:
        #     for sec in sections:
        #         sec.ki = ki
        #         sec.ko = ko
        #         # sec.nai = nai
        #         # sec.nao = nao
        # set globals
        self.set_cl(cli=cli, clo=clo)
        h.ki0_k_ion = ki
        h.ko0_k_ion = ko
        h.nai0_na_ion = nai
        h.nao0_na_ion = nao
        h.hco3i0_hco3_ion = hco3i
        h.hco3o0_hco3_ion = hco3o
        self.ions = dict(
            ki=ki, ko=ko, nai=nai, nao=nao, cli=cli, clo=clo, hco3i=hco3i, hco3o=hco3o
        )

    def add_kcc2(self, tonic=False, **kwargs):
        if tonic or "pas" in [mech.name for mech in self.mechanisms]:
            # default 'pas' mechanism doesn't have a tonic chloride current,
            # so need the tonic version of KCC2
            kcc2_name = "KCC2tonic"
        else:
            kcc2_name = "KCC2"
        parameters = {}
        kws = ["Pa", "diff"]
        for kw in kws:
            if kw in kwargs:
                parameters[kw] = kwargs[kw]
        kcc2 = Mechanism(kcc2_name, **parameters)
        for section in self.sections:
            kcc2.insert_into(section)
        self.kcc2_inserted = True
        self.mechanisms.append(kcc2)
        return self

    def remove_kcc2(self):
        for section in self.sections:
            # https://www.neuron.yale.edu/phpBB/viewtopic.php?t=3335
            h("uninsert KCC2", sec=section)
        self.kcc2_inserted = False

    def set_cl(self, cli=5, clo=None):
        try:
            h.cli0_cl_ion = cli
            self.ions["cli"] = cli
            if clo is not None:
                h.clo0_cl_ion = clo
                self.ions["clo"] = clo
        except LookupError:
            logger.warning("cli not in the neuron")
            pass
        # try:
        #     for sec in self.sections:
        #         for seg in sec:
        #             seg.cli = cli
        #             if clo is not None:
        #                 seg.clo = clo
        # except (NameError, AttributeError):
        #     # cli is global not per section segment
        #     pass

    def add_n_synapses(
        self,
        syn_type="AlphaSynapse",
        sections=None,
        n=0,
        locations=None,
        syn_label_extra="",
        **kwargs,
    ):
        location_is_n = False
        if n == 0 and type(locations) is int:
            # 'locations was' actually 'n'
            n = locations
            locations = None
            location_is_n = True
        assert n > 0 or (
            locations is not None and len(locations) > 0
        ), "'total' or 'locations' must be specified"
        if sections is None:
            sections = self.sections
        elif type(sections) is str:
            if "[" in sections:
                section, index = sections.split("[")
                index = int(index[:-1])  # ignore ']'
                sections = getattr(self, section)[index]
            else:
                sections = getattr(self, sections)

        if type(sections) is not list:
            sections = [sections]

        if n == 0:
            n = len(locations)
        elif n < 0:
            raise BaseException("'n' incorrectly specified as '{}'".format(n))

        if locations is None:
            locations = np.linspace(0.0, 1.0, int(n / len(sections)) + 2)
            logger.debug("number of synapses: {}".format(n))
            logger.debug("number of branches: {}".format(len(sections)))
            logger.debug(
                "number of synapses per branch: {}".format(int(n / len(sections)))
            )
            # remove first and last locations (which would be 0 and 1)
            locations = locations[1:-1]

        if len(locations) == 1 and n > 1 and not location_is_n:
            locations = list(locations) * n  # place n synapses in the same location
        num_unique_locs = len(set(locations))
        all_new_synapses = []
        for i, section in enumerate(sections):
            assert section in self.sections
            if section.nseg < num_unique_locs:
                old_nseg = section.nseg
                section.nseg = num_unique_locs
                logger.info(f"increased nseg from {old_nseg} to {section.nseg}")
            # unique name
            synapse_label = section.name() + "_" + syn_type
            if syn_label_extra != "":
                synapse_label = synapse_label + "_" + syn_label_extra
            section.add_synapses(synapse_label, syn_type, locations=locations, **kwargs)
            added_synapses = getattr(section, synapse_label)
            if type(added_synapses) is not list:
                added_synapses = [added_synapses]
            new_synapses = []
            for s, syn in enumerate(added_synapses):
                new_synapses.append(
                    {
                        "label": synapse_label + "_" + str(s),
                        "object": syn,
                        "sec": section,
                        "loc": syn.get_loc(),
                    }
                )
                h.pop_section()  # important after `get_loc()`
            # logger.debug("actual locations: {}".format([syn['object'].get_loc() for syn in new_synapses]))
            all_new_synapses += new_synapses
        self.synapses += all_new_synapses
        if "GABA" in syn_type or "inhfluct" in syn_type:
            self._synapses["inh"] += all_new_synapses
        else:
            self._synapses["exc"] += all_new_synapses
        # logger.debug("Synapses:")
        # logger.debug(pformat(self.synapses))

        return all_new_synapses

    def netstim(
        self,
        synapses=None,
        hz=5,
        start=0,
        duration=5,
        noise=0,
        weight=1,
        delay=0,
        n_sources=1,
        own_stream=False,
    ):
        """
        Stimulate the synapses with NET_RECEIVE blocks
        Uses RandomStream in "ranstream.hoc"



        :param synapses:        list of synapses to receive stimulation
        :param hz:              frequency of stimulation (Hz)
        :param start:           start of stimulation (ms)
        :param duration:        duration of stimulation (ms)
        :param noise:           noise of Poisson distribution [0-1]
        :param weight:          weight of stimulation to synapse (first argument for NET_RECEIVE block)
        :param delay:           delay between netstim activation and the target receiving the input (can be list)
        :param n_sources:       whether a single netstim should stim all synapses (default:1)
                                or each synapses has its own input (0)
        :param own_stream:      give each netstim it's own random stream. Will only have an effect is self.stim is None

        :type delay: List or int

        """
        if synapses is None:
            synapses = self.synapses
        if duration is None:
            duration = h.tstop
        elif duration > h.tstop:
            h.tstop = duration
        if type(delay) is list:
            assert len(delay) == len(synapses)
        netstim_every_n_synapses = (
            int(len(synapses) / n_sources) + 1 if n_sources > 0 else 1
        )
        prev_net_stim_obj = None
        for i, syn in enumerate(synapses):
            if syn["label"] not in self.netstims:
                # create a new NetStim object every n synapses and when i==0
                if i % netstim_every_n_synapses == 0:
                    net_stim_obj = h.NetStim()
                    prev_net_stim_obj = net_stim_obj
                else:
                    net_stim_obj = prev_net_stim_obj

                self.netstims[syn["label"]] = net_stim_obj
                if own_stream:
                    # give each netstim it's own random stream
                    ran_stream = h.RandomStream(len(self.netstims))
                    net_stim_obj.noiseFromRandom(ran_stream.r)
                    ran_stream.r.negexp(
                        1
                    )  # must specify negexp distribution with mean = 1
                    ran_stream.start()
                    # store object in memory else hoc's initialisation fails as it can't find the object
                    self.netstims_ranstreams[syn["label"]] = ran_stream
                net_stim_obj.seed(settings.RANDOM_SEED)
            else:
                # get existing object if this method is called again
                # Note that the seed is set again to repeat the simulation (change start time to continue the run
                # instead)
                net_stim_obj = self.netstims[syn["label"]]
                net_stim_obj.seed(settings.RANDOM_SEED)

            if i % netstim_every_n_synapses == 0:
                net_stim_obj.interval = 1000 / hz
                net_stim_obj.number = hz * (
                    duration / 1000
                )  # number = freq[1/s] * (duration[ms] / 1000[convert ms to s])
                net_stim_obj.start = start
                net_stim_obj.noise = noise
            # NetCon for connecting NetStim to synapses
            if syn["label"] not in self.netcons:
                # create net connect object. args: source, target, thresh, delay, weight
                net_con_obj = h.NetCon(net_stim_obj, syn["object"], 0, 0, weight)
                self.netcons[syn["label"]] = net_con_obj
            else:
                net_con_obj = self.netcons[syn["label"]]

            if type(delay) is list:
                net_con_obj.delay = delay[i]
            else:
                net_con_obj.delay = delay
            net_con_obj.weight[0] = weight  # NetCon weight is a vector.

    def plot(self, ax=None, section=None, location=None, **kwargs):
        assert len(ax) == 2
        plot_v(self.vec_hoc, ax=ax[0], section=section, location=location, **kwargs)
        plot_cli(self.vec_hoc, ax=ax[1], section=section, location=location, **kwargs)

    def rotateZ(self, theta):
        """Rotate the cell about the Z axis."""
        rot_m = np.array(
            [[np.sin(theta), np.cos(theta)], [np.cos(theta), -np.sin(theta)]]
        )
        for sec in self.all:
            logger.debug(
                "{}.n3d()={} | rot_m={}".format(sec.name(), h.n3d(sec=sec), rot_m)
            )
            for i in range(int(h.n3d(sec=sec))):
                xy = np.dot([h.x3d(i, sec=sec), h.y3d(i, sec=sec)], rot_m)
                logger.debug("\ti={} | xy={} | z3d={}".format(i, xy, h.z3d(i, sec=sec)))
                h.pt3dchange(
                    i,
                    float(xy[0]),
                    float(xy[1]),
                    h.z3d(i, sec=sec),
                    h.diam3d(i, sec=sec),
                )

    def record(self, recordings=None):
        """Record variables at various locations of the neurons
        :param recordings: list of variables to record, with their locations
        :return:
        """
        if recordings is None:
            recordings = [
                {
                    "record_var": "v",
                    "locations":  {
                        self.soma: [0.5]
                    }
                }]
        for recording_dict in recordings:
            record_var, locations = recording_dict["record_var"], recording_dict["locations"]
            self.vec_hoc[record_var] = {}
            for compartment, loc in locations.items():
                if type(compartment) is str:
                    for sec in self.sections:
                        if sec.hname() == compartment or (hasattr(sec, 'name') and sec.name == compartment):
                            compartment = sec
                            break
                    else:
                        raise Exception("compartment with name {} not found in sections".format(compartment))
                if compartment not in self.sections:
                    logger.debug("Not a valid compartment {}.{}".format(self.name, compartment.hname()))
                    continue
                    # raise Exception("Not a valid compartment {}".format(compartment.hname()))

                if loc == 'all' or loc is None:
                    loc = [0] + [seg.x for seg in compartment] + [1]
                elif loc == 'middle':
                    loc = [0.5]
                elif type(loc) is not list:
                    loc = [loc]

                if hasattr(compartment, 'name'):
                    compartment_name = compartment.name()
                else:
                    compartment_name = compartment.hname()
                self.vec_hoc[record_var][compartment_name] = {}

                for x in loc:
                    x = float("{:.5f}".format(x))
                    vec_ref = self.vec_hoc[record_var][compartment_name][x] = h.Vector()
                    try:
                        vec_ref.record(compartment(x).__getattribute__("_ref_{}".format(record_var)))
                    except NameError as ne:
                        logger.warning(ne)
                        raise ne
        from shared import t_vec
        t_vec.record(h.__getattribute__("_ref_t"))

    def convert_recordings(self):
        """
        Convert the recordings from the latest to a pd.DataFrame object for easier selecting of data
        :return: A dataframe with simulation time as the index and var/compartment/seg.x as the columns
        """
        import pandas as pd
        from shared import t_vec
        
        index = np.array(t_vec)
        self.vec_np = pd.DataFrame()
        for record_var, compartments in self.vec_hoc.items():
            for compartment_name, loc in compartments.items():
                # create a new dataframe for this compartment, with time as index and
                #   each column will be in form var/compartment/x
                columns = pd.MultiIndex.from_product([[record_var], [compartment_name], sorted(loc.keys())],
                                                     names=['record_var', 'compartment_name', 'seg.x'])
                df_t = pd.DataFrame(columns=columns, index=index)
                for x, rec in loc.items():
                    # logger.debug("{:5s} {:10s} {:.3f}".format(record_var, compartment_name, x))
                    df_t.loc[:, (record_var, compartment_name, x)] = np.array(rec)
                # sort the columns
                # self.vec_np[compartment_name]=self.vec_np[compartment_name].sort_index(axis=1)
                # add the columns of the new dataframe to the main dataframe
                if self.vec_np.shape[1] > 0:
                    self.vec_np = pd.concat([self.vec_np, df_t], axis=1)
                else:
                    # assign it the first time so MultiIndex is used for the columns
                    self.vec_np = df_t
        return self.vec_np

def rotate_z(selected_sec, theta):
    """Rotate the cell about the Z axis."""
    rot_m = np.array([[np.sin(theta), np.cos(theta)], [np.cos(theta), -np.sin(theta)]])
    # TOFIX: don't iterate through all sections to find desired one (explicitly using selected_sec didn't seem to work)
    for sec in h.allsec():
        if sec.hoc_internal_name() != selected_sec.hoc_internal_name():
            continue
        logger.debug("{}.n3d()={} | rot_m={}".format(sec.name, h.n3d(sec=sec), rot_m))
        logger.debug(h.n3d(sec=sec))
        for i in range(int(h.n3d(sec=sec))):
            xy = np.dot([h.x3d(i, sec=sec), h.y3d(i, sec=sec)], rot_m)
            logger.debug("\ti={} | xy={} | z3d={}".format(i, xy, h.z3d(i, sec=sec)))
            h.pt3dchange(
                i, float(xy[0]), float(xy[1]), h.z3d(i, sec=sec), h.diam3d(i, sec=sec)
            )

