# coding=utf-8
from __future__ import division

import logging

from nrnutils import Section, PROXIMAL
from model.mechanisms import pas_mechanism
from model.base import BaseNeuron

logger = logging.getLogger('morphology')


# geometry and topology

def create_soma(self, soma_L=15, soma_diam=15, soma_nseg=1, mechanisms=None, **kwargs):
    if mechanisms is None:
        mechanisms = self.mechanisms
    self.soma = Section(name='soma', L=soma_L, diam=soma_diam, nseg=soma_nseg, Ra=self.Ra, cm=self.cm,
                        mechanisms=mechanisms)
    self.sections.append(self.soma)


def create_axon(self, axon_L=100, axon_diam=0.2, mechanisms=None, **kwargs):
    if mechanisms is None:
        mechanisms = self.mechanisms
    self.axon = Section(name='axon', L=axon_L, diam=axon_diam, Ra=self.Ra, cm=self.cm,
                        mechanisms=mechanisms,
                        parent=self.soma,
                        connection_point=PROXIMAL)
    self.sections.append(self.axon)


def create_radial_dends(self, num_dendrites=1, parent=None, **kwargs):
    if parent is None:
        parent = self.soma
    self.radial_dends = []
    for i in range(num_dendrites):
        self.radial_dends.append(Section(name='radial_dends_{}'.format(i + 1),
                                         L=self.radial_L,
                                         diam=self.radial_diam,
                                         nseg=self.radial_nseg,
                                         Ra=self.Ra, cm=self.cm,
                                         mechanisms=self.mechanisms,
                                         parent=parent))
    self.sections += self.radial_dends

def create_sinks(self, num_sinks=1, parent=None, **kwargs):
    if parent is None:
        parent = self.soma
    self.sinks = []
    for i in range(num_sinks):
        self.sinks.append(Section(name='sink_{}'.format(i + 1),
                                         L=self.sink_L,
                                         diam=self.sink_diam,
                                         nseg=self.sink_nseg,
                                         Ra=self.Ra, cm=self.cm,
                                         mechanisms=self.mechanisms,
                                         parent=parent))
    self.sections += self.sinks

# noinspection PyAttributeOutsideInit
class MultiDend(BaseNeuron):
    """
    Soma with [num_dendrites] projecting from it.
    """

    def __init__(self, name="MultiDend", num_dendrites=1,
                 radial_l=50, radial_diam=2, radial_nseg=9,
                 reduced=False, call_geom_nseg=False,
                 *args, **kwargs):
        self.num_dendrites = num_dendrites
        self.radial_L = radial_l
        self.radial_diam = radial_diam
        self.radial_nseg = radial_nseg
        super(MultiDend, self).__init__(name, call_geom_nseg=call_geom_nseg, *args, **kwargs)

    def build_sections(self, soma_L=0.01, soma_diam=2, **kwargs):
        super(MultiDend, self).build_sections(soma_L=soma_L, soma_diam=soma_diam, **kwargs)
        self.soma.Ra = 0.2

        create_radial_dends(self, num_dendrites=self.num_dendrites)

        self.dend = self.radial_dends

    create_mechanisms = pas_mechanism

# noinspection PyAttributeOutsideInit
class MultiDendWithSink(MultiDend):
    def __init__(self, name="MultiDendWithSink", 
                 sink_num=1, sink_l=50, sink_diam=2, sink_nseg=9,
                 *args, **kwargs):
        self.num_sinks = sink_num
        self.sink_L = sink_l
        self.sink_diam = sink_diam
        self.sink_nseg = sink_nseg
        super(MultiDendWithSink, self).__init__(name, *args, **kwargs)

    def build_sections(self, **kwargs):
        super(MultiDendWithSink, self).build_sections(**kwargs)

        create_sinks(self, num_sinks=self.num_sinks)

        self.dend = self.radial_dends + self.sinks



# noinspection PyAttributeOutsideInit,PyPep8Naming
class SingleDend(BaseNeuron):
    """
    Soma with a single dendrite and (optionally) an axon
    don't auto-calculate nseg
    """

    def __init__(self, name="SingleDend", call_geom_nseg=False, *args, **kwargs):
        super(SingleDend, self).__init__(name, call_geom_nseg=call_geom_nseg, *args, **kwargs)
        # reset soma area: set L and diam to 2.8012206 , so that area = 24.651565 (from rho = 0.1)
        # self.soma.L = 2.8012206
        # self.soma.diam = 2.8012206

    create_mechanisms = pas_mechanism

    def build_sections(self, dend_l=707, dend_diam=1., dend_nseg=81, axon=False, **kwargs):
        super(SingleDend, self).build_sections(**kwargs)
        if axon:
            create_axon(self, **kwargs)
        self.dend.append(Section(name='dend_1',
                                 L=dend_l, diam=dend_diam, nseg=dend_nseg, Ra=self.Ra, cm=self.cm,
                                 mechanisms=self.mechanisms,
                                 parent=self.soma))
        self.sections += self.dend


