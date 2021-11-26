from nrnutils import Mechanism


# create curried version for runtime assignment in a class
def create_mechanism(self, mechanism):
    def _mechanism(**kwargs):
        mechanism(self, **kwargs)
    return _mechanism


########################################################################################################################
# MECHANISMS
########################################################################################################################


# noinspection PyPep8Naming
def pas_mechanism(self, Rm=20000.0, pas_e=-65.0, **kwargs):
    """ Method to add the NEURON 'pas' mechanism to an object's mechanisms list
    :param Rm: membrane resistance (Ohm/cm^2)
    :param pas_e: reversal potential (mV)
    """
    self.pas = Mechanism('pas', e=pas_e, g=(1.0 / Rm))
    assert self.pas.parameters['g'] > 0, "division not working correctly"
    self.mechanisms.append(self.pas)


def pasghk_mechanism(self, g_na_bar=2000, g_k_bar=5, g_im_bar=0.00012,
                      g_pas_k=0.000125, p_na=0.23, p_cl=0.4,
                      **kwargs):
    g_pas_na = g_pas_k * p_na
    g_pas_cl = g_pas_k * p_cl
    # mechanisms
    self.na = Mechanism('na', gbar=g_na_bar)
    self.kv = Mechanism('kv', gbar=g_k_bar)
    self.im = Mechanism('im', gkbar=g_im_bar)
    self.pasghk = Mechanism('pasghk', gclpbar=g_pas_cl, gnapbar=g_pas_na, gkpbar=g_pas_k)
    self.mechanisms.append(self.na)
    self.mechanisms.append(self.kv)
    self.mechanisms.append(self.im)
    self.mechanisms.append(self.pasghk)


@staticmethod
def get_g_pas_ions(g_pas_k=0.000125, p_na=0.23, p_cl=0.4):
    return g_pas_k, g_pas_k * p_na, g_pas_k * p_cl