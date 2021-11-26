# coding=utf-8
import os
import platform
import logging

import colorlog
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors

# ----------------------------------------------------------------------------------------------------------------------
# GLOBAL DEFAULTS
# ----------------------------------------------------------------------------------------------------------------------
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm, ListedColormap, Normalize

RASTERIZED = False  # True for faster rendering/saving
SAVE_FIGURES = True
STD_i = 0
STD_e = 0
TSTOP = 150
DIAM = 1
ZOOM_BORDER_PAD = 0.5
ZOOM_BORDER_PAD_TIGHT = 0.

# ----------------------------------------------------------------------------------------------------------------------
# SET LOGGER
# ----------------------------------------------------------------------------------------------------------------------
handler = colorlog.StreamHandler()
fmt = '%(asctime)s %(name)-10s [%(filename)-10s:%(lineno)4d] %(levelname)-8s \t %(message)s'
datefmt = '%m-%d %H:%M:%S'
handler.setFormatter(colorlog.ColoredFormatter('%(log_color)s' + fmt, datefmt=datefmt))

logging.basicConfig(
        level=logging.DEBUG,
        format=fmt,
        handlers=[handler],
        datefmt=datefmt)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("colormath.color_conversions").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)
logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)


# ----------------------------------------------------------------------------------------------------------------------
# NEURON
# ----------------------------------------------------------------------------------------------------------------------
NEURON_GUI = False
NEURON_RECOMPILE = False
NEURON_QUICK_DEFAULT = True  # use cvode where implicit
HOC_PATH = "utils/hoc_files/"
MOD_PATH = "utils/mod_files/"
NRNMECH_PATH = ''

if not os.path.isdir(MOD_PATH):
    # find the dir
    dir_path = os.path.dirname(os.path.realpath(__file__))
    MOD_PATH = os.path.join(dir_path, MOD_PATH)
    HOC_PATH = os.path.join(dir_path, HOC_PATH)
if platform.system() == 'Linux' or platform.system() == 'Darwin':
    NRNMECH_PATH = MOD_PATH + "x86_64/.libs/libnrnmech.so"
elif platform.system() == 'Windows':
    NRNMECH_PATH = MOD_PATH + "nrnmech.dll"
else:
    print("unknown system")
    exit(-1)
NRNMECH_PATH = NRNMECH_PATH.replace("\\", "/")

# ----------------------------------------------------------------------------------------------------------------------
# RANDOM
# ----------------------------------------------------------------------------------------------------------------------
RANDOM_SEED = 0
# max # events in a NetStim's stream  (Adjacent streams will be correlated by this offset.)
# // before it begins to repeat values already generated
# // by a different stream.
# // set to 0 and all NetStims will produce identical streams
RANDOM_STREAM_OFFSET = 1000

# ----------------------------------------------------------------------------------------------------------------------
# MATPLOTLIB PLOT CONFIG
# ----------------------------------------------------------------------------------------------------------------------
article_style_path = "article.mplstyle"
if not os.path.isfile(article_style_path):
    # find the file
    dir_path = os.path.dirname(os.path.realpath(__file__))
    article_style_path = os.path.join(dir_path, article_style_path)
plt.style.use(article_style_path)
logging.getLogger("settings").debug("imported style {}".format(article_style_path))

# DEFINE FIGURE USEFUL SIZES (in inches)
PAGE_W_FULL = 7.5
PAGE_H_FULL = 7.5  # make square so there's space for caption
PAGE_H_FULL_no_cap = 8.75  # no space for caption
PAGE_W_half = PAGE_W_FULL/2
PAGE_H_half = PAGE_H_FULL_no_cap/2
PAGE_W_3rd = PAGE_W_FULL/3
PAGE_H_3rd = PAGE_H_FULL_no_cap/3
PAGE_W_4th = PAGE_W_FULL/4
PAGE_H_4th = PAGE_H_FULL_no_cap/4
PAGE_W_column = 5.2  # according to https://journals.plos.org/ploscompbiol/s/figures#loc-dimensions
# GridSpec layout
GS_R = 36
GS_C = 36
GS_R_half = int(GS_R/2)
GS_C_half = int(GS_C/2)
GS_R_third = int(GS_R/3)
GS_C_third = int(GS_C/3)
GS_R_4th = int(GS_R/4)
GS_C_4th = int(GS_C/4)
grid_spec_size = (GS_R, GS_C)
HPAD = 4
WPAD = 4

# DEFINE SOME HELPFUL COLOR CONSTANTS
default_colors = ['1f77b4', 'ff7f0e', '2ca02c', 'd62728', '9467bd', '8c564b', 'e377c2', '7f7f7f', 'bcbd22', '17becf']


def blend(c1="#d62728", c2="#1f77b4", ratio=0.5):
    """Shorthand for most common form of mixing colors
    >>> COLOR.blend(COLOR.E, COLOR.I, 0.5).hexcode
    '#7b4f6e'
    """
    from spectra import html
    return html(c1).blend(html(c2), ratio)


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    if type(cmap) is str:
        cmap = plt.get_cmap(cmap)
    new_cmap = colors.LinearSegmentedColormap.from_list(
            'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
            cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    >> lighten_color((.3,.55,.1), 2.)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    h, l, s = colorsys.rgb_to_hls(*mc.to_rgb(c))
    l = 1 - amount*(1 - l)
    l = min(1, l)
    l = max(0, l)
    return colorsys.hls_to_rgb(h, l, s)


def categorical_cmap(nc: int, nsc: int, cmap="tab10", continuous=False):
    """
    You may use the HSV system to obtain differently saturated and luminated colors for the same hue. Suppose you
    have at most 10 categories, then the tab10 map can be used to get a certain number of base colors. From those you
    can choose a couple of lighter shades for the subcategories.

    https://stackoverflow.com/questions/47222585/matplotlib-generic-colormap-from-tab10

    :param nc: number of categories
    :param nsc: number of subcategories
    :param cmap: base colormap
    :param continuous: smooth out subcategory colors
    :return: colormap with nc*nsc different colors, where for each category there are nsc colors of same hue
    """
    if nc > plt.get_cmap(cmap).N:
        raise ValueError("Too many categories for colormap.")
    if continuous:
        ccolors = plt.get_cmap(cmap)(np.linspace(0, 1, nc))
    else:
        ccolors = plt.get_cmap(cmap)(np.arange(nc, dtype=int))
    cols = np.zeros((nc*nsc, 3))
    for i, c in enumerate(ccolors):
        chsv = colors.rgb_to_hsv(c[:3])
        arhsv = np.tile(chsv, nsc).reshape(nsc, 3)
        arhsv[:, 1] = np.linspace(chsv[1], 0.25, nsc)
        arhsv[:, 2] = np.linspace(chsv[2], 1, nsc)
        rgb = colors.hsv_to_rgb(arhsv)
        cols[i*nsc:(i + 1)*nsc, :] = rgb
    cmap = colors.ListedColormap(cols)
    return cmap


class COLOR(object):
    """COLOR object for consistent choice of COLORS wherever settings.py is used
    """
    B = '#1f77b4'
    O = '#ff7f0e'
    G = '#2ca02c'
    R = '#d62728'
    Pu = '#9467bd'
    Br = '#8c564b'
    Pi = '#e377c2'
    K = '#7f7f7f'
    Ye = '#bcbd22'
    Cy = '#17becf'
    R1_B2 = '#552f72'
    R1_B3 = '#403580'
    R2_B1 = '#802456'
    R3_B1 = '#bf122b'
    E_I = '#7b4f6e'  # 50-50 mix
    # assign semantic colors
    E = R
    I = B
    A = K
    E2 = O
    NMDA = E2
    AMPA = E
    GABA = I
    C_E_E = Br
    C_I_E = Ye
    C_E_I = Pu
    C_I_I = Cy

    blockGABA = Pi
    benzoGABA = '#98fb98'

    EIA_list = [E, I, K]
    g_list = [NMDA, AMPA, GABA]
    CONN_list = [C_E_E, C_I_E, C_E_I, C_I_I]

    g_dict = dict(NMDA=NMDA, AMPA=AMPA, GABA=GABA)
    RATE_dict = dict(r_E=E, r_I=I, r_all=A)
    CONN_dict = dict(C_E_E=C_E_E, C_I_E=C_I_E, C_E_I=C_E_I, C_I_I=C_I_I,
                     synapse_mon_cee=C_E_E, synapse_mon_cie=C_I_E,
                     synapse_mon_cei=C_E_I, synapse_mon_cii=C_I_I,
                     )

    CONN_BLEND = dict(E=R3_B1, I=R1_B3)

    # to get the appropriate EGABA color in range [-80, -40] call EGABA_SM.to_rgba(<egaba value>)
    EGABA_SM = ScalarMappable(norm=Normalize(-80, -40),
                              cmap="Blues_r")
    EGABA_2_SM = ScalarMappable(norm=TwoSlopeNorm(vmin=-74, vcenter=-60, vmax=-42),
                                cmap='coolwarm')
    G_AMPA_SM = ScalarMappable(norm=Normalize(0, 20),
                               cmap="Reds_r")

    blend = staticmethod(blend)
    truncate_colormap = staticmethod(truncate_colormap)
    lighten_color = staticmethod(lighten_color)
    categorical_cmap = staticmethod(categorical_cmap)

    @staticmethod
    def get_egaba_color(egaba):
        """Helper method to get the color value for a given EGABA.

        Range is [-88, -40]"""
        return COLOR.EGABA_SM.to_rgba(egaba)


benzo_map = {
    'default': COLOR.I
    }
_picro_N = np.round(np.linspace(0, 1, 100, endpoint=False), 2)
for i in _picro_N:
    benzo_map[i] = lighten_color(COLOR.Pi, 1 - i + 0.3*i)  # [1, 0.3]
_benzo_N = np.linspace(1, 10, 19)
for i in _benzo_N:
    benzo_map[i] = lighten_color(COLOR.benzoGABA, 1 + i*2/10)  # [1,2]
benzo_map[1] = benzo_map['default']

# Have colormaps separated into categories:
# http://matplotlib.org/examples/color/colormaps_reference.html
cmaps = {
    'Perceptually Uniform Sequential': [
        'viridis', 'plasma', 'inferno', 'magma'],
    'Sequential':                      [
        'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
        'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
        'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn'],
    'Sequential (2)':                  [
        'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
        'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
        'hot', 'afmhot', 'gist_heat', 'copper'],
    'Diverging':                       [
        'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
        'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic'],
    'Qualitative':                     [
        'Pastel1', 'Pastel2', 'Paired', 'Accent',
        'Dark2', 'Set1', 'Set2', 'Set3',
        'tab10', 'tab20', 'tab20b', 'tab20c'],
    'Miscellaneous':                   [
        'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
        'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'hsv',
        'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar']
    }

n_branches_cmaps = {
    1:  'Greys_r',
    2:  'Blues_r',
    3:  'summer_r',
    4:  'Greens_r',
    6:  'Purples_r',
    8:  'Oranges_r',
    16: 'Reds_r'
    }
n_branches_cmap = {}
e_cmaps = {
    +0.0: 'PuOr',
    -1.0: 'PRGn',
    -2.0: 'RdBu',
    -3.0: 'BrBG',
    -4.0: 'RdGy',
    -5.0: 'GnYl',
    }
e_cmap_keys = sorted(list(e_cmaps.keys()), reverse=True)
kcc2_cmaps = {
    False: 'bone',
    True:  'summer'
    }
t_offset_cmap = {}
syn_dist_cmaps = {
    "diffused":    "Greens",
    "clustered":   "BuGn_r",
    "clustered_n": "BuPu_r",
    }
cmap_dict = {
    'num_synapses':   n_branches_cmaps,
    'e_offsets':      e_cmaps,
    'kcc2':           kcc2_cmaps,
    't_offset':       t_offset_cmap,
    'synapse_dist':   syn_dist_cmaps,
    'num_synapses_e': {},
    'n_e':            {},
    }
_N = 200
_frac = _N//(len(e_cmap_keys) + 1)

for n, _cmap in n_branches_cmaps.items():
    # continuous matplotlib colormap that is truncated
    cmap_dict['num_synapses_e'][n] = COLOR.truncate_colormap(_cmap.replace("_r", ""),
                                                             0.3, 0.9, 256)
    _tmp = cmap_dict['n_e'][n] = {}
    # seaborn palette
    _tmp['palette'] = sns.color_palette(_cmap.replace("_r", ""), n_colors=len(e_cmap_keys) + 1)[1:]
    # _tmp['palette_N'] = sns.color_palette(_cmap.replace("_r", ""), n_colors=_N+_frac)[_frac:]
    # matplotlib colormap
    _tmp['cmap'] = ListedColormap(_tmp['palette'])
    # color for a line
    _tmp['line'] = sns.color_palette(_cmap.replace("_r", ""), n_colors=1)[0]
    n_branches_cmap[n] = _tmp['line']
    # reversal-specific colors
    for idx, _e in enumerate(e_cmap_keys):
        _tmp[_e] = _tmp['palette'][idx]
        _tmp[str(_e)] = _tmp[_e]


# ----------------------------------------------------------------------------------------------------------------------
# VARIABLE NAMES (specifically for rendering in plots)
# ----------------------------------------------------------------------------------------------------------------------

# Helper functions to clean math text
def math_clean(_s: str):
    """Remove all '$' symbols in a string"""
    return _s.replace("$", "")


def math_fix(_s: str):
    """Keep only first and last '$' in a math expression"""
    num_dollar = 0
    first = 0
    last = 0
    for idx, c in enumerate(_s):
        if c == '$':
            num_dollar += 1
            if num_dollar == 1:
                first = idx
            last = idx
    if num_dollar > 2:
        return f"{_s[:first + 1]}{math_clean(_s[first + 1:last])}{_s[last:]}"
    elif num_dollar%2 == 1:
        return f"${math_clean(_s)}$"
    else:
        return _s


# IONS
CL = cl = "$Cl\it{^-} $"
CLI = CL_i = cli = "$[{}]\mathregular{{_i}}$".format(math_clean(CL))
MILLIMOLAR = mM = "mM"
# CHLORIDE
STATIC_CHLORIDE_STR_ABBR = "Static {}".format(cl)
STATIC_CHLORIDE_STR_LONG = "Static Chloride"
DYNAMIC_CHLORIDE_STR_ABBR = "Dynamic {}".format(cl)
DYNAMIC_CHLORIDE_STR_LONG = "Dynamic Chloride"
ECL0 = f'$E{math_clean(CL)}_0$'
ECL = f'$E{math_clean(CL)}$'

TAU_KCC2 = '$\it{\\tau}_{\\rm{KCC2}}$'

# SYNAPSES
GABA = 'GABA'
EGABA = 'EGABA'
E_GABA = '$E_{GABA}$'
I_GABA = '$I_{GABA}$'
G_GABA = '$g_{GABA_{max}}$'
G_AMPA = '$g_{AMPA_{max}}$'
G_NMDA = '$g_{NMDA_{max}}$'
GABAA = GABAa = '$GABA_{A}$'
GABAAR = GABAaR = GABAA + 'R'
DELTA = '$\Delta $'
NABLA = '$\\nabla $'
DELTAEGABA = f'{DELTA}EGABA'
NABLAEGABA = GRADEGABA = f'{NABLA}EGABA'

# map synapses in in synapse.py to nicely formatted names
SYNAPSE_MAP = {
    'inhfluct':            '$inhfluct$',
    'inhfluct_simple':     '$inhfluct_{simple}$',
    'exfluct':             '$exfluct$',
    'exfluct_simple':      '$exfluct_{simple}$',
    'GABAa':               GABAa,
    'ProbAMPANMDA2_RATIO': '$AMPA + NMDA$',
    'ampanmda':        '$AMPA + NMDA$',
    'nmda_Tian':           '$NMDA_{Tian}$',
    'ampa_Tian':           '$AMPA_{Tian}$',
    }
# POPULATIONS
POPULATION_MAP = {
    'E': 'PC',
    'I': 'IN',
    'A': 'Average',
    }
POPULATION_RATE_MAP = {
    'r_E':   'PC',
    'r_I':   'IN',
    'r_all': '$\overline{x}$',
    }
# population-specific tau
TAU_KCC2_E = TAU_KCC2.replace("KCC2", f"KCC2_{{{POPULATION_MAP['E']}}}")
TAU_KCC2_I = TAU_KCC2.replace("KCC2", f"KCC2_{{{POPULATION_MAP['I']}}}")

CONNECTION_MAP = {
    'C_E_E': 'E→E', 'synapse_mon_cee': 'E→E',
    'C_I_E': 'E→I', 'synapse_mon_cie': 'E→I',
    'C_E_I': 'I→E', 'synapse_mon_cei': 'I→E',
    'C_I_I': 'I→I', 'synapse_mon_cii': 'I→I',
    }

VESICLES_SYM = "$x_S$"
VESICLES_TEXT = "vesicle pool"
VESICLES_LONG = f"{VESICLES_TEXT} \n [{VESICLES_SYM}]"
EFFICACY_SYM = "$u_S$"
EFFICACY_TEXT = "synaptic efficacy"
EFFICACY_LONG = f"{EFFICACY_TEXT} \n [{EFFICACY_SYM}]"
WEIGHT_SYM = "$w$"
WEIGHT_TEXT = "resources used"
WEIGHT_LONG = f"{WEIGHT_TEXT} \n [{WEIGHT_SYM}]"

# SLi 	Shunt level DRi / Ri due to activation of single or multiple conductance perturbations; (0%IL%1;
#             dimensionless).
SHUNT_LEVEL = "Shunt Level"
SL = "SL"
INHIBITORY_LEVEL = "Inhibitory Level"
INHIB_LEVEL = IL = "IL"
ILdiff = f"{DELTA}IL"
ILd = "$IL_{d}$"
ILdi = "$IL_{d=i}$"
SLd = "$SL_{d}$"
ILh = "$IL_{h}$"
SLh = "$SL_{h}$"

ACC_IDX_FULL = "Accumulation Index"
ACC_IDX = "AccIdx"

LOCATION_X_ = 'Inhibitory synapse location (X)'
DISTANCE_X_ = 'Distance from junction (X)'

# SLi,j	Attenuation of IL (SLj/ SLi) for a single conductance perturbation at location i;(0%SLi,j%1;
#             dimensionless).
INHIB_LEVEL_ATTENUATION = ILid = "IL attenuation [$IL_{i->d}$]"
# Input resistance at location i;(U).
INPUT_RESISTANCE_i = Ri = "$R_{i}$"
INPUT_RESISTANCE_d = Rd = "$R_{d}$"
INPUT_RESISTANCE_d_star = Rdstar = Rdi = "$R_{d}^{i}$"
# Change in Ri due to synaptic conductance perturbation; (U).
CHANGE_IN_INPUT_RESISTANCE_i = DRi = "$\Delta R_X_i$"
# Voltage attenuation, Vj/Vi, for current perturbation at location i;(0%Ai,j%1; dimensionless).
VOLTAGE_ATTENUATION_i_d = Aid = "$A_{i,d}$"
VOLTAGE_ATTENUATION_d_i = Adi = "$A_{d,i}$"
# Dendritic-to-somatic conductance ratio; (G dendrite/G soma; dimensionless).
RHO = p = "$\rho$"
# Conductance perturbation at location i
CONDUCTANCE_i = gi = "$g_{i}$"
# Membrane Potential
MEMBRANE_POTENTIAL = "Membrane Potential"
MILLIVOLTS = mV = "mV"
VOLTAGE_SYMBOL = Vm = "$Vm$"
RELATIVE_VOLTAGE_SYMBOL = V = "$V$"
VOLTAGE_d = Vd = "$V_{d}$"
VOLTAGE_d_star = Vdstar = Vdi = "$V_{d}^{i}$"
# Distance
MICROMETERS = um = "$\mu m$"
DISTANCE = "Distance"

# Time
TIME = "Time"
MILLISECONDS = ms = 'ms'
SECONDS = s = 's'


# noinspection PyMissingOrEmptyDocstring
def UNITS(text):
    return '(' + text + ')'


# noinspection PyMissingOrEmptyDocstring
def ITALICISE(text):
    return r'\it\{' + text + r'\}'


X_UNIT_OPTIONS = [um, 'X']
X_UNITS = X_UNIT_OPTIONS[1]

IL_config = {
    STATIC_CHLORIDE_STR_ABBR:  {
        'label':    STATIC_CHLORIDE_STR_ABBR,
        'cb_label': 'IL',
        'cmap':     'viridis_r'
        },
    DYNAMIC_CHLORIDE_STR_ABBR: {
        'label':    DYNAMIC_CHLORIDE_STR_ABBR,
        'cb_label': 'IL',
        'cmap':     'viridis_r'
        },
    IL:                        {
        'label':    IL,
        'cb_label': 'IL',
        'cmap':     'viridis_r'
        },
    ILdiff:                    {
        'label':    "IL Difference",
        'cb_label': ILdiff,
        'cmap':     'plasma'
        },
    'EGABA':                   {
        'label':    'EGABA',
        'cb_label': 'mV',
        'cmap':     'winter'
        },
    'relative':                {
        'label':    "Relative IL",
        'cb_label': 'IL',
        'cmap':     'viridis_r'
        },
    }

##
