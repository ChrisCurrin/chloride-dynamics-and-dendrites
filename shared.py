# coding=utf-8
from __future__ import print_function, division

import glob
import hashlib
import logging
import os
import sys
import seaborn as sns
import numpy as np
import pandas as pd

from utils import settings
from utils.use_files import create_dir

# set up logging to file - see previous section for more details

# logging.basicConfig(level=settings.LOG_LEVEL,
#                     format='%(asctime)s %(name)-12s [%(filename)-15s:%(lineno)4d] %(levelname)-8s \t %(message)s',
#                     datefmt='%m-%d %H:%M')
logger = logging.getLogger('shared')
# to log to file, add:
# filename='/temp/myapp.log',
# filemode='w'

t_vec = None
initialised = False
__KWARGS__ = {}


def INIT(reinit=False):
    global initialised, t_vec, __KWARGS__
    if initialised and not reinit:
        return True
    else:
        initialised = True
    import os
    from neuron import h
    h.load_file("stdrun.hoc")

    # compile mod files
    if __mod_files_changed(settings.MOD_PATH) or settings.NEURON_RECOMPILE:
        # noinspection PyUnresolvedReferences
        from utils.use_files import main
        output = main(path=settings.MOD_PATH, mod=True)
        if "Error" in str(output):
            raise Exception("MOD FILES not compiled successfully")
    # load mod files
    h.nrn_load_dll(settings.NRNMECH_PATH)
    # load hoc files including usefulFns.hoc
    for hoc_file in glob.glob(settings.HOC_PATH + "/*.hoc"):
        h.load_file(hoc_file.replace("\\", "/"))
    # show GUI
    if settings.NEURON_GUI:
        # noinspection PyUnresolvedReferences
        from neuron import gui
        # h.showV()
        h.showRunControl()
        # h.topology()

    # general properties
    h.celsius = 37
    h.v_init = -65
    h.random_stream_offset_ = settings.RANDOM_STREAM_OFFSET
    logger.info("celsius={} and v_init={}".format(h.celsius, h.v_init))
    t_vec = h.Vector()
    np.random.seed(settings.RANDOM_SEED)

    __KWARGS__ = {}
    env_var(celsius=h.celsius, v_init=h.v_init)


def env_var(**kwargs):
    if kwargs:
        for k, v in kwargs.items():
            __KWARGS__[k] = v
    return __KWARGS__


def show_n(n=1, ion=False):
    import matplotlib.pyplot as plt
    for i in plt.get_fignums():
        if i > n:
            plt.close(i)
    if ion:
        plt.ion()
    plt.show()


def hashable(cls):
    def __hash__(self):
        return hashlib.md5(str(self).encode('utf-8')).hexdigest()

    def hash_extra(self, extra=""):
        full_str = str(self) + extra
        return hashlib.md5(full_str.encode('utf-8')).hexdigest()

    cls.__hash__ = __hash__
    cls.hash_extra = hash_extra
    return cls


def is_dvv_saved(serializable_obj, extra):
    logger.debug(serializable_obj)
    logger.debug(extra)
    full_hash = serializable_obj.hash_extra(extra)
    create_dir("./temp")
    fname = "./temp/{}.h5".format(full_hash)
    if os.path.isfile(fname):
        return True, fname
    return False, full_hash


def load_dvv(fname):
    logger.info("loading from file ({})".format(fname))
    df_v = pd.read_hdf(fname, 'df_v')
    df_v_star = pd.read_hdf(fname, 'df_v_star')
    input_events = pd.read_hdf(fname, 'input_events')
    try:
        df_sl = pd.read_hdf(fname, 'df_sl')
    except KeyError:
        df_sl = None
    try:
        df_ecl = pd.read_hdf(fname, 'df_ecl')
    except KeyError:
        df_ecl = None
    return df_v, df_v_star, input_events, df_sl, df_ecl


def save_dvv(serializable_obj, extra, df_v, df_v_star, input_events, df_sl=None, df_ecl=None):
    full_str = str(serializable_obj) + extra
    full_hash = serializable_obj.hash_extra(extra)
    fname = "./temp/{}.h5".format(full_hash)
    logger.info("saving to file ({})".format(fname))
    df_v.to_hdf(fname, key='df_v', mode='w')  # new file
    df_v_star.to_hdf(fname, key='df_v_star')
    input_events.to_hdf(fname, key='input_events')
    if df_sl is not None:
        df_sl.to_hdf(fname, key='df_sl')
    if df_ecl is not None:
        df_ecl.to_hdf(fname, key='df_ecl')
    with open(fname.replace("h5", "txt"), 'w') as arg_file:
        arg_file.write(full_str)


def __mod_files_changed(path=settings.MOD_PATH):
    md5_files = glob.glob(path + "/hash.md5")
    if len(md5_files) == 0:
        old_md5 = ''
    elif len(md5_files) == 1:
        with open(md5_files[0]) as f:
            old_md5 = f.read()
    else:
        raise BaseException("Too many hash files")

    new_md5_hash = hashlib.md5()
    for mod_file in glob.glob(path + "/*.mod"):
        new_md5_hash.update(__md5(mod_file).encode('utf-8'))
    new_md5 = new_md5_hash.hexdigest()
    if new_md5 == old_md5:
        return False
    else:
        # there are changes
        with open(path + "/hash.md5", 'w') as hash_file:
            hash_file.write(new_md5)
        logger.info("there were changes in the mod file directory")
        return True


def __md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(2 ** 20), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()
