# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''internal module for user functions'''

import logging
from contextlib import contextmanager
import numpy as np


def zrange(zmin, zmax, *, dz=None, num=None):
    '''returns redshift slices with uniform redshift spacing'''
    if (dz is None) == (num is None):
        raise ValueError('exactly one of "dz" or "num" must be given')
    if dz is not None:
        z = np.arange(zmin, zmax, dz)
    else:
        z = np.linspace(zmin, zmax, num+1)
    return z


def xrange(cosmo, zmin, zmax, *, dx=None, num=None):
    '''returns redshift slices with uniform comoving distance spacing'''
    if (dx is None) == (num is None):
        raise ValueError('exactly one of "dx" or "num" must be given')
    xmin, xmax = cosmo.dc(zmin), cosmo.dc(zmax)
    if dx is not None:
        x = np.arange(xmin, xmax, dx)
    else:
        x = np.linspace(xmin, xmax, num+1)
    return cosmo.dc_inv(x)


@contextmanager
def logger(level='info'):
    '''context manager for logging'''
    level = str(level).upper()
    log = logging.getLogger('glass')
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    handler.setLevel(level)
    old_level = log.level
    log.addHandler(handler)
    log.setLevel(level)
    try:
        yield log
    finally:
        log.setLevel(old_level)
        log.removeHandler(handler)
        handler.close()
