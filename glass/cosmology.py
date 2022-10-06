# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''module for cosmology'''

import logging
from collections.abc import Iterator, Iterable
import numpy as np

from .generator import generator

logger = logging.getLogger(__name__)

# variable definitions
ZMIN = 'lower redshift bound'
'''Redshift of the lower boundary of the current shell.'''
ZMAX = 'upper redshift bound'
'''Redshift of the upper boundary of the current shell.'''


@generator(yields=(ZMIN, ZMAX))
def zgen(z):
    '''generator for contiguous redshift intervals from a redshift array'''
    if isinstance(z, Iterable):
        z = iter(z)
    elif not isinstance(z, Iterator):
        raise TypeError('must be iterable or iterator')
    zmin, zmax = None, next(z)
    yield
    while True:
        try:
            zmin, zmax = zmax, next(z)
        except StopIteration:
            break
        logger.info('zmin: %f', zmin)
        logger.info('zmax: %f', zmax)
        yield zmin, zmax


@generator(yields=(ZMIN, ZMAX))
def zspace(zmin, zmax, *, dz=None, num=None):
    '''generator for redshift intervals with uniform redshift spacing'''
    logger.info('redshifts between %g and %g', zmin, zmax)
    if (dz is None) == (num is None):
        raise ValueError('exactly one of "dz" or "num" must be given')
    if dz is not None:
        logger.info('redshift spacing: %g', dz)
        z = np.arange(zmin, np.nextafter(zmax+dz, zmax), dz)
    else:
        logger.info('number of intervals: %d', num)
        z = np.linspace(zmin, zmax, num+1)
    yield from zgen(z)


@generator(yields=(ZMIN, ZMAX))
def xspace(cosmo, zmin, zmax, *, dx=None, num=None):
    '''genrator for redshift intervals with uniform comoving distance spacing'''
    logger.info('redshifts between %g and %g', zmin, zmax)
    if (dx is None) == (num is None):
        raise ValueError('exactly one of "dx" or "num" must be given')
    xmin, xmax = cosmo.dc(zmin), cosmo.dc(zmax)
    if dx is not None:
        logger.info('comoving distance spacing: %g', dx)
        x = np.arange(xmin, np.nextafter(xmax+dx, xmax), dx)
    else:
        logger.info('number of intervals: %d', num)
        x = np.linspace(xmin, xmax, num+1)
    z = cosmo.dc_inv(x)
    yield from zgen(z)
