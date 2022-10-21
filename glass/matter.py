# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''module for large scale structure'''

import numpy as np

from .generator import generator
from .random import generate_lognormal, generate_normal

from .cosmology import ZMIN, ZMAX

# variable definitions
WZ = 'matter weight function'
CL = 'angular matter power spectrum'
DELTA = 'matter density contrast'


@generator(receives=(ZMIN, ZMAX), yields=WZ)
def mat_wht_function(w):
    '''generate matter weights from a weight function'''
    wz = None
    while True:
        try:
            zmin, zmax = yield wz
        except GeneratorExit:
            break

        z = np.linspace(zmin, zmax, 100)
        wz = (z, w(z))


@generator(receives=(ZMIN, ZMAX), yields=WZ)
def mat_wht_redshift(zlin=None):
    '''uniform matter weights in redshift

    If ``zlin`` is given, the weight ramps up linearly from 0 at z=0 to 1 at
    z=zlin.  This can help prevent numerical issues with some codes for angular
    power spectra.

    '''
    if zlin is None:
        yield from mat_wht_function(lambda z: np.ones_like(z))
    else:
        yield from mat_wht_function(lambda z: np.clip(z/zlin, None, 1))


@generator(receives=(ZMIN, ZMAX), yields=WZ)
def mat_wht_distance(cosmo, zlin=None):
    '''uniform matter weights in comoving distance

    If ``zlin`` is given, the weight ramps up linearly from 0 at z=0 to its
    value at z=zlin.  This can help prevent numerical issues with some codes for
    angular power spectra.

    '''
    if zlin is None:
        yield from mat_wht_function(lambda z: 1/cosmo.ef(z))
    else:
        yield from mat_wht_function(lambda z: np.clip(z/zlin, None, 1)/cosmo.ef(z))


@generator(receives=(ZMIN, ZMAX), yields=WZ)
def mat_wht_volume(cosmo):
    '''uniform matter weights in comoving volume'''
    yield from mat_wht_function(lambda z: cosmo.xm(z)**2/cosmo.ef(z))


@generator(receives=(ZMIN, ZMAX), yields=WZ)
def mat_wht_density(cosmo):
    '''uniform matter weights in matter density'''
    yield from mat_wht_function(lambda z: cosmo.rho_m_z(z)*cosmo.xm(z)**2/cosmo.ef(z))


@generator(receives=CL, yields=DELTA)
def lognormal_matter(nside, rng=None):
    '''generate lognormal matter fields from Cls'''
    yield from generate_lognormal(nside, shift=1., rng=rng)


@generator(receives=CL, yields=DELTA)
def gaussian_matter(nside, rng=None):
    '''generate Gaussian matter fields from Cls'''
    yield from generate_normal(nside, rng=rng)
