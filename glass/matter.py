# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''
===================================
Matter fields (:mod:`glass.matter`)
===================================

.. currentmodule:: glass.matter

Generators
==========

Random fields
-------------

.. autosummary::
   :template: generator.rst
   :toctree: generated/

   gaussian_matter
   lognormal_matter


Weight functions
----------------

.. autosummary::
   :template: generator.rst
   :toctree: generated/

   mat_wht_function
   mat_wht_redshift
   mat_wht_distance
   mat_wht_volume
   mat_wht_density

'''

import numpy as np

from .generator import generator
from .random import generate_lognormal, generate_normal


@generator(receives=('zmin', 'zmax'), yields='wz')
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


@generator(receives=('zmin', 'zmax'), yields='wz')
def mat_wht_redshift():
    '''uniform matter weights in redshift

    The weight ramps up linearly from 0 at z=0 to 1 at z=0.1 to prevent
    numerical issues with some codes for angular power spectra.

    '''
    yield from mat_wht_function(lambda z: np.clip(z/0.1, None, 1))


@generator(receives=('zmin', 'zmax'), yields='wz')
def mat_wht_distance(cosmo):
    '''uniform matter weights in comoving distance

    The weight ramps up linearly from 0 at z=0 to its value at z=0.1 to prevent
    numerical issues with some codes for angular power spectra.

    '''
    yield from mat_wht_function(lambda z: np.clip(z/0.1, None, 1)/cosmo.e(z))


@generator(receives=('zmin', 'zmax'), yields='wz')
def mat_wht_volume(cosmo):
    '''uniform matter weights in comoving volume'''
    yield from mat_wht_function(lambda z: cosmo.xm(z)**2/cosmo.e(z))


@generator(receives=('zmin', 'zmax'), yields='wz')
def mat_wht_density(cosmo):
    '''uniform matter weights in matter density'''
    yield from mat_wht_function(lambda z: cosmo.rho_m(z)*cosmo.xm(z)**2/cosmo.e(z))


@generator(receives='cl', yields='delta')
def lognormal_matter(nside, rng=None):
    '''generate lognormal matter fields from Cls'''
    yield from generate_lognormal(nside, shift=1., rng=rng)


@generator(receives='cl', yields='delta')
def gaussian_matter(nside, rng=None):
    '''generate Gaussian matter fields from Cls'''
    yield from generate_normal(nside, rng=rng)
