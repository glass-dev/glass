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

from .core import generator
from .random import generate_lognormal, generate_normal


@generator('zmin, zmax -> wz')
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


@generator('zmin, zmax -> wz')
def mat_wht_redshift():
    '''uniform matter weights in redshift'''
    yield from mat_wht_function(lambda z: np.ones_like(z))


@generator('zmin, zmax -> wz')
def mat_wht_distance(cosmo):
    '''uniform matter weights in comoving distance'''
    yield from mat_wht_function(lambda z: 1/cosmo.e(z))


@generator('zmin, zmax -> wz')
def mat_wht_volume(cosmo):
    '''uniform matter weights in comoving volume'''
    yield from mat_wht_function(lambda z: cosmo.xm(z)**2/cosmo.e(z))


@generator('zmin, zmax -> wz')
def mat_wht_density(cosmo):
    '''uniform matter weights in matter density'''
    yield from mat_wht_function(lambda z: cosmo.rho_m(z)*cosmo.xm(z)**2/cosmo.e(z))


@generator('cl -> delta')
def lognormal_matter(nside, rng=None):
    '''generate lognormal matter fields from Cls'''
    yield from generate_lognormal(nside, shift=1., rng=rng)


@generator('cl -> delta')
def gaussian_matter(nside, rng=None):
    '''generate Gaussian matter fields from Cls'''
    yield from generate_normal(nside, rng=rng)
