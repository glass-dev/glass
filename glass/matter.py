# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''
Matter (:mod:`glass.matter`)
============================

.. currentmodule:: glass.matter

The :mod:`glass.matter` module provides functionality for discretising and
simulating the matter distribution in the universe.

Matter shells
-------------

.. autosummary::
   :template: generator.rst
   :toctree: generated/
   :nosignatures:

   distance_shells
   redshift_shells


Matter weights
--------------

.. autosummary::
   :template: generator.rst
   :toctree: generated/
   :nosignatures:

   MatterWeights
   uniform_weights
   distance_weights
   volume_weights
   density_weights

'''

from collections import namedtuple
import numpy as np


MatterWeights = namedtuple('MatterWeights', ['z', 'w'])
MatterWeights.__doc__ = '''Matter weight functions for shells.'''
MatterWeights.z.__doc__ = '''List of redshift arrays :math:`z`.'''
MatterWeights.w.__doc__ = '''List of weight arrays :math:`w(z)`.'''


def effective_redshifts(weights: MatterWeights):
    '''Compute the effective redshifts of matter weight functions.'''
    return np.array([np.trapz(w*z, z, axis=-1)/np.trapz(w, z, axis=-1)
                     for z, w in zip(weights.z, weights.w)])


def redshift_shells(zmin, zmax, *, dz=None, num=None):
    '''shells with uniform redshift spacing'''
    if (dz is None) == (num is None):
        raise ValueError('exactly one of "dz" or "num" must be given')
    if dz is not None:
        z = np.arange(zmin, np.nextafter(zmax+dz, zmax), dz)
    else:
        z = np.linspace(zmin, zmax, num+1)
    return z


def distance_shells(cosmo, zmin, zmax, *, dx=None, num=None):
    '''shells with uniform comoving distance spacing'''
    if (dx is None) == (num is None):
        raise ValueError('exactly one of "dx" or "num" must be given')
    xmin, xmax = cosmo.dc(zmin), cosmo.dc(zmax)
    if dx is not None:
        x = np.arange(xmin, np.nextafter(xmax+dx, xmax), dx)
    else:
        x = np.linspace(xmin, xmax, num+1)
    return cosmo.dc_inv(x)


def make_weights(shells, wfun, subs=200):
    '''Apply a weight function to a redshift grid for each shell.'''
    za, wa = [], []
    for zmin, zmax in zip(shells, shells[1:]):
        z = np.linspace(zmin, zmax, subs)
        w = wfun(z)
        za.append(z)
        wa.append(w)
    return MatterWeights(za, wa)


def uniform_weights(shells, zlin=None):
    '''Matter weights with uniform distribution in redshift.

    If ``zlin`` is given, the weight ramps up linearly from 0 at z=0 to 1 at
    z=zlin.  This can help prevent numerical issues with some codes for angular
    power spectra.

    '''
    if zlin is None:
        return make_weights(shells, lambda z: np.ones_like(z))
    else:
        return make_weights(shells, lambda z: np.clip(z/zlin, 0, 1))


def distance_weights(shells, cosmo, zlin=None):
    '''Uniform matter weights in comoving distance.

    If ``zlin`` is given, the weight ramps up linearly from 0 at z=0 to its
    value at z=zlin.  This can help prevent numerical issues with some codes for
    angular power spectra.

    '''
    if zlin is None:
        return make_weights(shells, lambda z: 1/cosmo.ef(z))
    else:
        return make_weights(shells, lambda z: np.clip(z/zlin, 0, 1)/cosmo.ef(z))


def volume_weights(shells, cosmo):
    '''Uniform matter weights in comoving volume.'''
    return make_weights(shells, lambda z: cosmo.xm(z)**2/cosmo.ef(z))


def density_weights(shells, cosmo):
    '''Uniform matter weights in matter density.'''
    return make_weights(shells, lambda z: cosmo.rho_m_z(z)*cosmo.xm(z)**2/cosmo.ef(z))
