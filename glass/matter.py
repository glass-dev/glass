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
   effective_redshifts
   uniform_weights
   distance_weights
   volume_weights
   density_weights

'''

from collections import namedtuple
from typing import Sequence, Optional, Callable, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from cosmology import Cosmology

MatterWeights = namedtuple('MatterWeights', ['z', 'w'])
MatterWeights.__doc__ = '''Matter weight functions for shells.'''
MatterWeights.z.__doc__ = '''List of redshift arrays :math:`z`.'''
MatterWeights.w.__doc__ = '''List of weight arrays :math:`w(z)`.'''


def effective_redshifts(weights: MatterWeights) -> np.ndarray:
    '''Compute the effective redshifts of matter weight functions.'''
    return np.array([np.trapz(w*z, z, axis=-1)/np.trapz(w, z, axis=-1)
                     for z, w in zip(weights.z, weights.w)])


def redshift_shells(zmin: float, zmax: float, *, dz: Optional[float] = None,
                    num: Optional[int] = None) -> np.ndarray:
    '''shells with uniform redshift spacing'''
    z: np.ndarray
    if dz is not None and num is None:
        z = np.arange(zmin, np.nextafter(zmax+dz, zmax), dz)
    elif dz is None and num is not None:
        z = np.linspace(zmin, zmax, num+1)
    else:
        raise ValueError('exactly one of "dz" or "num" must be given')
    return z


def distance_shells(cosmo: 'Cosmology', zmin: float, zmax: float, *,
                    dx: Optional[float] = None, num: Optional[int] = None
                    ) -> np.ndarray:
    '''shells with uniform comoving distance spacing'''
    x: np.ndarray
    xmin, xmax = cosmo.dc(zmin), cosmo.dc(zmax)
    if dx is not None and num is None:
        x = np.arange(xmin, np.nextafter(xmax+dx, xmax), dx)
    elif dx is None and num is not None:
        x = np.linspace(xmin, xmax, num+1)
    else:
        raise ValueError('exactly one of "dx" or "num" must be given')
    return cosmo.dc_inv(x)


def make_weights(shells: Sequence[float],
                 wfun: Callable[[np.ndarray], np.ndarray], subs: int = 200
                 ) -> MatterWeights:
    '''Apply a weight function to a redshift grid for each shell.'''
    za, wa = [], []
    for zmin, zmax in zip(shells, shells[1:]):
        z = np.linspace(zmin, zmax, subs)
        w = wfun(z)
        za.append(z)
        wa.append(w)
    return MatterWeights(za, wa)


def uniform_weights(shells: Sequence[float], zlin: Optional[float] = None
                    ) -> MatterWeights:
    '''Matter weights with uniform distribution in redshift.

    If ``zlin`` is given, the weight ramps up linearly from 0 at z=0 to 1 at
    z=zlin.  This can help prevent numerical issues with some codes for angular
    power spectra.

    '''
    if zlin is None:
        def wfun(z: np.ndarray) -> np.ndarray:
            return np.ones_like(z)
    else:
        def wfun(z: np.ndarray) -> np.ndarray:
            return np.clip(z/zlin, 0, 1)

    return make_weights(shells, wfun)


def distance_weights(shells: Sequence[float], cosmo: 'Cosmology',
                     zlin: Optional[float] = None) -> MatterWeights:
    '''Uniform matter weights in comoving distance.

    If ``zlin`` is given, the weight ramps up linearly from 0 at z=0 to its
    value at z=zlin.  This can help prevent numerical issues with some codes for
    angular power spectra.

    '''
    if zlin is None:
        def wfun(z: np.ndarray) -> np.ndarray:
            return 1/cosmo.ef(z)
    else:
        def wfun(z: np.ndarray) -> np.ndarray:
            return np.clip(z/zlin, 0, 1)/cosmo.ef(z)

    return make_weights(shells, wfun)


def volume_weights(shells: Sequence[float], cosmo: 'Cosmology'
                   ) -> MatterWeights:
    '''Uniform matter weights in comoving volume.'''
    def wfun(z: np.ndarray) -> np.ndarray:
        return cosmo.xm(z)**2/cosmo.ef(z)

    return make_weights(shells, wfun)


def density_weights(shells: Sequence[float], cosmo: 'Cosmology'
                    ) -> MatterWeights:
    '''Uniform matter weights in matter density.'''
    def wfun(z: np.ndarray) -> np.ndarray:
        return cosmo.rho_m_z(z)*cosmo.xm(z)**2/cosmo.ef(z)

    return make_weights(shells, wfun)
