# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''
Shells (:mod:`glass.shells`)
============================

.. currentmodule:: glass.shells

The :mod:`glass.shells` module provides functions for the definition of
matter shells, i.e. the radial discretisation of the light cone.


Window functions
----------------

.. autofunction:: tophat_windows


Window function tools
---------------------

.. autofunction:: restrict
.. autofunction:: partition
.. autofunction:: effective_redshifts


Redshift grids
--------------

.. autofunction:: redshift_grid
.. autofunction:: distance_grid


Weight functions
----------------

.. autofunction:: distance_weight
.. autofunction:: volume_weight
.. autofunction:: density_weight

'''

import warnings
import numpy as np

from .math import ndinterp


# type checking
from typing import (Union, Sequence, List, Tuple, Optional, Callable,
                    TYPE_CHECKING)
from numpy.typing import ArrayLike
if TYPE_CHECKING:
    from cosmology import Cosmology

# types
ArrayLike1D = Union[Sequence[float], np.ndarray]
WeightFunc = Callable[[ArrayLike1D], np.ndarray]


def distance_weight(z: ArrayLike, cosmo: 'Cosmology') -> np.ndarray:
    '''Uniform weight in comoving distance.'''
    return 1/cosmo.ef(z)


def volume_weight(z: ArrayLike, cosmo: 'Cosmology') -> np.ndarray:
    '''Uniform weight in comoving volume.'''
    return cosmo.xm(z)**2/cosmo.ef(z)


def density_weight(z: ArrayLike, cosmo: 'Cosmology') -> np.ndarray:
    '''Uniform weight in matter density.'''
    return cosmo.rho_m_z(z)*cosmo.xm(z)**2/cosmo.ef(z)


def tophat_windows(zbins: ArrayLike1D, dz: float = 1e-3,
                   wfunc: Optional[WeightFunc] = None
                   ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    '''Tophat window functions from the given redshift bin edges.

    Uses the *N+1* given redshifts as bin edges to construct *N* tophat
    window functions.  The redshifts of the windows have linear spacing
    approximately equal to ``dz``.

    An optional weight function :math:`w(z)` can be given using
    ``wfunc``; it is applied to the tophat windows.

    Parameters
    ----------
    zbins : (N+1,) array_like
        Redshift bin edges for the tophat window functions.
    dz : float, optional
        Approximate spacing of the redshift grid.
    wfunc : callable, optional
        If given, a weight function to be applied to the window
        functions.

    Returns
    -------
    zs, ws : (N,) list of array_like
        List of window functions.

    '''
    if len(zbins) < 2:
        raise ValueError('zbins must have at least two entries')
    if zbins[0] != 0:
        warnings.warn('first tophat window does not start at redshift zero')

    wf: WeightFunc
    if wfunc is not None:
        wf = wfunc
    else:
        wf = np.ones_like

    zs, ws = [], []
    for zmin, zmax in zip(zbins, zbins[1:]):
        n = max(round((zmax - zmin)/dz), 2)
        z = np.linspace(zmin, zmax, n)
        zs.append(z)
        ws.append(wf(z))
    return zs, ws


def effective_redshifts(zs: Sequence[ArrayLike1D], ws: Sequence[ArrayLike1D]
                        ) -> np.ndarray:
    '''Compute the effective redshifts of window functions.'''
    return np.ndarray([np.trapz(w*z, z, axis=-1)/np.trapz(w, z, axis=-1)
                       for z, w in zip(zs, ws)])


def restrict(z: ArrayLike1D, f: ArrayLike1D, za: ArrayLike1D, wa: ArrayLike1D
             ) -> Tuple[np.ndarray, np.ndarray]:
    '''Restrict a function to a redshift window.

    Multiply the function :math:`f(z)` by a window function :math:`w(z)`
    to produce :math:`w(z) f(z)` over the support of :math:`w`.

    The function :math:`f(z)` is given by redshifts ``z`` of shape
    *(N,)* and function values ``f`` of shape *(..., N)*, with any
    number of leading axes allowed.

    The window function :math:`w(z)` is given by redshifts ``zr`` and
    values ``wr``, which must have the same shape *(M,)*.

    The restriction has redshifts that are the union of the redshifts of
    the function and window over the support of the window.
    Intermediate function values are found by linear interpolation

    Parameters
    ----------
    z, f : array_like
        The function to be restricted.
    za, wa : array_like
        The window function for the restriction.

    Returns
    -------
    zr, fr : array
        The restricted function.

    '''

    z_ = np.compress(np.greater(z, za[0]) & np.less(z, za[-1]), z)
    zr = np.union1d(za, z_)
    fr = ndinterp(zr, z, f, left=0., right=0.) * ndinterp(zr, za, wa)
    return zr, fr


def partition(z: ArrayLike1D, f: ArrayLike1D, zs: Sequence[ArrayLike1D],
              ws: Sequence[ArrayLike1D]
              ) -> Tuple[np.ndarray, np.ndarray]:
    '''Partition a function by a sequence of windows.

    Partitions the given function into a sequence of functions
    restricted to each window function.

    The function :math:`f(z)` is given by redshifts ``z`` of shape
    *(N,)* and function values ``f`` of shape *(..., N)*, with any
    number of leading axes allowed.

    The window functions are pairs ``(zs[i], ws[i])`` of redshifts and
    values where both ``zs[i]`` and ``ws[i]`` must have the same shape
    *(Mi,)*.  Redshifts ``zs[i]`` and sizes *Mi* can differ for
    different values of *i*.

    The partitions has redshifts that are the union of the redshifts of
    the function and each window over the support of said window.
    Intermediate function values are found by linear interpolation

    Parameters
    ----------
    z, f : array_like
        The function to be partitioned.
    zs, ws : sequence of array_like
        Ordered sequence of window functions for the partition.

    Returns
    -------
    zp, fp : list of array
        The partitioned functions, ordered as the given windows.

    '''

    zp, fp = [], []
    for za, wa in zip(zs, ws):
        zr, fr = restrict(z, f, za, wa)
        zp.append(zr)
        fp.append(fr)
    return zp, fp


def redshift_grid(zmin, zmax, *, dz=None, num=None):
    '''Redshift grid with uniform spacing in redshift.'''
    if dz is not None and num is None:
        z = np.arange(zmin, np.nextafter(zmax+dz, zmax), dz)
    elif dz is None and num is not None:
        z = np.linspace(zmin, zmax, num+1)
    else:
        raise ValueError('exactly one of "dz" or "num" must be given')
    return z


def distance_grid(cosmo, zmin, zmax, *, dx=None, num=None):
    '''Redshift grid with uniform spacing in comoving distance.'''
    xmin, xmax = cosmo.dc(zmin), cosmo.dc(zmax)
    if dx is not None and num is None:
        x = np.arange(xmin, np.nextafter(xmax+dx, xmax), dx)
    elif dx is None and num is not None:
        x = np.linspace(xmin, xmax, num+1)
    else:
        raise ValueError('exactly one of "dx" or "num" must be given')
    return cosmo.dc_inv(x)
