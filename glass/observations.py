# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''
========================================
Observations (:mod:`glass.observations`)
========================================

.. currentmodule:: glass.observations

Generators
==========

Visibility
----------

.. autosummary::
   :template: generator.rst
   :toctree: generated/

   vis_constant


Other
=====

Visibility
----------

.. autosummary::
   :toctree: generated/

   vmap_galactic_ecliptic

'''

import logging
import numpy as np
import healpy as hp
import math

from .core import generator
from .util import cumtrapz


log = logging.getLogger(__name__)


@generator('-> visibility')
def vis_constant(m, nside=None):
    '''constant visibility map

    Makes a copy of the given :term:`visibility map` and yields it on every
    iteration.  If a ``nside`` parameter is given, the map is resampled to that
    resolution.

    Parameters
    ----------
    m : array_like
        A HEALPix :term:`visibility map`.  The input is not checked.
    nside : int, optional
        If given, resample map to the resolution of the simulation.

    Yields
    ------
    m : array_like
        A copy of the :term:`visibility map` ``m``, as given.

    '''

    # make a copy of the input at the given resolution
    if nside is not None and hp.get_nside(m) != nside:
        log.info('visibility map: NSIDE=%d to %d', hp.get_nside(m), nside)
        m = hp.ud_grade(m, nside)
    else:
        log.info('visibility map: NSIDE=%d', nside)
        m = np.copy(m)

    # yield on every iteration, or stop on exit
    while True:
        try:
            yield m
        except GeneratorExit:
            break


def vmap_galactic_ecliptic(nside, galactic=(30, 90), ecliptic=(20, 80)):
    '''visibility map masking galactic and ecliptic plane

    This function returns a :term:`visibility map` that blocks out stripes for
    the galactic and ecliptic planes.  The location of the stripes is set with
    optional parameters.

    Parameters
    ----------
    nside : int
        The NSIDE parameter of the resulting HEALPix map.
    galactic, ecliptic : (2,) tuple of float
        The location of the galactic and ecliptic plane in their respective
        coordinate systems.

    Returns
    -------
    vis : array_like
        A HEALPix :term:`visibility map`.

    Raises
    ------
    TypeError
        If the ``galactic`` or ``ecliptic`` arguments are not pairs of numbers.

    '''
    if np.ndim(galactic) != 1 or len(galactic) != 2:
        raise TypeError('galactic stripe must be a pair of numbers')
    if np.ndim(ecliptic) != 1 or len(ecliptic) != 2:
        raise TypeError('ecliptic stripe must be a pair of numbers')

    m = np.ones(hp.nside2npix(nside))
    m[hp.query_strip(nside, *galactic)] = 0
    m = hp.Rotator(coord='GC').rotate_map_pixel(m)
    m[hp.query_strip(nside, *ecliptic)] = 0
    m = hp.Rotator(coord='CE').rotate_map_pixel(m)

    return m


def fixed_zbins(zmin, zmax, *, nbins=None, dz=None):
    '''tomographic redshift bins of fixed size

    This function creates contiguous tomographic redshift bins of fixed size.
    It takes either the number or size of the bins.

    Parameters
    ----------
    zmin, zmax : float
        Extent of the redshift binning.
    nbins : int, optional
        Number of redshift bins.  Only one of ``nbins`` and ``dz`` can be given.
    dz : float, optional
        Size of redshift bin.  Only one of ``nbins`` and ``dz`` can be given.

    Returns
    -------
    zbins : list of tuple of float
        List of redshift bin edges.
    '''

    if (nbins is None) != (dz is None):
        raise ValueError('either nbins or dz must be given')

    if nbins is not None:
        zbinedges = np.linspace(z[0], z[-1], nbins+1)
    if dz is not None:
        zbinedges = np.arange(z[0], z[-1], dz)

    return list(zip(zbinedges, zbinedges[1:]))


def equal_dens_zbins(z, dndz, nbins):
    '''equal density tomographic redshift bins

    This function subdivides a source redshift distribution into ``nbins``
    tomographic redshift bins with equal density.

    Parameters
    ----------
    z, dndz : array_like
        The source redshift distribution. Must be one-dimensional.
    nbins : int
        Number of redshift bins.

    Returns
    -------
    zbins : list of tuple of float
        List of redshift bin edges.

    '''
    # compute the normalised cumulative distribution function
    # first compute the cumulative integral (by trapezoidal rule)
    # then normalise: the first z is at CDF = 0, the last z at CDF = 1
    # interpolate to find the z values at CDF = i/nbins for i = 0, ..., nbins
    cuml_dndz = cumtrapz(dndz, z)
    cuml_dndz /= cuml_dndz[[-1]]
    zbinedges = np.interp(np.linspace(0, 1, nbins+1), cuml_dndz, z)

    return list(zip(zbinedges, zbinedges[1:]))


def tomo_gaussian_error(z, dndz, sigma_z, zbins):
    '''tomographic redshift bins with a Gaussian redshift error

    This function takes a dndz, redshift bin edges, and applies a
    gaussian error as in Refregier & Amara, 2007
    Parameters
    ----------
    z, dndz : array_like
        The true source redshift distribution. Must be one-dimensional.
    sigma_0 : float
        Redshift error in the tomographic binning at zero redshift.
    zbins: tuple
        a tuple with the redshift tomographic bins edges.
        Can be generated using glass.observation.equal_dens_zbins or
        glass.observation.equal_spaced_zbins

    '''
    # converting zbins into an array:
    zbins = np.asanyarray(zbins)

    # bin edges and adds a new axis
    z_lower = zbins[:, 0, np.newaxis]
    z_upper = zbins[:, 1, np.newaxis]

    # we need a vectorised version of the error function:
    erf = np.vectorize(math.erf, otypes=(float,))

    # compute the probabilities that redshift z ends up between za and zb
    # then apply probability as weights to given dndz
    # leading axis corresponds to the different bins
    sz = 2**0.5*sigma_0*(1 + z)
    binned_dndz = erf((z - z_lower)/sz)
    binned_dndz += erf((z - z_upper)/sz)
    binned_dndz /= 1 + erf(z/sz)
    binned_dndz *= dndz

    return binned_dndz
