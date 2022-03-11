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

from ._generator import generator


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
