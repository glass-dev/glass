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
