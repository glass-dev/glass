# author: Arthur Loureiro <arthur.loureiro@ucl.ac.uk>
# license: MIT
'''galaxy tracers'''

import numpy as np
import healpy as hp
import logging
import os

from glass.typing import MatterFields, GalaxyFields

log = logging.getLogger('glass.galaxies')

__glass__ = [
    'galaxies_from_matter',
]


def galaxies_from_matter(delta: MatterFields,
                         number_of_galaxies_arcmin2,
                         visibility: Visibility = None,
                         ) -> GalaxyFields:
    r'''galaxy number density field computed from the matter density field

    .. math::

        \N_g^i(\Omega) = \text{Poisson}\[n(z_i) \, m(\Omega)\, \left(\delta_m + 1 \right)\]

    '''

    log.debug('computing galaxy tracers')

    # get the bin axes of the delta array, last axis is pixels
    *nbins, npix = np.shape(delta)
    nside = hp.npix2nside(npix)

    # broadcast the number density over the bins, add one axis for pixels
    try:
        # the transverse is there because it doesn't allow me to make a nbins,npix shape
        numb = np.broadcast_to(number_of_galaxies_arcmin2, (npix, nbins)).T
    except ValueError:
        raise ValueError('matter fields and number densities have incompatible shape: ') from None

    pixel_area = hp.nside2pixarea(nside)

    # converts it to galaxies/pixel:
    numb = numb*(60**2)*((180**2)/(np.pi**2))*pixel_area

    # first line creates a new array, other lines modify in place
    lamda = delta + 1
    lamda *= numb
    if visibility is not None:
        lamda *= visibility

    return np.random.poisson(lamda)
