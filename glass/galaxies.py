# author: Arthur Loureiro <arthur.loureiro@ucl.ac.uk>
# license: MIT
'''galaxy tracers'''

import numpy as np
import healpy as hp
import logging

from glass.typing import MatterFields, GalaxyTracers, Visibility

log = logging.getLogger('glass.galaxies')

__glass__ = [
    'galaxies_from_matter',
]


def galaxies_from_matter(number_of_galaxies_arcmin2,
                         delta: MatterFields,
                         visibility: Visibility = None,
                         ) -> GalaxyTracers:
    r'''galaxy number density field computed from the matter density field

    .. math::

        \N_g^i(\Omega) = \text{Poisson}\[n(z_i) \, m(\Omega)\, \left(\delta_m + 1 \right)\]

    '''

    log.debug('computing galaxy tracers')

    # get the bin axes of the delta array, last axis is pixels
    *nbins, npix = np.shape(delta)
    nside = hp.npix2nside(npix)

    # add pixel axis to number densities
    numb = np.expand_dims(number_of_galaxies_arcmin2, np.ndim(number_of_galaxies_arcmin2))

    # check if the number density and delta shapes are compatible
    try:
        np.broadcast(delta, numb)
    except ValueError:
        raise ValueError('matter fields and number densities have incompatible shape') from None

    pixel_area = hp.nside2pixarea(nside)

    # converts it to galaxies/pixel:
    numb = numb*(60**2)*((180**2)/(np.pi**2))*pixel_area

    # first line creates a new array, other lines modify in place
    lamda = delta + 1
    lamda *= numb
    if visibility is not None:
        lamda *= visibility

    return np.random.poisson(lamda)
