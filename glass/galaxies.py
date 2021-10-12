# author: Arthur Loureiro <arthur.loureiro@ucl.ac.uk>
# license: MIT
'''galaxy tracers'''

import numpy as np
import healpy as hp
import logging
import os

from glass.typing import MatterFields, GalaxyFields

log = logging.getLogger('glass.galaxies')

__all__ = [
    'galaxies_from_matter']


def galaxies_from_matter(delta: MatterFields,
                         number_of_galaxies_arcmin2,
                         mask_file=None) -> GalaxyFields:
    r'''galaxy number density field computed from the matter density field

    .. math::

        \N_g^i(\Omega) = \text{Poisson}\[n(z_i) \, m(\Omega)\, \left(\delta_m + 1 \right)\]

    '''

    log.debug('computing galaxy tracers')

    # get the bin axes of the delta array, last axis is pixels
    nbins = np.shape(delta)[0]
    npix = np.shape(delta)[1]
    nside = hp.npix2nside(npix)

    # broadcast the number density over the bins, add one axis for pixels
    try:
        # the transverse is there because it doesn't allow me to make a nbins,npix shape
        numb = np.broadcast_to(number_of_galaxies_arcmin2, (npix, nbins)).T
    except ValueError:
        raise ValueError('matter fields and number densities have incompatible shape: ') from None

    if mask_file is None:
        mask = None
    else:
        # reads the mask from file.
        # FIXME: this needs to go somewhere else
        log.debug(f'Reading the given mask from {mask_file}')

        assert os.path.isfile(mask_file) is True, f"[!] Mask not found in {mask_file}"

        mask = hp.read_map(mask_file)

    pixel_area = hp.nside2pixarea(nside)

    # converts it to galaxies/pixel:
    numb = numb*(60**2)*((180**2)/(np.pi**2))*pixel_area

    # first line creates a new array, other lines modify in place
    lamda = delta + 1
    lamda *= numb
    if mask is not None:
        lamda *= mask

    return np.random.poisson(lamda)
