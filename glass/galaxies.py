# author: Arthur Loureiro <arthur.loureiro@ucl.ac.uk>
# license: MIT
'''galaxy tracers'''

import numpy as np
import healpy as hp
import logging
import os

from glass.typing import MatterFields, Annotated, Fields, NumberOfBins, NSide, GalaxyFields

log = logging.getLogger('glass.galaxies')

__all__ = [
    'galaxies_from_matter']

def galaxies_from_matter(delta: MatterFields, 
                         nbins: NumberOfBins,
                         nside: NSide,
                         number_of_galaxies_arcmin2,
                         mask_file = None) -> GalaxyFields:
    r'''galaxy number density field computed from the matter density field
    
    .. math::
        
        \N_g^i(\Omega) = \text{Poisson}\[n(z_i) \, m(\Omega)\, \left(\delta_m + 1 \right)\]

    '''

    log.debug('computing galaxy tracers')

    if len(number_of_galaxies_arcmin2) != 1:

        assert len(number_of_galaxies_arcmin2) == nbins, "Number of bins doesn't match len(number_of_galaxies_arcmin2)! "

    if mask_file is not None:
        log.debug(f'Reading the given mask from {mask_file}')

        assert os.path.isfile(mask_file) == True, f"[!] Mask not found in {mask_file}"

        mask =  hp.read_map(mask_file)
 
    else:
        mask = np.ones(hp.nside2npix(nside))

    pixel_area = hp.nside2pixarea(nside)

    pois = []

    for i in range(nbins):
        numb_gal_steradians = number_of_galaxies_arcmin2[i]*(60**2)*((180**2)/(np.pi**2))
        lamb = (numb_gal_steradians*pixel_area)*mask*(delta[i] + 1)
        map_temp = np.where(mask != 0, np.random.poisson(lamb), 0)
        pois.append(map_temp)

    return np.asanyarray(pois)