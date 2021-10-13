# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''module for observational effects'''

__glass__ = [
    'visibility_from_file',
]


import numpy as np
import healpy as hp
import logging

from .typing import NSide, Visibility


log = logging.getLogger('glass.observations')


def visibility_from_file(filename: str,
                         hdu: int = 1,
                         *,
                         nside: NSide = None,
                         field: tuple[int] = None,
                         partial: bool = False,
                         ) -> Visibility:
    '''load visibility for fields from file'''

    log.info('reading visibility maps from %s...', filename)

    # read the maps
    vis = hp.read_map(filename, hdu=hdu, field=field, partial=partial, dtype=float)

    log.debug('shape of visibility maps: %s', vis.shape)

    log.info('read visibility for %d field(s)', len(vis) if vis.ndim > 1 else 1)

    # replace UNSEEN values by zero
    unseen = np.where(vis == hp.UNSEEN)
    if len(unseen[0]):
        log.info('setting %d unseen pixels to zero...', len(unseen[0]))

        vis[unseen] = 0

    # convert to nside if given
    if nside is not None and hp.nside2npix(nside) != vis.shape[-1]:
        log.info('changing map resolution to NSIDE=%d...', nside)

        vis = hp.ud_grade(vis, nside)

        log.debug('new shape of visibility maps: %s', vis.shape)

    # return the visibility maps
    return vis
