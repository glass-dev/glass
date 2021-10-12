# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''outputs and analysis'''


__glass__ = [
    'write_cls',
    'cls_from_fields',
]


import numpy as np
import healpy as hp
import os
import logging
from sortcl import enumerate_cls

from .typing import WorkDir, LMax, SampleCls


log = logging.getLogger('glass.analysis')


def cls_from_fields(fields, lmax: LMax = None) -> SampleCls:
    '''compute the sample cls of the given fields

    Only scalar fields can analysed with this function.

    '''

    log.info('computing cls for %d probes...', len(fields))

    # make a flat list of names and fields
    _names = [f'{name}[{i}]' for name, _ in fields.items() for i in range(len(_))]
    _fields = sum(map(list, fields.values()), [])

    log.debug('total number of fields: %d', len(fields))
    for name in _names:
        log.debug('- %s', name)

    # compute the flat list of cls
    _cls = hp.anafast(_fields, lmax=lmax, pol=False, use_pixel_weights=True)

    log.debug('computed %d cls', len(_cls))

    # sort cls into a dict
    cls = {}
    for i, j, cl in enumerate_cls(_cls):
        cls[_names[i], _names[j]] = cl

    # return the dict of sample cls
    return cls


def write_cls(out_cls, workdir: WorkDir = None) -> None:
    '''write cls to file'''

    import fitsio

    if workdir is None:
        log.info('workdir not set, skipping...')
        return

    filename = 'cls.fits'

    log.info('writing "%s"...', filename)

    fits = fitsio.FITS(os.path.join(workdir, filename), fitsio.READWRITE)

    for name, cls in out_cls.items():
        # get the largest lmax for this set of cls
        maxlen = max(len(cl) for cl in cls)

        names, data = [], []
        for (field1, field2), cl in cls.items():
            if cl is not None:
                if len(cl) < maxlen:
                    cl = np.pad(cl, (0, maxlen-len(cl)))
                names.append(f'{field1}-{field2}')
                data.append(cl)

        log.info('- %s: %d cls', name, len(data))

        fits.write(data, extname=name.upper(), names=names)
