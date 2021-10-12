# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''outputs and analysis'''


__all__ = [
    'write_cls',
]


import numpy as np
import os
import logging

from .typing import WorkDir


log = logging.getLogger('glass.analysis')


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
