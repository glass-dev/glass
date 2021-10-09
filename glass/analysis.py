# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''outputs and analysis'''


__all__ = [
    'write_cls',
]


import numpy as np
import os
import logging

from sortcl import cl_indices

from .typing import WorkDir, Cls, GaussianCls, RegGaussianCls, RegularizedCls


log = logging.getLogger('glass.analysis')


def write_cls(workdir: WorkDir = None,
              cls: Cls = None,
              gaussian_cls: GaussianCls = None,
              reg_gaussian_cls: RegGaussianCls = None,
              regularized_cls: RegularizedCls = None,
              ) -> None:
    '''write cls to file'''

    import fitsio

    if workdir is None:
        log.info('workdir not set, skipping...')
        return

    locals_ = locals()

    filename = 'cls.fits'

    log.info('writing "%s"...', filename)

    fits = fitsio.FITS(os.path.join(workdir, filename), fitsio.READWRITE)

    for name in 'cls', 'gaussian_cls', 'reg_gaussian_cls', 'regularized_cls':
        _cls = locals_[name]
        if _cls is not None:
            # number of fields from number of cls
            n = int((2*len(_cls))**0.5)

            # get the largest lmax for this set of cls
            maxlen = max(len(cl) for cl in _cls)

            names, data = [], []
            for i, j, cl in zip(*cl_indices(n), _cls):
                if cl is not None:
                    if len(cl) < maxlen:
                        cl = np.pad(cl, (0, maxlen-len(cl)))
                    names.append(f'CL-{i}-{j}')
                    data.append(cl)

            log.info('- %s: %d fields, %d cls', name, n, len(data))

            fits.write(data, extname=name.upper(), names=names)
