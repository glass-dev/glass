# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''module for CCL interoperability'''

import numpy as np

import pyccl

from .._generator import generator


@generator('zmin, zmax -> cl')
def ccl_matter_cls(cosmo, lmax):
    '''generator for the matter angular power spectrum from CCL'''

    l = np.arange(lmax+1)

    tr = cls = None

    while True:
        try:
            zmin, zmax = yield cls
        except GeneratorExit:
            break

        zz = np.linspace(zmin, zmax, 100)
        bz = np.ones_like(zz)
        aa = 1/(1 + zz)
        nz = cosmo.comoving_angular_distance(aa)**2/cosmo.h_over_h0(aa)

        tr_, tr = tr, pyccl.NumberCountsTracer(cosmo, False, (zz, nz), (zz, bz), None)

        cl = pyccl.angular_cl(cosmo, tr, tr, l)
        cl_ = pyccl.angular_cl(cosmo, tr, tr_, l) if tr_ is not None else None

        cls = [cl, cl_]
