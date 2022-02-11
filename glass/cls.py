'''module for angular power spectra'''

import logging
import numpy as np


log = logging.getLogger('glass.cls')


def cls_from_pyccl(lmax, cosmo, dz=0.01):
    '''compute theory cls with pyccl'''

    from pyccl import Cosmology, NumberCountsTracer, angular_cl

    c = Cosmology(h=cosmo.H0/100, Omega_c=cosmo.Om, Omega_b=0.05, sigma8=0.8, n_s=0.96)

    l = np.arange(lmax+1)

    tr = cls = None

    while True:
        try:
            zmin, zmax = yield cls
        except GeneratorExit:
            break

        zz = np.linspace(zmin, zmax, int(np.ceil((zmax-zmin)/dz)+0.1))
        bz = np.ones_like(zz)
        nz = 1/cosmo.e(zz)

        tr_ = tr
        tr = NumberCountsTracer(c, False, (zz, nz), (zz, bz), None)

        cl_ = angular_cl(c, tr, tr_, l) if tr_ is not None else None
        cl = angular_cl(c, tr, tr, l)

        cls = [cl, cl_]
