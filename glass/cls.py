'''module for angular power spectra'''

import logging
import numpy as np


log = logging.getLogger('glass.cls')


def cls_from_pyccl(lmax, cosmo, dz=1e-4, fix_corr=False):
    '''compute theory cls with pyccl'''

    from pyccl import Cosmology, NumberCountsTracer, angular_cl

    c = Cosmology(h=cosmo.h, Omega_c=cosmo.Om, Omega_b=0.05, sigma8=0.8, n_s=0.96)

    l = np.arange(lmax+1)

    tr = cls = None

    while True:
        try:
            zmin, zmax = yield cls
        except GeneratorExit:
            break

        zz = np.linspace(zmin, zmax, int(np.ceil((zmax-zmin)/dz)+0.1))
        bz = np.ones_like(zz)
        nz = cosmo.dvc(zz)

        tr_, tr = tr, NumberCountsTracer(c, False, (zz, nz), (zz, bz), None)

        if fix_corr is False:
            cl = angular_cl(c, tr, tr, l)
            cl_ = angular_cl(c, tr, tr_, l) if tr_ is not None else None
        else:
            cl_ = np.sqrt(cl, out=cl) if tr_ is not None else None
            cl = angular_cl(c, tr, tr, l)
            if cl_ is not None:
                cl_ *= np.sqrt(cl)
                cl_ *= fix_corr

        cls = [cl, cl_]
