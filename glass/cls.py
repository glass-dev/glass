# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''module for angular power spectra'''

__all__ = [
    'cls_from_file',
    'cls_from_files',
    'cls_from_pyccl',
    'cls_from_classy',
]


import numpy as np
import logging
from itertools import combinations_with_replacement
from interpcl import interpcl
from sortcl import cl_indices

from .types import RedshiftBins, NumberOfBins, Cosmology, ClsDict, ClsList


log = logging.getLogger('glass.cls')


def collect_cls(fields: list[str],
                cls: ClsDict,
                *,
                allow_missing: bool = False) -> ClsList:
    '''collect cls for a list of field names in healpy order'''

    # total number of fields in each bin
    n = len(fields)

    log.debug('collecting cls for %d fields...', n)

    _cls = []
    for i, j in zip(*cl_indices(n)):
        a, b = fields[i], fields[j]
        if (a, b) in cls:
            log.debug(f'- {a}-{b}')
            _cls.append(cls[a, b])
        elif (b, a) in cls:
            log.debug(f'- {b}-{a}')
            _cls.append(cls[b, a])
        elif allow_missing:
            log.debug(f'- MISSING: {a}-{b}')
            _cls.append(None)
        else:
            raise KeyError(f'missing cls: {a}-{b}')

    log.debug('collected %d cls, of which %d are None', len(_cls), sum(cl is None for cl in _cls))

    return _cls


def cls_from_file(n, file, lmax=None, dipole=True, monopole=False, *, dens=False, lens=False):
    '''read Cls from file

    Read the Cls from `filename` and return an array of cls for all integer
    modes from 0 to `lmax`, or the highest mode in the input file if `lmax` is
    not set.  Missing values are interpolated, including the dipole if
    `dipole` is ``True``, and the monopole if `monopole` is ``True``.

    '''

    # reads the cls with column names
    data = np.genfromtxt(file, names=True, dtype=float, deletechars='')

    # assumes the first column in the file is the multipoles
    l = data[data.dtype.names[0]]

    # assemble all names
    names = []
    if dens:
        names += [f'dens[{i}]' for i in range(n)]
    if lens:
        names += [f'lens[{i}]' for i in range(n)]

    cls = []
    for i, j in zip(*cl_indices(n)):
        cl = data[f'{names[i]}-{names[j]}']
        cl = interpcl(l, cl, lmax=lmax, dipole=dipole, monopole=monopole)
        cls.append(cl)

    return cls


def cls_from_files(pattern: str,
                   fields: dict[str, str],
                   nbins: NumberOfBins,
                   *,
                   lmax: int = None,
                   dipole: bool = True,
                   monopole: bool = False) -> ClsDict:
    '''read cls from multiple individual files'''

    log.debug('reading cls from multiple files')
    log.debug('filename pattern: %s')

    fields_and_bins = [(field, name, i) for field, name in fields.items() for i in range(nbins)]

    log.debug('all fields and bins: %s', ', '.join(str(fb) for fb in fields_and_bins))

    cls = {}
    for (fi, ni, i), (fj, nj, j) in combinations_with_replacement(fields_and_bins, 2):
        filename = pattern.format(name1=ni, bin01=i, bin1=i+1, name2=nj, bin02=j, bin2=j+1)

        log.debug('file for %s[%d]-%s[%d]: %s', fi, i, fj, j, filename)

        try:
            l, cl = np.loadtxt(filename, usecols=[0, 1]).T
        except OSError:
            l, cl = None, None

        if cl is not None:
            log.debug('lmax for %s[%d]-%s[%d]: %s', fi, i, fj, j, l[-1])
            cls[f'{fi}[{i}]', f'{fj}[{j}]'] = interpcl(l, cl, lmax=lmax, dipole=dipole, monopole=monopole)
        else:
            log.debug('file not found')

    return cls


def cls_from_pyccl(fields, lmax, zbins: RedshiftBins, cosmo: Cosmology) -> ClsDict:
    import pyccl

    c = pyccl.Cosmology(h=cosmo.H0/100, Omega_c=cosmo.Om, Omega_b=0.05, sigma8=0.8, n_s=0.96)

    zz = np.arange(0., zbins[-1]+0.001, 0.01)
    bz = np.ones_like(zz)

    # add tracers for every field based on its physical type
    names, tracers = [], []
    for field, phys in fields.items():
        for i, (za, zb) in enumerate(zip(zbins, zbins[1:])):
            nz = ((zz >= za) & (zz < zb)).astype(float)

            if phys == 'matter':
                tracer = pyccl.NumberCountsTracer(c, False, (zz, nz), (zz, bz), None)
            elif phys == 'convergence':
                tracer = pyccl.WeakLensingTracer(c, (zz, nz), True, None, False)
            else:
                raise ValueError(f'cls_from_pyccl: unknown type "{phys}" for field "{field}"')

            names.append(f'{field}[{i}]')
            tracers.append(tracer)

    n = len(tracers)
    l = np.arange(lmax+1)

    cls = {}
    for i, j in zip(*cl_indices(n)):
        cls[names[i], names[j]] = pyccl.angular_cl(c, tracers[i], tracers[j], l)

    return cls


def cls_from_classy(z, lmax, cosmo, *, nonlinear='halofit', return_lensing=False):
    from classy import Class

    # number of bins
    n = len(z) - 1

    # midpoint and half-width of the tophat profiles
    zm = np.add(z[:-1], z[1:])/2
    dz = np.subtract(z[1:], z[:-1])/2

    c = Class()

    output = ['dCl']
    if return_lensing:
        output.append('sCl')

    c.set({
        'output': ', '.join(output),
        'number count contributions': 'density',
        'modes': 's',
        'H0': cosmo.H0,
        'Omega_cdm': cosmo.Om,
        'Omega_k': cosmo.Ok,
        'non linear': nonlinear,
        # 'P_k_max_1/Mpc': 3.0,
        'l_max_lss': lmax,
        'selection': 'tophat',
        'selection_mean': ', '.join(map(str, zm)),
        'selection_width': ', '.join(map(str, dz)),
        'selection_bias': 1.,
        'selection_magnification_bias': 0.,
        'non_diagonal': n-1,
        'l_switch_limber_for_nc_local_over_z': 10000,
        'l_switch_limber_for_nc_los_over_z': 2000,
        'lensing': 'no',
    })

    c.compute()

    result = c.density_cl()

    ell, dd = result['ell'], result['dd']

    assert np.all(ell == np.arange(lmax+1))

    # reorder the cls
    cls = []
    for i, j in zip(*cl_indices(n)):
        k = i*(2*n - i - 1)//2 + j
        cls.append(dd[k])

    if return_lensing:
        ll = result['ll']
        lens_cls = []
        for i, j in zip(*cl_indices(n)):
            k = i*(2*n - i - 1)//2 + j
            lens_cls.append(ell**2*(ell+1)**2/4 * ll[k])

    return cls if not return_lensing else (cls, lens_cls)
