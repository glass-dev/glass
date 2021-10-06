# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''weak gravitational lensing'''


__all__ = [
    'normal_convergence',
    'lognormal_convergence',
    'convergence_from_matter',
    'shear_from_convergence',
]


import numpy as np
import healpy as hp
import logging

from .types import Matter, Convergence, Shear, Random, ArrayLike, NumberOfBins, RedshiftBins, Cosmology
from .random import NormalField, LognormalField


log = logging.getLogger('glass.lensing')


def kappa0_hilbert11(z):
    r'''return the shift of a lognormal convergence field

    Uses the fit (38) of Hilbert et al. (2011) to compute the lognormal field
    shift (i.e. the minimum kappa value).

    '''

    return 0.008*z*(1 + 3.625*z*(1. - 0.272414*z*(1. - 0.0822785*z)))


def normal_convergence(nbins: NumberOfBins) -> Convergence[Random]:
    '''convergence field following a normal distribution'''

    return [NormalField()]*nbins


def lognormal_convergence(zbins: RedshiftBins) -> Convergence[Random]:
    '''convergence field following a lognormal distribution'''

    z = np.add(zbins[:-1], zbins[1:])/2
    kappa0 = kappa0_hilbert11(z)
    return [LognormalField(shift=k0) for k0 in kappa0]


def convergence_from_matter(delta: Matter[ArrayLike],
                            zbins: RedshiftBins,
                            cosmo: Cosmology,
                            *,
                            growth: bool = False) -> Convergence[ArrayLike]:
    '''compute convergence fields from projection of the matter field

    For reference, see e.g. Chapter 6.2 of Schneider, Kochanek & Wambsganss
    (2006).

    '''

    nsub = 100

    n = len(zbins) - 1

    def kern(zi, zj):
        k = (zi >= zj).astype(float)
        k *= cosmo.xm(zj)
        k *= cosmo.xm(zj, zi)
        k /= cosmo.xm(zi) + 1e-100
        k *= 1 + zj
        k /= cosmo.e(zj)
        if growth:
            gj = cosmo.gf(zj)
            gn = np.trapz(gj, zj)/(zj[-1] - zj[0])
            k *= gj/gn
        return k

    # matrix for projection
    km = np.zeros((n, n))
    for i in range(n):
        zi = np.linspace(zbins[i], zbins[i+1], nsub)
        wi = zbins[i+1] - zbins[i]
        for j in range(i+1):
            zj = np.linspace(zbins[j], zbins[j+1], nsub)
            km[i, j] = np.trapz(np.trapz(kern(zi[:, np.newaxis], zj), zj), zi)/wi
    km *= 3/2 * cosmo.Om

    kappa = np.matmul(km, delta)

    return kappa


def shear_from_convergence(kappa: Convergence[ArrayLike]) -> Shear[ArrayLike]:
    r'''weak lensing shear field from convergence

    Notes
    -----
    The shear field is computed from the convergence or deflection potential in
    the following way.

    Define the spin-raising and spin-lowering operators of the spin-weighted
    spherical harmonics as

    .. math::

        \eth {}_sY_{lm}
        = +\sqrt{(l-s)(l+s+1)} \, {}_{s+1}Y_{lm} \;, \\
        \bar{\eth} {}_sY_{lm}
        = -\sqrt{(l+s)(l-s+1)} \, {}_{s-1}Y_{lm} \;.

    The convergence field :math:`\kappa` is related to the deflection potential
    field :math:`\phi` as

    .. math::

        2 \kappa = \eth\bar{\eth} \, \phi = \bar{\eth}\eth \, \phi \;.

    The convergence modes :math:`\kappa_{lm}` are hence related to the
    deflection potential modes :math:`\phi_{lm}` as

    .. math::

        2 \kappa_{lm} = -l \, (l+1) \, \phi_{lm} \;.

    The shear field :math:`\gamma` is related to the deflection potential field
    as

    .. math::

        2 \gamma = \eth\eth \, \phi
        \quad\text{or}\quad
        2 \gamma = \bar{\eth}\bar{\eth} \, \phi \;,

    depending on the definition of the shear field spin weight as :math:`2` or
    :math:`-2`.  In either case, the shear modes :math:`\gamma_{lm}` are
    related to the deflection potential modes as

    .. math::

        2 \gamma_{lm} = \sqrt{(l+2) \, (l+1) \, l \, (l-1)} \, \phi_{lm} \;.

    The shear modes can therefore be obtained from the convergence.

    '''

    log.debug('compute shear maps from convergence maps')

    size = np.shape(kappa)

    log.debug('kappa shape: %s', size)

    nside = hp.npix2nside(size[-1])

    log.debug('nside from kappa: %d', nside)

    log.debug('computing kappa alms from kappa maps')

    alms = np.atleast_2d(hp.map2alm(kappa, pol=False, use_pixel_weights=True))

    size = np.shape(alms)

    log.debug('alms shape: %s', size)

    lmax = hp.Alm.getlmax(size[-1])

    log.debug('lmax from alms: %s', lmax)

    l = np.arange(lmax+1)

    log.debug('turning kappa alms into gamma alms')

    fl = np.sqrt((l+2)*(l+1)*l*(l-1))
    fl /= np.clip(l*(l+1), 1, None)
    fl *= -1
    for alm in alms:
        hp.almxfl(alm, fl, inplace=True)

    blm = np.zeros_like(alms[0])

    log.debug('transforming gamma alms to gamma maps')

    gamma1 = np.empty_like(kappa)
    gamma2 = np.empty_like(kappa)
    for i, alm in enumerate(alms):
        g1, g2 = hp.alm2map_spin([alm, blm], nside, 2, lmax)
        gamma1[i, :] = g1
        gamma2[i, :] = g2

    return gamma1, gamma2
