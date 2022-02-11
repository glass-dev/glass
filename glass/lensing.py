# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''weak gravitational lensing'''


import logging
import numpy as np
import healpy as hp
from scipy.integrate import dblquad


log = logging.getLogger('glass.lensing')


def gamma_from_kappa(kappa):
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

    *nmap, npix = np.shape(kappa)
    nside = hp.npix2nside(npix)

    log.debug('nside from kappa: %d', nside)

    log.debug('computing kappa alms from kappa maps')

    alm = hp.map2alm(kappa, pol=False, use_pixel_weights=True)

    *_, nalm = np.shape(alm)

    log.debug('number of alms: %s', nalm)

    lmax = hp.Alm.getlmax(nalm)

    log.debug('lmax from alms: %s', lmax)

    log.debug('turning kappa alms into gamma alms')

    l = np.arange(lmax+1)
    fl = np.sqrt((l+2)*(l+1)*l*(l-1))
    fl /= np.clip(l*(l+1), 1, None)
    fl *= -1
    for i in np.ndindex(*nmap):
        hp.almxfl(alm[i], fl, inplace=True)

    log.debug('transforming gamma alms to gamma maps')

    blm = None
    gamma1, gamma2 = np.empty((2, *nmap, npix))
    for i in np.ndindex(*nmap):
        if blm is None:
            blm = np.zeros_like(alm[i])
        g1, g2 = hp.alm2map_spin([alm[i], blm], nside, 2, lmax)
        gamma1[i] = g1
        gamma2[i] = g2

    return gamma1, gamma2


def convergence_from_matter(cosmo, pz=None, *, auto=True):
    '''compute convergence fields from projection of the matter fields'''

    # prefactor
    f = 3*cosmo.Om/2

    # uniform weighting in comoving distance if not given
    if pz is None:
        pz = lambda z: 1/cosmo.e(z)

    # integrand for auto-convergence
    def autow(z_, z):
        return pz(z)*cosmo.xm(z_)*cosmo.xm(z_, z)/cosmo.xm(z)*(1 + z_)/cosmo.e(z_)

    if auto:
        log.info('computing auto-convergence')
    else:
        log.info('ignoring auto-convergence')

    # get first redshift slice
    zmin, zmax, delta = yield

    # initial values
    kappa = np.zeros_like(delta)
    kappa_ = np.zeros_like(delta)
    delta_ = 0
    z2 = z3 = 0
    r23 = 1

    # auto-convergence factor, remains zero if not computed
    a = 0

    # loop redshift slices
    while True:
        # midpoints of previous, current, and next redshift slice
        z1, z2, z3 = z2, z3, cosmo.xc_inv(np.mean(cosmo.xc([zmin, zmax])))

        # distance ratio law
        r12 = r23
        r13, r23 = cosmo.xm([z1, z2], z3)/cosmo.xm(z3)
        t = r13/r12

        # compute next convergence plane in place of last
        # subtract the auto-convergence if previously added (... - a*t)
        kappa_ *= 1-t
        kappa_ += t*kappa
        kappa_ += f*((1 + z2)*cosmo.xm(z2)*r23*cosmo.xc(zmin, zmax) - a*t)*delta_

        # cycle convergence planes
        kappa_, kappa = kappa, kappa_

        # add auto-convergence of slice if requested
        if auto:
            a, aerr = dblquad(autow, zmin, zmax, zmin, lambda z: z)
            kappa += a*f*delta

        # keep current matter slice for next convergence
        delta_ = delta

        # return and wait for next redshift slice, or stop
        try:
            zmin, zmax, delta = yield kappa
        except GeneratorExit:
            break
