# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''weak gravitational lensing'''

import logging
import numpy as np
import healpy as hp

from ._generator import generator


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


@generator('zmin, zmax, delta -> kappa')
def convergence_from_matter(cosmo):
    '''compute convergence fields from projection of the matter fields'''

    # prefactor
    f = 3*cosmo.Om/2

    # get first redshift slice
    zmin, zmax, delta3 = yield

    # initial values
    kappa2 = np.zeros_like(delta3)
    kappa3 = np.zeros_like(delta3)
    delta2 = 0
    z2 = z3 = 0
    r23 = 1
    x3 = v3 = 1

    # loop redshift slices
    while True:
        # cycle convergence planes
        # normally: kappa1, kappa2, kappa3 = kappa2, kappa3, <empty>
        # but then we set: kappa3 = (1-t)*kappa1 + ...
        # so we can set kappa3 to previous kappa2 and modify in place
        kappa2, kappa3 = kappa3, kappa2

        # redshifts of lensing planes
        z1, z2, z3 = z2, z3, cosmo.xc_inv(np.mean(cosmo.xc([zmin, zmax])))

        # transverse comoving distances
        x2, x3 = x3, cosmo.xm(z3)

        # distance ratio law
        r12 = r23
        r13, r23 = cosmo.xm([z1, z2], z3)/x3
        t = r13/r12

        # comoving volumes of redshift slices
        v2, v3 = v3, cosmo.vc(zmin, zmax)

        # compute next convergence plane in place of last
        kappa3 *= 1-t
        kappa3 += t*kappa2
        kappa3 += f*(1 + z2)*r23*v2/x2*delta2

        # output some statistics
        log.info('zsrc: %f', z3)
        log.info('κbar: %f', np.mean(kappa3))
        log.info('κmin: %f', np.min(kappa3))
        log.info('κmax: %f', np.max(kappa3))
        log.info('κrms: %f', np.sqrt(np.mean(np.square(kappa3))))

        # before losing it, keep current matter slice for next round
        delta2 = delta3

        # return and wait for next redshift slice, or stop
        try:
            zmin, zmax, delta3 = yield kappa3
        except GeneratorExit:
            break
