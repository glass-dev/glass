# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''weak gravitational lensing'''

import logging
import numpy as np
import healpy as hp

from ._generator import generator


log = logging.getLogger('glass.lensing')


@generator('kappa -> gamma1, gamma2')
def shear(lmax=None):
    r'''weak lensing shear from convergence

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

    The shear modes can therefore be obtained via the convergence, or
    directly from the deflection potential.

    '''

    # set to None for the initial iteration
    gamma1, gamma2 = None, None

    while True:
        # return the shear field and wait for the next convergence field
        # break the loop when asked to exit the generator
        try:
            kappa = yield gamma1, gamma2
        except GeneratorExit:
            break

        alm = hp.map2alm(kappa, lmax=lmax, pol=False, use_pixel_weights=True)

        # initialise everything on the first iteration
        if gamma1 is None:
            nside = hp.get_nside(kappa)
            lmax = hp.Alm.getlmax(len(alm))

            log.debug('nside from kappa: %d', nside)
            log.debug('lmax from alms: %s', lmax)

            blm = np.zeros_like(alm)

            l = np.arange(lmax+1)
            fl = np.sqrt((l+2)*(l+1)*l*(l-1))
            fl /= np.clip(l*(l+1), 1, None)
            fl *= -1

        # convert convergence to shear modes
        hp.almxfl(alm, fl, inplace=True)
        gamma1, gamma2 = hp.alm2map_spin([alm, blm], nside, 2, lmax)


@generator('zmin, zmax, delta -> kappa')
def convergence(cosmo, weight='midpoint'):
    '''convergence from integrated matter shells'''

    # prefactor
    f = 3*cosmo.Om/2

    # these are the different ways in which the matter can be weighted
    if weight == 'midpoint':

        log.info('will use midpoint lensing weights')

        # consider the lensing weight constant, and integrate the matter
        def w(zi, zj, zk):
            z = cosmo.xc_inv(np.mean(cosmo.xc([zi, zj])))
            x = cosmo.xm(z) if z != 0 else 1.
            return cosmo.xm(z, zk)/cosmo.xm(zk)*(1 + z)*cosmo.vc(zi, zj)/x

    elif weight == 'integrated':

        log.info('will use integrated lensing weights')

        # consider the matter constant, and integrate the lensing weight
        def w(zi, zj, zk):
            z = np.linspace(zi, zj, 100)
            f = cosmo.xm(z)
            f *= cosmo.xm(z, zk)/cosmo.xm(zk)
            f *= (1 + z)/cosmo.e(z)
            return np.trapz(f, z)

    else:
        raise ValueError(f'invalid value for weight: {weight}')

    # initial yield
    kappa3 = None

    # return convergence and get new matter shell, or stop on exit
    while True:
        try:
            zmin, zmax, delta23 = yield kappa3
        except GeneratorExit:
            break

        # set up variables on first iteration
        if kappa3 is None:
            kappa2 = np.zeros_like(delta23)
            kappa3 = np.zeros_like(delta23)
            delta12 = 0
            z2 = z3 = zmin
            r23 = 1
            w33 = 0

        # deal with non-contiguous redshift intervals
        if z3 != zmin:
            raise NotImplementedError('shells must be contiguous')

        # cycle convergence planes
        # normally: kappa1, kappa2, kappa3 = kappa2, kappa3, <empty>
        # but then we set: kappa3 = (1-t)*kappa1 + ...
        # so we can set kappa3 to previous kappa2 and modify in place
        kappa2, kappa3 = kappa3, kappa2

        # redshifts of source planes
        z1, z2, z3 = z2, z3, zmax

        # extrapolation law
        r12 = r23
        r13, r23 = cosmo.xm([z1, z2], z3)/cosmo.xm(z3)
        t123 = r13/r12

        # weights for the lensing recurrence
        w22 = w33
        w23 = w(z1, z2, z3)
        w33 = w(z2, z3, z3)

        # compute next convergence plane in place of last
        kappa3 *= 1 - t123
        kappa3 += t123*kappa2
        kappa3 += f*(w23 - t123*w22)*delta12
        kappa3 += f*w33*delta23

        # output some statistics
        log.info('κbar: %f', np.mean(kappa3))
        log.info('κmin: %f', np.min(kappa3))
        log.info('κmax: %f', np.max(kappa3))
        log.info('κrms: %f', np.sqrt(np.mean(np.square(kappa3))))

        # before losing it, keep current matter slice for next round
        delta12 = delta23
