# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''module for weak gravitational lensing'''

import logging
import numpy as np
import healpy as hp

from .generator import generator, optional
from .util import restrict_interval

from .matter import DELTA

log = logging.getLogger(__name__)

# variable definitions
ZSRC = 'weak lensing source redshift'
'''The source redshift for which the lensing fields are evaluated.'''
KAPPA = 'weak lensing convergence'
'''The convergence field from weak lensing, commonly called :math:`\\kappa`, for
the source redshift :data:`ZSRC`.'''
GAMMA = 'weak lensing shear'
'''The shear field from weak lensing, commonly called :math:`\\gamma`, for the
source redshift :data:`ZSRC`.'''
KAPPA_BAR = 'weak lensing mean convergence over distribution'
'''The integrated convergence field of a source distribution :math:`n(z)`.'''
GAMMA_BAR = 'weak lensing mean shear over distribution'
'''The integrated shear field of a source distribution :math:`n(z)`.'''


@generator(receives=KAPPA, yields=GAMMA)
def gen_shear(lmax=None):
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
    gamma = None

    while True:
        # return the shear field and wait for the next convergence field
        # break the loop when asked to exit the generator
        try:
            kappa = yield gamma
        except GeneratorExit:
            break

        alm = hp.map2alm(kappa, lmax=lmax, pol=False, use_pixel_weights=True)

        # initialise everything on the first iteration
        if gamma is None:
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
        gamma = hp.alm2map_spin([alm, blm], nside, 2, lmax)


@generator(
    receives=DELTA,
    yields=(ZSRC, KAPPA))
def gen_convergence(mweights, cosmo):
    '''convergence from integrated matter shells'''

    f = 3*cosmo.omega_m/2

    # initial yield to get the first mass plane
    delta3 = yield

    # set up initial values for recurrence
    z2 = z3 = x3 = m3 = 0
    r23 = 1
    kappa2 = np.zeros_like(delta3)
    kappa3 = np.zeros_like(delta3)
    delta2 = 0

    # go through matter shells, new values for delta3 are obtained at the end
    for z, w in zip(mweights.z, mweights.w):

        # redshift "mass" in weight
        m2, m3 = m3, np.trapz(w, z)

        # redshifts of source planes
        z1, z2, z3 = z2, z3, np.trapz(w*z, z)/m3

        # extrapolation law
        x2, x3 = x3, cosmo.xm(z3)
        r12, (r13, r23) = r23, cosmo.xm([z1, z2], z3)/x3
        t3 = r13/r12

        # lensing weight of matter plane to be added
        w3 = f * x2*r23 * (1 + z2)/cosmo.ef(z2) * m2

        # cycle convergence planes
        # normally: kappa1, kappa2, kappa3 = kappa2, kappa3, <empty>
        # but then we set: kappa3 = (1-t)*kappa1 + ...
        # so we can set kappa3 to previous kappa2 and modify in place
        kappa2, kappa3 = kappa3, kappa2

        # compute next convergence plane in place of last
        kappa3 *= 1 - t3
        kappa3 += t3*kappa2
        kappa3 += w3*delta2

        # output some statistics
        log.info('zsrc: %f', z3)
        log.info('κbar: %f', np.mean(kappa3))
        log.info('κmin: %f', np.min(kappa3))
        log.info('κmax: %f', np.max(kappa3))
        log.info('κrms: %f', np.sqrt(np.mean(np.square(kappa3))))

        # keep current matter field for next iteration
        delta2 = delta3

        # yield convergence and return new matter field
        delta3 = yield z3, kappa3


@generator(
    receives=(ZSRC, optional(KAPPA), optional(GAMMA)),
    yields=(KAPPA_BAR, GAMMA_BAR))
def gen_lensing_dist(z, nz, cosmo):
    '''generate weak lensing maps for source distributions

    This generator takes a single or multiple (via leading axes) redshift
    distribution(s) of sources and computes their integrated mean lensing maps.

    The generator receives the convergence and/or shear maps for each source
    plane.  It then averages the lensing maps by interpolation between source
    planes [1]_, and yields the result up to the last source plane received.

    Parameters
    ----------
    z, nz : array_like
        The redshift distribution(s) of sources.  The redshifts ``z`` must be a
        1D array.  The density of sources ``nz`` must be at least a 1D array,
        with the last axis matching ``z``.  Leading axes define multiple source
        distributions.
    cosmo : Cosmology
        A cosmology instance to obtain distance functions.

    Receives
    --------
    zsrc : float
        Source plane redshift.
    kappa, gamma1, gamma2 : array_like, optional
        HEALPix maps of convergence and/or shear.  Unavailable maps are ignored.

    Yields
    ------
    kappa_bar, gamma1_bar, gamma2_bar : array_like
        Integrated mean lensing maps, or ``None`` where there are no input maps.
        The maps have leading axes matching the distribution.

    References
    ----------
    .. [1] Tessore et al., in prep.

    '''

    # check inputs
    if np.ndim(z) != 1:
        raise TypeError('redshifts must be one-dimensional')
    if np.ndim(nz) == 0:
        raise TypeError('distribution must be at least one-dimensional')
    *sh, sz = np.shape(nz)
    if sz != len(z):
        raise TypeError('redshift axis mismatch')

    # helper function to get normalisation
    # takes the leading distribution axes into account
    def norm(nz_, z_):
        return np.expand_dims(np.trapz(nz_, z_), np.ndim(nz_)-1)

    # normalise distributios
    nz = np.divide(nz, norm(nz, z))

    # total accumulated weight for each distribution
    # shape needs to match leading axes of distributions
    w = np.zeros((*sh, 1))

    # initial lensing plane
    # give small redshift > 0 to work around division by zero
    zsrc = 1e-6
    kap = gam1 = gam2 = 0

    # initial yield
    kap_bar = gam1_bar = gam2_bar = None
    result = None

    # wait for next source plane and return result, or stop on exit
    while True:
        zsrc_, kap_, gam1_, gam2_ = zsrc, kap, gam1, gam2
        try:
            zsrc, kap, (gam1, gam2) = yield result
        except GeneratorExit:
            break

        # integrated maps are initialised to zero on first iteration
        # integrated maps have leading axes for distributions
        if kap is not None and kap_bar is None:
            kap_bar = np.zeros((*sh, np.size(kap)))
        if gam1 is not None and gam1_bar is None:
            gam1_bar = np.zeros((*sh, np.size(gam1)))
        if gam2 is not None and gam2_bar is None:
            gam2_bar = np.zeros((*sh, np.size(gam2)))

        # get the restriction of n(z) to the interval between source planes
        nz_, z_ = restrict_interval(nz, z, zsrc_, zsrc)

        # get integrated weight and update total weight
        # then normalise n(z)
        w_ = norm(nz_, z_)
        w += w_
        nz_ /= w_

        # integrate interpolation factor against source distributions
        t = norm(cosmo.xm(zsrc_, z_)/cosmo.xm(z_)*nz_, z_)
        t /= cosmo.xm(zsrc_, zsrc)/cosmo.xm(zsrc)

        # interpolate convergence planes and integrate over distributions
        for m, m_, m_bar in (kap, kap_, kap_bar), (gam1, gam1_, gam1_bar), (gam2, gam2_, gam2_bar):
            if m is not None:
                m_bar += (t*m + (1-t)*m_ - m_bar)*(w_/w)

        result = kap_bar, (gam1_bar, gam2_bar)
