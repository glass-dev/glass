# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''
================================
Galaxies (:mod:`glass.galaxies`)
================================

.. currentmodule:: glass.galaxies

Generators
==========

Distribution
------------

.. autosummary::
   :template: generator.rst
   :toctree: generated/

   gal_dist_fullsky


Ellipticity
-----------

.. autosummary::
   :template: generator.rst
   :toctree: generated/

   gal_ellip_ryden04
   gal_shear_interp


Other
=====

Ellipticity
-----------

.. autosummary::
   :toctree: generated/

   ellipticity_ryden04

'''

import logging
import numpy as np
import healpy as hp

from ._generator import generator
from ._utils import ARCMIN2_SPHERE, restrict_interval, cumtrapz, triaxial_axis_ratio


log = logging.getLogger(__name__)


@generator('zmin, zmax, delta, visibility? -> ngal, gal_z, gal_pop, gal_lon, gal_lat')
def gal_dist_fullsky(z, dndz, bz=None, *, bias='log-linear', rng=None):
    '''sample galaxy distributions from density, bias, and visibility

    The galaxies are sampled by rejection sampling over the full sky.  This is
    potentially very inefficient if the visible sky is small.

    Distributions for multiple populations of galaxies (e.g. different
    photometric redshift bins) can be passed as a leading axis of the ``dndz``
    array.  However, the spatial distribution of galaxies (i.e. the bias and
    visibility) does not take different galaxy populations into account.  For
    individual biases and visibilities, use a :func:`~glass.group` with one
    :func:`galaxy_positions` generator per source population.

    '''

    # get default RNG if not given
    if rng is None:
        rng = np.random.default_rng()

    # make sure valid number count distributions are passed
    if np.ndim(z) != 1:
        raise TypeError('redshifts must be 1d array')
    if not np.all(np.diff(z) > 0):
        raise ValueError('redshifts are not strictly increasing')
    if not np.all(np.greater_equal(dndz, 0)):
        raise ValueError('negative number counts in distribution')
    if bz is not None and np.ndim(bz) > 1:
        raise TypeError('bias must be number or 1d array')

    # get axes of the arrays
    # if redshift axes mismatch, try to broadcast to shape of z
    # the leading axis of dndz is the populations to sample
    az, = np.shape(z)
    *apop, az_ = np.shape(dndz)
    if az_ != az:
        dndz = np.broadcast_to(dndz, (*apop, az), subok=True)
    if bz is not None and np.shape(bz) != (az,):
        bz = np.broadcast_to(bz, az, subok=True)

    # flatten the population axes, if any, and keep multi-indices as labels
    if apop:
        dndz = np.reshape(dndz, (-1, az))
        npop = len(dndz)
    else:
        npop = None

    log.info('number of galaxy populations: %s', npop)

    # define the bias function, as requested
    # arguments are delta and the mean redshift of the interval
    # return value must be a new array which will be modified in place later
    if bz is None:

        # just make a copy of the input
        def bf(delta, zbar):
            return np.copy(delta)

    elif bias == 'linear':

        # a linear bias model: delta_g = b*delta
        def bf(delta, zbar):
            b = np.interp(zbar, z, bz)
            return b*delta

    elif bias == 'log-linear':

        # a log-linear bias model: log(1 + delta_g) = b*log(1 + delta)
        def bf(delta, zbar):
            b = np.interp(zbar, z, bz)
            delta_g = np.log1p(delta)
            delta_g *= b
            np.expm1(delta_g, out=delta_g)
            return delta_g

    elif bias == 'function':

        # custom bias function
        if not callable(bz):
            raise TypeError('a "function" bias requires a callable bz')
        bf = bz

    else:
        raise ValueError(f'invalid value for bias: {bias}')

    # keep track of total number of galaxies sampled
    nsam = 0

    # initial yield
    ngal = red = pop = lon = lat = None

    # wait for next redshift slice and return positions, or stop on exit
    while True:
        try:
            zmin, zmax, delta, vis = yield ngal, red, pop, lon, lat
        except GeneratorExit:
            break

        # get the restriction of dndz to the redshift interval
        dndz_, z_ = restrict_interval(dndz, z, zmin, zmax)

        # compute the number density of galaxies in redshift interval
        # the result is potentially an array over populations
        p = np.trapz(dndz_, z_, axis=-1)

        log.info('galaxies/arcmin2 in interval: %s', p)

        # get the total number of galaxies across all populations
        # we are assuming Poisson statistics, so we can sample from the sum
        ntot = np.sum(p, axis=-1)

        log.info('expected total galaxies in interval: %s', f'{ntot*ARCMIN2_SPHERE:,.2f}')

        # if there are no galaxies, we are done
        if ntot == 0:
            red = lon = lat = np.empty(0)
            if npop is not None:
                pop = np.empty(0)
            log.info('no galaxies, skipping...')
            continue

        # normalise the number densities to get propability densities
        dndz_ /= np.where(p > 0, p, 1)[..., np.newaxis]

        # normalise to get probability to find galaxy in each population
        p /= ntot

        # compute the mean redshift over all distributions
        zbar = np.dot(p, np.trapz(dndz_*z_, z_, axis=-1))

        log.info('galaxies mean redshift: %g', zbar)

        # compute cumulative distribution in place for redshift sampling
        cumtrapz(dndz_, z_, out=dndz_)

        # compute the distribution of the galaxies
        # first, compute the galaxy overdensity using the bias function
        # then, modifying the array in place, turn into number count
        dist = bf(delta, zbar)
        dist += 1
        dist *= ARCMIN2_SPHERE/np.shape(dist)[-1]*ntot

        log.info('expected total galaxies from density: %s', f'{np.sum(dist):,.2f}')

        # apply visibility if given
        if vis is not None:
            dist *= vis

        # expected number of visible galaxies
        nvis = np.sum(dist)

        log.info('expected visible galaxies from density: %s', f'{nvis:,.2f}')

        # sample number of galaxies
        ngal = rng.poisson(nvis)

        log.info('number of galaxies to be sampled: %s', f'{ngal:,d}')

        # turn into conditional probability distribution
        dist /= np.max(dist)

        log.info('sampling efficiency: %g', np.mean(dist))

        # for converting randomly sampled positions to HEALPix indices
        nside = hp.get_nside(dist)

        # these will hold the results
        red = np.empty(ngal)
        lon = np.empty(ngal)
        lat = np.empty(ngal)
        if npop is not None:
            pop = np.empty(ngal, dtype=int)

        # rejection sampling of galaxies
        # propose batches of 10000 galaxies over the full sky
        # then accept or reject based on the spatial distribution in dist
        # for accepted galaxies, pick a population and then a redshift
        nrem = ngal
        while nrem > 0:
            npro = min(nrem, 10000)
            lon_pro = rng.uniform(-180, 180, size=npro)
            lat_pro = np.rad2deg(np.arcsin(rng.uniform(-1, 1, size=npro)))
            pix_pro = hp.ang2pix(nside, lon_pro, lat_pro, lonlat=True)
            acc = (rng.uniform(0, 1, size=npro) < dist[pix_pro])
            nacc = acc.sum()
            sli = slice(ngal-nrem, ngal-nrem+nacc)
            if npop is not None:
                pop_ = rng.choice(npop, p=p, size=nacc)
                red_ = np.empty(nacc)
                for i in range(npop):
                    sel = (pop_ == i)
                    nsel = sel.sum()
                    red_[sel] = np.interp(rng.uniform(0, 1, size=nsel), dndz_[i], z_)
                pop[sli] = pop_
                red[sli] = red_
            else:
                red[sli] = np.interp(rng.uniform(0, 1, size=nacc), dndz_, z_)
            lon[sli] = lon_pro[acc]
            lat[sli] = lat_pro[acc]
            nrem -= nacc

        # mark some variables as disposable
        dndz_ = z_ = p = dist = lon_pro = lat_pro = pix_pro = acc = None

        # add to total sampled
        nsam += ngal

    log.info('total number of galaxies sampled: %s', f'{nsam:,d}')


def ellipticity_ryden04(mu, sigma, gamma, sigma_gamma, size=None, *, rng=None):
    r'''ellipticity distribution following Ryden (2004)

    The ellipticities are sampled by randomly projecting a 3D ellipsoid with
    principal axes :math:`A > B > C` [1]_.  The distribution of :math:`\log(1 -
    B/A)` is normal with mean :math:`\mu` and standard deviation :math:`\sigma`.
    The distribution of :math:`1 - C/B` is normal with mean :math:`\gamma` and
    standard deviation :math:`\sigma_\gamma` [2]_.  Both distributions are
    truncated to produce ratios in the range 0 to 1 using rejection sampling.

    Parameters
    ----------
    mu : array_like
        Mean of the truncated normal for :math:`\log(1 - B/A)`.
    sigma : array_like
        Standard deviation for :math:`\log(1 - B/A)`.
    gamma : array_like
        Mean of the truncated normal for :math:`1 - C/B`.
    sigma_gamma : array_like
        Standard deviation for :math:`1 - C/B`.
    size : int or tuple of ints or None
        Sample size.  If ``None``, the size is inferred from the parameters.
    rng : :class:`~numpy.random.Generator`, optional
        Random number generator.  If not given, a default RNG will be used.

    Returns
    -------
    eps : array_like
        Ellipticities from projected axis ratios.

    Notes
    -----
    If :math:`q = b/a` is the axis ratio of an elliptical isophote with
    semi-major axis :math:`a` and semi-minor axis :math:`b`, the ellipticity
    sampled by this function is :math:`\epsilon = (1 - q)/(1 + q)`.

    References
    ----------
    .. [1] Ryden B. S., 2004, ApJ, 601, 214.
    .. [2] Padilla N. D., Strauss M. A., 2008, MNRAS, 388, 1321.

    '''

    # default RNG if not provided
    if rng is None:
        rng = np.random.default_rng()

    # draw gamma and epsilon from truncated normal -- eq.s (10)-(11)
    # first sample unbounded normal, then rejection sample truncation
    eps = rng.normal(mu, sigma, size=size)
    bad = (eps > 0)
    while np.any(bad):
        eps[bad] = rng.normal(mu, sigma, size=eps[bad].shape)
        bad = (eps > 0)
    gam = rng.normal(gamma, sigma_gamma, size=size)
    bad = (gam < 0) | (gam > 1)
    while np.any(bad):
        gam[bad] = rng.normal(gamma, sigma_gamma, size=gam[bad].shape)
        bad = (gam < 0) | (gam > 1)

    # compute triaxial axis ratios zeta = B/A, xi = C/A
    zeta = -np.expm1(eps)
    xi = (1 - gam)*zeta

    # random projection of random triaxial ellipsoid
    q = triaxial_axis_ratio(zeta, xi, rng=rng)

    # return the ellipticity
    return (1-q)/(1+q)


@generator('ngal -> gal_ell')
def gal_ellip_ryden04(mu, sigma, gamma, sigma_gamma, *, rng=None):
    r'''generator for galaxy ellipticities following Ryden (2004)

    The ellipticities are sampled by randomly projecting a 3D ellipsoid with
    principal axes :math:`A > B > C` [1]_.  The distribution of :math:`\log(1 -
    B/A)` is normal with mean :math:`\mu` and standard deviation :math:`\sigma`.
    The distribution of :math:`1 - C/B` is normal with mean :math:`\gamma` and
    standard deviation :math:`\sigma_\gamma` [2]_.  Both distributions are
    truncated to produce ratios in the range 0 to 1 using rejection sampling.

    Parameters
    ----------
    mu : array_like
        Mean of the truncated normal for :math:`\log(1 - B/A)`.
    sigma : array_like
        Standard deviation for :math:`\log(1 - B/A)`.
    gamma : array_like
        Mean of the truncated normal for :math:`1 - C/B`.
    sigma_gamma : array_like
        Standard deviation for :math:`1 - C/B`.
    size : int or tuple of ints or None
        Sample size.  If ``None``, the size is inferred from the parameters.
    rng : :class:`~numpy.random.Generator`, optional
        Random number generator.  If not given, a default RNG will be used.

    Receives
    --------
    ngal : int
        Number of galaxies for which ellipticities are sampled.

    Yields
    ------
    gal_ell : (ngal,) array_like
        Array of complex-valued galaxy ellipticity.

    Notes
    -----
    If :math:`q = b/a` is the axis ratio of an elliptical isophote with
    semi-major axis :math:`a` and semi-minor axis :math:`b`, and :math:`\phi` is
    the orientation of the elliptical isophote, the output of this function is
    the complex-valued :math:`\epsilon`-ellipticity

    .. math::

        \epsilon = \frac{1 - q}{1 + q} \, \mathrm{e}^{\mathrm{i} \, 2\phi} \;.

    References
    ----------
    .. [1] Ryden B. S., 2004, ApJ, 601, 214
    .. [2] Padilla N. D., Strauss M. A., 2008, MNRAS, 388, 1321.

    '''

    # default RNG if not provided
    if rng is None:
        rng = np.random.default_rng()

    # initial yield
    e = None

    # wait for inputs and return ellipticities, or stop on exit
    while True:
        try:
            ngal = yield e
        except GeneratorExit:
            break

        # compute random complex phase
        # sample magnitude according to model
        e = np.exp(1j*rng.uniform(0, 2*np.pi, size=ngal))
        e *= ellipticity_ryden04(mu, sigma, gamma, sigma_gamma, size=ngal, rng=rng)

        log.info('per-component std. dev.: %.3f, %.3f', np.std(e.real), np.std(e.imag))
