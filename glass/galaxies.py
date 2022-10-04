# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''
================================
Galaxies (:mod:`glass.galaxies`)
================================

.. currentmodule:: glass.galaxies


Galaxy bias
===========

Variables
---------

.. autodata:: B
.. autodata:: BFN


Generators
----------

.. autosummary::
   :template: generator.rst
   :toctree: generated/

   gal_b_const
   gal_b_eff
   gal_bias_linear
   gal_bias_loglinear
   gal_bias_function


Galaxy distribution
===================

Variables
---------

.. autodata:: NGAL
.. autodata:: NZ
.. autodata:: GAL_LEN
.. autodata:: GAL_LON
.. autodata:: GAL_LAT


Generators
----------

.. autosummary::
   :template: generator.rst
   :toctree: generated/

   gal_density_const
   gal_density_dndz
   gal_positions_mat
   gal_positions_unif


Galaxy redshifts
================

Variables
---------

.. autodata:: GAL_Z
.. autodata:: GAL_POP


Generators
----------

.. autosummary::
   :template: generator.rst
   :toctree: generated/

   gal_redshifts_nz


Galaxy ellipticities
====================

Variables
---------

.. autodata:: GAL_ELL


Generators
----------

.. autosummary::
   :template: generator.rst
   :toctree: generated/

   gal_ellip_gaussian
   gal_ellip_intnorm
   gal_ellip_ryden04


Other
-----

.. autosummary::
   :toctree: generated/

   ellipticity_ryden04


Galaxy shears
=============

Variables
---------

.. autodata:: GAL_SHE


Generators
----------

.. autosummary::
   :template: generator.rst
   :toctree: generated/

   gal_shear_interp


Photometric redshifts
=====================

Variables
---------

.. autodata:: GAL_PHZ


Generators
----------

.. autosummary::
   :template: generator.rst
   :toctree: generated/

   gal_phz_gausserr

'''

import logging
import numpy as np
import healpy as hp

from .generator import generator, optional
from .util import (ARCMIN2_SPHERE, restrict_interval, trapz_product, cumtrapz,
                   triaxial_axis_ratio, hp_integrate)

from .sim import ZMIN, ZMAX
from .matter import DELTA, WZ
from .lensing import ZSRC, KAPPA, GAMMA
from .observations import VIS


log = logging.getLogger(__name__)

# variable definitions
B = 'galaxy bias parameter'
'''Parameter used in galaxy bias models.  In nonlinear galaxy bias models, this
should usually be the linearised galaxy bias near :math:`\\delta = 0`.'''
BFN = 'galaxy bias function'
'''Callable :math:`B_g` that implements a galaxy bias model :math:`\\delta_g =
B_g(\\delta)`.'''
NGAL = 'mean galaxy density'
'''Expected number of galaxies per arcmin2.'''
NZ = 'galaxy redshift distribution'
'''Redshift distribution function :math:`n(z)` of galaxies.  The function does
not have to be normalised.'''
GAL_LEN = 'galaxy count'
'''The total number of sampled galaxies, i.e. the length of the galaxy column
data.'''
GAL_LON = 'galaxy longitudes'
'''Column data with sampled galaxy longitudes.'''
GAL_LAT = 'galaxy latitudes'
'''Column data with sampled galaxy latitudes.'''
GAL_Z = 'galaxy redshifts'
'''Column data with sampled galaxy redshifts.'''
GAL_POP = 'galaxy populations'
'''Column data with sampled galaxy populations, or ``None`` if no galaxy
populations were given.'''
GAL_ELL = 'galaxy ellipticities'
'''Column data with sampled galaxy ellipticities.'''
GAL_SHE = 'galaxy shears'
'''Column data with sampled galaxy shears, i.e. the weakly-lensed galaxy
ellipticities.'''
GAL_PHZ = 'galaxy photometric redshifts'
'''Column data with sampled photometric galaxy redshifts.'''


@generator(yields=B)
def gal_b_const(b):
    '''constant bias parameter

    This generator yields the given bias parameter ``b`` for every iteration.

    Parameters
    ----------
    b : float
        Constant bias parameter.

    '''
    while True:
        yield b


@generator(receives=WZ, yields=B)
def gal_b_eff(z, bz):
    '''effective bias parameter from a redshift-dependent bias function

    This generator takes a redshift-dependent galaxy bias function :math:`b(z)`
    and computes an effective galaxy bias parameter :math:`\\bar{b}_i` for
    iteration :math:`i` using the matter weight function.

    Parameters
    ----------
    z, bz : array_like
        Redshifts and values of bias function :math:`b(z)`.

    Notes
    -----
    The effective bias parameter :math:`\\bar{b}_i`` in shell :math:`i` is
    computed using the matter weight function :math:`W_i` as the weighted
    average

    .. math::

        \\bar{b} = \\frac{\\int_{z_{i-1}}^{z_i} b(z) \\, W(z) \\, dz}
                         {\\int_{z_{i-1}}^{z_i} W(z) \\, dz}  \\;.

    '''
    b = None
    while True:
        z_, wz = yield b
        b = trapz_product((z, bz), (z_, wz))/np.trapz(wz, z_)


@generator(receives=B, yields=BFN)
def gal_bias_linear():
    '''linear galaxy bias model :math:`\\delta_g = b \\, \\delta`'''
    b = 1
    while True:
        b = yield (lambda delta: b*delta)


@generator(receives=B, yields=BFN)
def gal_bias_loglinear():
    '''log-linear galaxy bias model :math:`\\ln(1 + \\delta_g) = b \\ln(1 + \\delta)`'''

    def f(delta, b):
        delta_g = np.log1p(delta)
        delta_g *= b
        np.expm1(delta_g, out=delta_g)
        return delta_g

    b = 1
    while True:
        b = yield (lambda delta: f(delta, b))


def gal_bias_function(bias_function, args=()):
    '''generic galaxy bias model :math:`\\delta_g = B_g(\\delta)`'''

    if not callable(bias_function):
        raise TypeError('bias function is not callable')

    @generator(receives=args, yields=BFN)
    def g():
        args = ()
        while True:
            args = yield (lambda delta: bias_function(delta, *args))

    g.__name__ = gal_bias_function.__name__

    return g()


@generator(receives=(ZMIN, ZMAX), yields=NGAL)
def gal_density_const(dndz):
    '''constant galaxy density

    Parameters
    ----------
    dndz : float
        Constant galaxy density in units of 1/arcmin2/dz.

    Yields
    ------
    :data:`NGAL`, float
        Expected galaxy density, equal to ``dndz*(ZMAX-ZMIN)`` for each
        iteration.

    Receives
    --------
    :data:`~glass.sim.ZMIN`, float
        Lower bound of redshift interval.
    :data:`~glass.sim.ZMAX`, float
        Upper bound of redshift interval.

    '''

    log.info('constant galaxy density: %g/arcmin2/dz', dndz)

    ngal = None
    while True:
        zmin, zmax = yield ngal
        ngal = dndz*(zmax - zmin)


@generator(
    receives=(ZMIN, ZMAX),
    yields=(NGAL, NZ))
def gal_density_dndz(z, dndz, *, ngal=None):
    '''galaxy density from a redshift distribution

    Distributions for multiple populations of galaxies (e.g. different
    photometric redshift bins) can be passed as a leading axis of the ``dndz``
    array.

    If ``ngal`` is given, the distribution is normalised such that the total
    number of galaxies over the entire redshift range is ``ngal``.

    Yields
    ------
    :data:`NGAL`, float
        Expected galaxy density, computed from the given distribution.
    :data:`NZ`, tuple of (K,), (..., K) array_like
        The restriction of ``z``, ``dndz`` to the current interval.  Leading
        array axes of ``dndz`` are treated as different galaxy populations.

    Receives
    --------
    :data:`~glass.sim.ZMIN`, float
        Lower bound for redshift distribution.
    :data:`~glass.sim.ZMAX`, float
        Upper bound for redshift distribution.

    '''

    # make sure valid number count distributions are passed
    if np.ndim(z) != 1:
        raise TypeError('redshifts must be 1d array')
    if not np.all(np.diff(z) > 0):
        raise ValueError('redshifts are not strictly increasing')
    if np.shape(z) != np.shape(dndz)[-1:]:
        raise TypeError('redshift and distribution axes mismatch')
    if not np.all(np.greater_equal(dndz, 0)):
        raise ValueError('negative number counts in distribution')

    # normalise distribution if ngal is given
    if ngal is not None:
        dndz = dndz/np.trapz(dndz, z)[..., np.newaxis]*ngal

    # print number of populations from leading axes of distribution
    npop = np.prod(np.shape(dndz)[:-1], dtype=int)

    log.info('number of galaxy populations: %d', npop)

    # initial yield
    result = None

    while True:
        zmin, zmax = yield result

        # get the restriction of dndz to the redshift interval
        dndz_, z_ = restrict_interval(dndz, z, zmin, zmax)

        # compute the number density of galaxies in redshift interval
        # the result is potentially an array over populations
        npop = np.trapz(dndz_, z_, axis=-1)

        log.info('galaxies/arcmin2 in interval: %s', npop)

        # sum over population axes
        ngal = np.sum(npop, axis=None)

        # the result to yield
        result = ngal, (z_, dndz_)

        # clean up
        del z_, dndz_, npop


@generator(
    receives=(NGAL, DELTA, optional(BFN), optional(VIS)),
    yields=(GAL_LEN, GAL_LON, GAL_LAT))
def gal_positions_mat(*, remove_monopole=False, rng=None):
    '''galaxy positions from matter distribution and a bias model

    The map of expected galaxy number counts is constructed from the galaxy
    number density, matter density contrast, an optional bias function, and an
    optional visibility map.

    If ``remove_monopole`` is set, the monopole of the computed galaxy density
    contrast is removed.  Over the full sky, the mean number density of the map
    will then match the given number density exactly.  This, however, means that
    an effectively different bias model is being used, unless the monopole is
    already zero in the first place.

    The galaxies are sampled by rejection sampling over the full sky.  This is
    potentially inefficient if the visible sky is small.

    Parameters
    ----------
    remove_monopole : bool, optional
        If set, the monopole of the galaxy density contrast is fixed to zero.

    Yields
    ------
    :data:`GAL_LEN`, int
        Number of sampled galaxies.
    :data:`GAL_LON`, (GAL_LEN,) array_like
        Column of longitudes for sampled galaxies.
    :data:`GAL_LAT`, (GAL_LEN,) array_like
        Column of latitudes for sampled galaxies.

    Receives
    --------
    :data:`NGAL`
        The mean galaxy density.  The output may have a different mean if the
        monopole of the resulting galaxy density contrast is not zero.
    :data:`~glass.matter.DELTA`
        The matter density contrast.  This is fed into the galaxy bias model to
        produce the galaxy density contrast.
    :data:`BFN`, optional
        The galaxy bias function.  If not given, the galaxy density contrast is
        equal to the matter density contrast.
    :data:`~glass.observations.VIS`, optional
        Visibility map for the observed galaxies.  This is multiplied with the
        full sky galaxy number map, and both must have the same NSIDE parameter.

    '''

    # get default RNG if not given
    if rng is None:
        rng = np.random.default_rng()

    # keep track of the total number of galaxies sampled
    ntot = 0

    # initial yield
    result = None

    # return positions and wait for next redshift slice
    while True:
        try:
            ngal, delta, b, v = yield result
        except GeneratorExit:
            break

        # compute galaxy density contrast from bias model
        if b is None:
            n = np.copy(delta)
        else:
            n = b(delta)

        # remove monopole if asked to
        if remove_monopole:
            n -= np.mean(n, keepdims=True)

        # turn into number count, modifying the array in place
        n += 1
        n *= ARCMIN2_SPHERE/n.size*ngal

        # average galaxy density contrast over HEALPix pixels
        n = hp_integrate(n)

        # apply visibility if given
        if v is not None:
            n *= v

        # expected number of visible galaxies
        nsum = np.sum(n)

        log.info('expected visible galaxies: %s', f'{nsum:,.2f}')

        # sample number of galaxies
        nsam = rng.poisson(nsum)

        log.info('sampled number of galaxies: %s', f'{nsam:,d}')

        # scaling for probability distribution
        nmax = np.max(n)

        log.info('sampling efficiency: %g', nsum/n.size/nmax)

        # for converting randomly sampled positions to HEALPix indices
        nside = hp.get_nside(n)

        # these will hold the results
        lon = np.empty(nsam)
        lat = np.empty(nsam)

        # rejection sampling of galaxies
        # propose batches of 10000 galaxies over the full sky
        # then accept or reject based on the spatial distribution in n
        nrem = nsam
        while nrem > 0:
            npro = min(nrem, 10000)
            lon_pro = rng.uniform(-180, 180, size=npro)
            lat_pro = np.rad2deg(np.arcsin(rng.uniform(-1, 1, size=npro)))
            pix_pro = hp.ang2pix(nside, lon_pro, lat_pro, lonlat=True)
            acc = (rng.uniform(0, nmax, size=npro) < n[pix_pro])
            nacc = acc.sum()
            sli = slice(nsam-nrem, nsam-nrem+nacc)
            lon[sli] = lon_pro[acc]
            lat[sli] = lat_pro[acc]
            nrem -= nacc
            del lon_pro, lat_pro, pix_pro, acc

        # results of the sampling
        result = nsam, lon, lat

        # add to total sampled
        ntot += nsam

        # clean up potentially large arrays; outputs are still kept in result
        del delta, v, n, lon, lat

    log.info('total number of galaxies sampled: %s', f'{ntot:,d}')


@generator(
    receives=NGAL,
    yields=(GAL_LEN, GAL_LON, GAL_LAT))
def gal_positions_unif(*, rng=None):
    '''galaxy positions uniformly over the sphere

    Yields
    ------
    :data:`GAL_LEN`, int
        Number of sampled galaxies.
    :data:`GAL_LON`, (GAL_LEN,) array_like
        Column of longitudes for sampled galaxies.
    :data:`GAL_LAT`, (GAL_LEN,) array_like
        Column of latitudes for sampled galaxies.

    Receives
    --------
    :data:`NGAL`, array_like
        Expected galaxy density.

    '''

    # get default RNG if not given
    if rng is None:
        rng = np.random.default_rng()

    # keep track of the total number of galaxies sampled
    ntot = 0

    # initial yield
    result = None

    # sample positions uniformly
    while True:
        try:
            ngal = yield result
        except GeneratorExit:
            break

        # expected number of visible galaxies
        nlam = ARCMIN2_SPHERE*ngal

        log.info('expected number of galaxies: %s', f'{nlam:,.2f}')

        # sample number of galaxies
        nsam = rng.poisson(nlam)

        log.info('sampled number of galaxies: %s', f'{nsam:,d}')

        # sample uniformly over the sphere
        lon = rng.uniform(-180, 180, size=nsam)
        lat = np.rad2deg(np.arcsin(rng.uniform(-1, 1, size=nsam)))

        # results of the sampling
        result = nsam, lon, lat

        # add to total sampled
        ntot += nsam

        # clean up potentially large arrays; outputs are still kept in result
        del lon, lat

    log.info('total number of galaxies sampled: %s', f'{ntot:,d}')


@generator(
    receives=(NZ, GAL_LEN),
    yields=(GAL_Z, GAL_POP))
def gal_redshifts_nz(*, rng=None):
    '''galaxy redshifts from a distribution

    Yields
    ------
    :data:`GAL_Z`, (GAL_LEN,) array_like
        Redshifts sampled from :data:`NZ`.
    :data:`GAL_POP`, (GAL_LEN,) tuple of int array_like or None
        Index of the galaxy population from the leading axes of :data:`NZ`; or
        ``None`` if there are no galaxy populations.

    Receives
    --------
    :data:`NZ`, tuple of (N,), (..., N) array_like
        Redshifts and densities of the galaxy redshift distribution.  Leading
        axes in the density are treated as different galaxy populations.
    :data:`GAL_LEN` int
        Number of galaxies to be sampled, i.e. length of galaxy data column.

    '''

    # get default RNG if not given
    if rng is None:
        rng = np.random.default_rng()

    # initial yield
    result = None

    # wait for next redshift slice and return redshifts, or stop on exit
    while True:
        try:
            (z, nz), n = yield result
        except GeneratorExit:
            break

        # get galaxy populations
        if np.ndim(nz) > 1:
            pop = list(np.ndindex(np.shape(nz)[:-1]))
        else:
            pop = None

        log.info('number of galaxy populations: %d', len(pop) if pop else 1)

        # compute the as-yet unnormalised CDF of each galaxy population
        cdf = cumtrapz(nz, z)

        # compute probability to be in each galaxy population
        p = cdf[..., -1]/cdf[..., -1].sum(axis=None, keepdims=True)

        # now normalise the CDFs
        cdf /= cdf[..., -1:]

        log.info('relative size of populations: %s', p)

        # sample redshifts and populations
        if pop is not None:
            x = rng.choice(len(pop), p=p, size=n)
            gal_z = np.empty(n)
            for i, j in enumerate(pop):
                s = (x == i)
                gal_z[s] = np.interp(rng.uniform(0, 1, size=s.sum()), cdf[j], z)
            gal_pop = np.take(pop, x)
            del x, s
        else:
            gal_z = np.interp(rng.uniform(0, 1, size=n), cdf, z)
            gal_pop = None

        # next yield
        result = gal_z, gal_pop


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
        Array of :term:`ellipticity (modulus)` from projected axis ratios.

    See also
    --------
    gal_ellip_ryden04:
        Generator for galaxy ellipticities using this distribution.

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


@generator(receives=GAL_LEN, yields=GAL_ELL)
def gal_ellip_gaussian(sigma, *, rng=None):
    r'''generator for Gaussian galaxy ellipticities

    The ellipticities are sampled from a normal distribution with standard
    deviation ``sigma`` for each component.  Samples where the ellipticity is
    larger than unity are discarded.  Hence, do not use this function with too
    large values of ``sigma``, or the sampling will become inefficient.

    Parameters
    ----------
    sigma : array_like
        Standard deviation in each component.
    rng : :class:`~numpy.random.Generator`, optional
        Random number generator.  If not given, a default RNG will be used.

    Yields
    ------
    :data:`GAL_ELL`, (GAL_LEN,) array_like
        Array of galaxy :term:`ellipticity (complex)`.

    Receives
    --------
    :data:`GAL_LEN`, int
        Number of galaxies for which ellipticities are sampled.

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

        # sample complex ellipticities
        # reject those where abs(e) > 0
        e = rng.standard_normal(2*ngal, np.float64).view(np.complex128)
        e *= sigma
        i = np.where(np.abs(e) > 1)[0]
        while len(i) > 0:
            rng.standard_normal(2*len(i), np.float64, e[i].view(np.float64))
            e[i] *= sigma
            i = i[np.abs(e[i]) > 1]


@generator(receives=GAL_LEN, yields=GAL_ELL)
def gal_ellip_intnorm(sigma_eta, *, rng=None):
    r'''generator for galaxy ellipticities with intrinsic normal distribution

    The ellipticities are sampled from an intrinsic normal distribution with
    standard deviation ``sigma_eta`` for each component.

    Parameters
    ----------
    sigma_eta : array_like
        Standard deviation in each component of the normal coordinates.
    rng : :class:`~numpy.random.Generator`, optional
        Random number generator.  If not given, a default RNG will be used.

    Yields
    ------
    :data:`GAL_ELL`, (GAL_LEN,) array_like
        Array of galaxy :term:`ellipticity (complex)`.

    Receives
    --------
    :data:`GAL_LEN`, int
        Number of galaxies for which ellipticities are sampled.

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

        # sample complex ellipticities
        e = rng.standard_normal(2*ngal, np.float64).view(np.complex128)
        e *= sigma_eta
        r = np.hypot(e.real, e.imag)
        e *= np.tanh(r/2)/r


@generator(receives=GAL_LEN, yields=GAL_ELL)
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

    Yields
    ------
    :data:`GAL_ELL`, (GAL_LEN,) array_like
        Array of galaxy :term:`ellipticity (complex)`.

    Receives
    --------
    :data:`GAL_LEN`, int
        Number of galaxies for which ellipticities are sampled.

    See also
    --------
    ellipticity_ryden04:
        Sample the ellipticity modulus distribution.

    References
    ----------
    .. [1] Ryden B. S., 2004, ApJ, 601, 214
    .. [2] Padilla N. D., Strauss M. A., 2008, MNRAS, 388, 1321.

    '''

    # get default RNG if not given
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

        if len(e) > 0:
            log.info('per-component std. dev.: %.3f, %.3f', np.std(e.real), np.std(e.imag))
        else:
            log.info('no galaxies')


@generator(
    receives=(ZSRC, KAPPA, GAMMA, GAL_Z, GAL_LON, GAL_LAT, GAL_ELL),
    yields=GAL_SHE)
def gal_shear_interp(cosmo):
    '''generator for observed galaxy shears from interpolated lensing

    Takes lensing maps for convergence and shear and produces a lensed
    ellipticity (shear) for each intrinsic galaxy ellipticity.

    The lensing maps are interpolated over the redshift interval using each
    individual galaxy redshift [1]_.

    Parameters
    ----------
    cosmo : cosmology
        Cosmology instance for distance functions.

    Receives
    --------
    zsrc : float
        Source plane redshift of lensing maps.
    kappa, gamma1, gamma2 : array_like
        HEALPix maps for convergence and shear.
    gal_z : (ngal,) array_like
        Array of galaxy redshifts.
    gal_lon, gal_lat : (ngal,) array_like
        Arrays for galaxy positions.
    gal_ell : (ngal,) array_like
        Array of complex-valued intrinsic galaxy ellipticities.

    Yields
    ------
    gal_she : (ngal,) array_like
        Array of complex-valued observed galaxy shears (lensed ellipticities).

    References
    ----------
    .. [1] Tessore et al., in prep

    '''

    # initial values
    zsrc = 0
    kap = gam1 = gam2 = np.zeros(12)

    # initial yield
    g = None

    # yield shears and get a new shell, or stop on exit
    # always keep previous lensing plane
    while True:
        zsrc_, kap_, gam1_, gam2_ = zsrc, kap, gam1, gam2
        try:
            zsrc, kap, (gam1, gam2), z, lon, lat, e = yield g
        except GeneratorExit:
            break

        # interpolation weight for galaxy redshift given source planes
        t = (cosmo.xm(zsrc_, z)/cosmo.xm(z))/(cosmo.xm(zsrc_, zsrc)/cosmo.xm(zsrc))

        # get the lensing maps at galaxy position
        # interpolate in redshifts
        ngal = len(z)
        k = np.empty(ngal)
        g = np.empty(ngal, dtype=complex)
        nside, nside_ = hp.get_nside(kap), hp.get_nside(kap_)
        for i in range(0, ngal, 10000):
            s = slice(i, i+10000)
            ipix = hp.ang2pix(nside, lon[s], lat[s], lonlat=True)
            ipix_ = hp.ang2pix(nside_, lon[s], lat[s], lonlat=True)
            for v, m, m_ in (k, kap, kap_), (g.real, gam1, gam1_), (g.imag, gam2, gam2_):
                v[s] = m_[ipix_]
                v[s] *= 1 - t[s]
                v[s] += t[s]*m[ipix]

        # compute reduced shear in place and forget about convergence
        k -= 1
        np.negative(k, out=k)
        g /= k
        k = None

        # compute lensed ellipticities
        g = (e + g)/(1 + g.conj()*e)


@generator(receives=GAL_Z, yields=GAL_PHZ)
def gal_phz_gausserr(sigma_0, rng=None):
    r'''photometric redshift with Gaussian error

    A simple toy model of photometric redshift errors that assumes a Gaussian
    error with redshift-dependent standard deviation :math:`\sigma(z) = (1 + z)
    \sigma_0` [1]_.

    Parameters
    ----------
    sigma_0 : float
        Redshift error in the tomographic binning at zero redshift.
    rng : :class:`~numpy.random.Generator`, optional
        Random number generator.  If not given, a default RNG will be used.

    Yields
    ------
    gal_z_phot : (ngal,) array_like
        Photometric galaxy redshifts assuming Gaussian errors.

    Receives
    --------
    gal_z : (ngal,) array_like
        Galaxy redshifts.

    See Also
    --------
    glass.observations.tomo_nz_gausserr :
        create tomographic redshift distributions assuming the same model

    References
    ----------
    .. [1] Amara A., Réfrégier A., 2007, MNRAS, 381, 1018.
           doi:10.1111/j.1365-2966.2007.12271.x

    Examples
    --------
    See the :doc:`examples:basic/photoz` example.

    '''

    # get default RNG if not given
    if rng is None:
        rng = np.random.default_rng()

    # initial yield
    zphot = None

    # sample photometric redshift for all true galaxy redshifts
    # normal random variates are truncated to positive values
    while True:
        try:
            z = yield zphot
        except GeneratorExit:
            break

        size = np.shape(z)
        z = np.reshape(z, (-1,))

        zphot = rng.normal(z, (1 + z)*sigma_0)

        trunc = np.where(zphot < 0)[0]
        while trunc.size:
            zphot[trunc] = rng.normal(z[trunc], (1 + z[trunc])*sigma_0)
            trunc = trunc[zphot[trunc] < 0]

        zphot = zphot.reshape(size)
