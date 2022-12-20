# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''module for galaxies'''

import logging
import numpy as np
import healpix

from .generator import generator, optional
from .util import (ARCMIN2_SPHERE, restrict_interval, trapz_product, cumtrapz,
                   triaxial_axis_ratio, hp_integrate, format_array)

from .matter import DELTA
from .lensing import ZSRC, KAPPA, GAMMA
from .observations import VIS


log = logging.getLogger(__name__)

# variable definitions
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


def effective_bias(z, b, shells, weights):
    '''Effective bias parameter from a redshift-dependent bias function.

    This function takes a redshift-dependent galaxy bias function :math:`b(z)`
    and computes an effective galaxy bias parameter :math:`\\bar{b}_i` for
    each shell :math:`i` using the matter weight function.

    Parameters
    ----------
    z, b : array_like
        Redshifts and values of bias function :math:`b(z)`.
    shells : array_like
        Redshifts of the shell boundaries.
    weights : :class:`glass.matter.MatterWeightFunction`
        The matter weight function for the shells.

    Returns
    -------
    beff : array_like
        Effective bias parameter for each shell.

    Notes
    -----
    The effective bias parameter :math:`\\bar{b}_i` in shell :math:`i` is
    computed using the matter weight function :math:`W` as the weighted
    average

    .. math::

        \\bar{b}_i = \\frac{\\int_{z_{i-1}}^{z_i} b(z) \\, W(z) \\, dz}
                           {\\int_{z_{i-1}}^{z_i} W(z) \\, dz}  \\;.

    '''
    beff = np.empty(len(shells)-1)
    for i, (zmin, zmax) in enumerate(zip(shells, shells[1:])):
        w_, z_ = restrict_interval(weights.w, weights.z, zmin, zmax)
        beff[i] = trapz_product((z, b), (z_, w_))/np.trapz(w_, z_)
    return beff


def linear_bias(delta, b):
    '''linear galaxy bias model :math:`\\delta_g = b \\, \\delta`'''
    return b*delta


def loglinear_bias(delta, b):
    '''log-linear galaxy bias model :math:`\\ln(1 + \\delta_g) = b \\ln(1 + \\delta)`'''
    delta_g = np.log1p(delta)
    delta_g *= b
    np.expm1(delta_g, out=delta_g)
    return delta_g


def constant_densities(dndz, shells):
    '''Constant galaxy density in each shell.

    Parameters
    ----------
    dndz : float
        Galaxy density per redshift interval.
    shells : array_like
        Redshift boundaries of the shells.

    Returns
    -------
    ngal : array_like
        Expected galaxy density in each shell, equal to ``dndz*(z[i]-z[i-1])``
        for shell ``i``.

    '''
    return dndz*np.diff(shells)


def densities_from_dndz(z, dndz, shells, ntot=None):
    '''Galaxy densities from a redshift distribution.

    Computes the galaxy density in each shell from a given distribution.

    Distributions for multiple populations of galaxies (e.g. different
    photometric redshift bins) can be passed as a leading axis of the ``dndz``
    array.

    If ``ngal`` is given, the distribution is normalised such that the total
    number of galaxies over the entire redshift range is ``ngal``.

    Parameters
    -----------
    z, dndz : array_like
        Redshift distribution.
    shells : array_like
        Redshifts of the shell boundaries.
    ntot : float, optional
        Fix the total number of galaxies over the entire redshift range.

    Returns
    -------
    ngal : array_like
        Expected galaxy density in each shell.

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
    if ntot is not None:
        dndz = dndz/np.trapz(dndz, z)[..., np.newaxis]*ntot

    ngal = np.empty(len(shells)-1)

    for i, (zmin, zmax) in enumerate(zip(shells, shells[1:])):
        # get the restriction of dndz to the redshift interval
        dndz_, z_ = restrict_interval(dndz, z, zmin, zmax)

        # compute the number density of galaxies in redshift interval
        # the result is potentially an array over populations
        npop = np.trapz(dndz_, z_, axis=-1)

        # sum over population axes
        ngal[i] = np.sum(npop, axis=None)

    return ngal


@generator(
    receives=(DELTA, optional(VIS)),
    yields=(GAL_LEN, GAL_LON, GAL_LAT))
def gen_positions_from_matter(densities, biases=None, *, bias_model='linear',
                              remove_monopole=False, rng=None):
    '''Generate galaxy positions tracing the matter density.

    The map of expected galaxy number counts is constructed from the galaxy
    number density, matter density contrast, an optional bias model, and an
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
    densities : array_like
        Galaxy density per arcmin2 for each shell.
    biases : float or array_like, optional
        If given, bias parameters for each shell, either constant or one value
        per shell.
    bias_model : str or callable, optional
        The bias model to apply.  If a string, refers to a function in the
        galaxies module, e.g. ``'linear'`` for ``glass.matter.linear_bias``
        or ``'loglinear'`` for ``glass.matter.loglinear_bias``.
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
    :data:`~glass.matter.DELTA`
        The matter density contrast.  This is fed into the galaxy bias model to
        produce the galaxy density contrast.
    :data:`~glass.observations.VIS`, optional
        Visibility map for the observed galaxies.  This is multiplied with the
        full sky galaxy number map, and both must have the same NSIDE parameter.

    '''

    # get default RNG if not given
    if rng is None:
        rng = np.random.default_rng()

    # get biases compatible with densities
    if biases is None:
        biases = np.full_like(densities, np.nan)
    else:
        biases = np.broadcast_to(biases, len(densities))

    # get the bias model
    if isinstance(bias_model, str):
        bias_model = globals()[f'{bias_model}_bias']
    elif not callable(bias_model):
        raise ValueError('bias_model must be string or callable')

    # keep track of the overall number of galaxies sampled
    nall = 0

    # wrap in try block for finalisation code
    try:

        # wait for initial yield
        delta, v = yield

        # sample from densities and biases for each shell
        for ngal, b in zip(densities, biases):

            # compute galaxy density contrast from bias model, or copy
            if np.isnan(b):
                n = np.copy(delta)
            else:
                n = bias_model(delta, b)

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

            # clip number density at zero
            np.clip(n, 0, None, out=n)

            log.info('expected number of galaxies: %s', f'{n.sum():,.2f}')

            # sample actual galaxy number in each pixel
            n = rng.poisson(n)

            # total number of sampled galaxies
            ntot = n.sum()

            log.info('realised number of galaxies: %s', f'{ntot:,d}')

            # for converting randomly sampled positions to HEALPix indices
            npix = n.shape[-1]
            nside = healpix.npix2nside(npix)

            # these will hold the results
            lon = np.empty(ntot)
            lat = np.empty(ntot)

            # sample batches of 10000 pixels
            batch = 10_000
            ncur = 0
            for i in range(0, npix, batch):
                k = n[i:i+batch]
                bpix = np.repeat(np.arange(i, i+k.size), k)
                blon, blat = healpix.randang(nside, bpix, lonlat=True, rng=rng)
                lon[ncur:ncur+blon.size] = blon
                lat[ncur:ncur+blat.size] = blat
                ncur += bpix.size
                del k, bpix, blon, blat

            assert ncur == ntot, 'internal error in sampling'

            # add to overall sampled
            nall += ntot

            # clean up potentially large arrays that are no longer needed
            del delta, v, n

            # yield positions and return values for next shell
            delta, v = yield ntot, lon, lat

        else:
            # this generator finished its loop, stop the iteration
            yield GeneratorExit

    except GeneratorExit:
        log.info('total number of galaxies sampled: %s', f'{nall:,d}')


@generator(yields=(GAL_LEN, GAL_LON, GAL_LAT))
def gen_uniform_positions(densities, *, rng=None):
    '''Generate galaxy positions uniformly over the sphere.

    Parameters
    ----------
    densities : array_like
        Galaxy density per arcmin2 for each shell.

    Yields
    ------
    :data:`GAL_LEN`, int
        Number of sampled galaxies.
    :data:`GAL_LON`, (GAL_LEN,) array_like
        Column of longitudes for sampled galaxies.
    :data:`GAL_LAT`, (GAL_LEN,) array_like
        Column of latitudes for sampled galaxies.

    '''

    # get default RNG if not given
    if rng is None:
        rng = np.random.default_rng()

    # keep track of the overall number of galaxies sampled
    nall = 0

    # wrap in try block for finalisation code
    try:
        # initial yield, generator does not receive any data
        yield

        # sample positions uniformly
        for ngal in densities:

            # expected number of visible galaxies
            nlam = ARCMIN2_SPHERE*ngal

            log.info('expected number of galaxies: %s', f'{nlam:,.2f}')

            # sample number of galaxies
            ntot = rng.poisson(nlam)

            log.info('sampled number of galaxies: %s', f'{ntot:,d}')

            # sample uniformly over the sphere
            lon = rng.uniform(-180, 180, size=ntot)
            lat = np.rad2deg(np.arcsin(rng.uniform(-1, 1, size=ntot)))

            # add to overall sampled
            nall += ntot

            # yield positions and wait for next shell
            yield ntot, lon, lat

        else:
            # this generator finished its loop, stop the iteration
            yield GeneratorExit

    except GeneratorExit:
        log.info('total number of galaxies sampled: %s', f'{nall:,d}')


@generator(
    receives=GAL_LEN,
    yields=(GAL_Z, GAL_POP))
def gen_redshifts_from_nz(z, nz, shells, *, rng=None):
    '''Generate galaxy redshifts from a source distribution.

    Parameters
    ----------
    z, nz : array_like
        Source distribution.  Leading axes are treated as different galaxy
        populations.
    shells : array_like
        Redshifts of the matter shell boundaries.

    Yields
    ------
    :data:`GAL_Z`, (GAL_LEN,) array_like
        Redshifts sampled from :data:`NZ`.
    :data:`GAL_POP`, (GAL_LEN,) tuple of int array_like or None
        Index of the galaxy population from the leading axes of :data:`NZ`; or
        ``None`` if there are no galaxy populations.

    Receives
    --------
    :data:`GAL_LEN` int
        Number of galaxies to be sampled, i.e. length of galaxy data column.

    '''

    # get default RNG if not given
    if rng is None:
        rng = np.random.default_rng()

    # get galaxy populations
    if np.ndim(nz) > 1:
        pop = list(np.ndindex(np.shape(nz)[:-1]))
        log.info('number of galaxy populations: %d', len(pop))
    else:
        pop = None
        log.info('no galaxy populations')

    # keep track of total redshifts sampled
    nall = np.zeros(np.shape(nz)[:-1], dtype=int)

    try:
        # initial yield
        n = yield

        # yield sample for each shell
        for zmin, zmax in zip(shells, shells[1:]):

            # restrict n(z) to [zmin, zmax]
            nz_, z_ = restrict_interval(nz, z, zmin, zmax)

            # compute the as-yet unnormalised CDF of each galaxy population
            cdf = cumtrapz(nz_, z_)

            # compute probability to be in each galaxy population
            p = cdf[..., -1]/cdf[..., -1].sum(axis=None, keepdims=True)

            # now normalise the CDFs
            cdf /= cdf[..., -1:]

            # sample redshifts and populations
            if pop is not None:
                log.info('relative size of populations: %s', p)
                x = rng.choice(len(pop), p=p, size=n)
                gal_z = rng.uniform(0, 1, size=n)
                for i, j in enumerate(pop):
                    s = (x == i)
                    gal_z[s] = np.interp(gal_z[s], cdf[j], z_)
                    nall[j] += s.sum()
                gal_pop = np.take(pop, x)
                del x, s
            else:
                gal_z = np.interp(rng.uniform(0, 1, size=n), cdf, z_)
                gal_pop = None
                nall += len(gal_z)

            # yield and return next value for n
            n = yield gal_z, gal_pop

        else:
            # this generator finished its loop, stop the iteration
            yield GeneratorExit

    # loop has exited
    except GeneratorExit:
        log.info('total number of redshifts sampled: %s',
                 format_array('{:,}', nall))


@generator(receives=GAL_LEN, yields=GAL_Z)
def gen_uniform_redshifts(shells, *, rng=None):
    '''Generate galaxy redshifts from a uniform distribution.

    Parameters
    ----------
    shells : array_like
        Redshifts of the matter shell boundaries.

    Yields
    ------
    :data:`GAL_Z`, (GAL_LEN,) array_like
        Uniformly sampled redshifts.

    Receives
    --------
    :data:`GAL_LEN` int
        Number of galaxies to be sampled, i.e. length of galaxy data column.

    '''

    # get default RNG if not given
    if rng is None:
        rng = np.random.default_rng()

    # keep track of total redshifts sampled
    nall = 0

    try:
        # initial yield
        n = yield

        # yield sample for each shell
        for zmin, zmax in zip(shells, shells[1:]):

            # sample redshifts
            gal_z = rng.uniform(zmin, zmax, size=n)
            nall += len(gal_z)

            # yield and return next value for n
            n = yield gal_z

        else:
            # this generator finished its loop, stop the iteration
            yield GeneratorExit

    # loop has exited
    except GeneratorExit:
        log.info('total number of redshifts sampled: %s',
                 format_array('{:,}', nall))


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
def gen_ellip_zero(*, rng=None):
    r'''generator for zero galaxy ellipticities

    This generator yields zero ellipticity for all galaxies, so that the
    resulting galaxy shear after lensing has no shape noise.

    Parameters
    ----------
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

    n = yield
    while True:
        ell = np.zeros(n, dtype=complex)
        n = yield ell


@generator(receives=GAL_LEN, yields=GAL_ELL)
def gen_ellip_gaussian(sigma, *, rng=None):
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
def gen_ellip_intnorm(sigma, *, rng=None):
    r'''generator for galaxy ellipticities with intrinsic normal distribution

    The ellipticities are sampled from an intrinsic normal distribution with
    standard deviation ``sigma_eta`` for each component.

    Parameters
    ----------
    sigma : array_like
        Standard deviation of the ellipticity in each component.
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

    # make sure sigma is admissible
    if not 0 <= sigma < 0.5**0.5:
        raise ValueError('sigma must be between 0 and sqrt(0.5)')

    # default RNG if not provided
    if rng is None:
        rng = np.random.default_rng()

    # convert to sigma_eta using fit
    sigma_eta = sigma*((8 + 5*sigma**2)/(2 - 4*sigma**2))**0.5

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
        e *= np.divide(np.tanh(r/2), r, where=(r > 0), out=r)

        if ngal > 0:
            mu = np.mean(e)
            log.info('ellipticity mean: %+.3f%+.3fi', mu.real, mu.imag)
            log.info('ellipticity sigma: %.3f', (np.var(e)/2)**0.5)
        else:
            log.info('(number of galaxies is zero, no ellipticities sampled)')


@generator(receives=GAL_LEN, yields=GAL_ELL)
def gen_ellip_ryden04(mu, sigma, gamma, sigma_gamma, *, rng=None):
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
    receives=(KAPPA, GAMMA, GAL_LON, GAL_LAT, GAL_ELL),
    yields=GAL_SHE)
def gen_shear_simple(cosmo):
    '''generator for observed galaxy shears from non-interpolated lensing

    Takes lensing maps for convergence and shear and produces a lensed
    ellipticity (shear) for each intrinsic galaxy ellipticity.

    The lensing maps are taken from the upper boundary of the matter shell,
    and not interpolated.

    Parameters
    ----------
    cosmo : cosmology
        Cosmology instance for distance functions.

    Receives
    --------
    :data:`~glass.lensing.KAPPA`, array_like
    :data:`~glass.lensing.GAMMA`, array_like
        HEALPix maps for convergence and shear.
    :data:`GAL_LON`, (N,) array_like
    :data:`GAL_LAT`, (N,) array_like
        Arrays for galaxy positions.
    :data:`GAL_ELL`, (N,) array_like
        Array of galaxy :term:`ellipticity (complex)`.

    Yields
    ------
    :data:`GAL_SHE`, (N,) array_like
        Array of complex-valued observed galaxy shears (lensed ellipticities).

    '''

    # initial values
    kap = gam1 = gam2 = np.zeros(12)

    # initial yield
    kap, (gam1, gam2), lon, lat, e = yield

    while True:
        # get the lensing maps at galaxy position
        ngal = len(e)
        k = np.empty(ngal)
        g = np.empty(ngal, dtype=complex)
        nside = healpix.npix2nside(kap.shape[-1])
        for i in range(0, ngal, 10000):
            s = slice(i, i+10000)
            ipix = healpix.ang2pix(nside, lon[s], lat[s], lonlat=True)
            k[s] = kap[ipix]
            g.real[s] = gam1[ipix]
            g.imag[s] = gam2[ipix]
            del ipix

        # compute reduced shear in place
        g /= 1 - k

        # compute lensed ellipticities
        g = (e + g)/(1 + g.conj()*e)

        # clean up
        del kap, gam1, gam2, lon, lat, e

        # yield galaxy shears and receive new data
        kap, (gam1, gam2), lon, lat, e = yield g


@generator(
    receives=(ZSRC, KAPPA, GAMMA, GAL_Z, GAL_LON, GAL_LAT, GAL_ELL),
    yields=GAL_SHE)
def gen_shear_interp(cosmo):
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
        t = cosmo.xm(zsrc_, z)
        np.divide(t, cosmo.xm(z), where=(t != 0), out=t)
        t /= cosmo.xm(zsrc_, zsrc)/cosmo.xm(zsrc)

        # get the lensing maps at galaxy position
        # interpolate in redshifts
        ngal = len(z)
        k = np.empty(ngal)
        g = np.empty(ngal, dtype=complex)
        nside = healpix.npix2nside(kap.shape[-1])
        nside_ = healpix.npix2nside(kap_.shape[-1])
        for i in range(0, ngal, 10000):
            s = slice(i, i+10000)
            ipix = healpix.ang2pix(nside, lon[s], lat[s], lonlat=True)
            ipix_ = healpix.ang2pix(nside_, lon[s], lat[s], lonlat=True)
            for v, m, m_ in (k, kap, kap_), (g.real, gam1, gam1_), (g.imag, gam2, gam2_):
                v[s] = m_[ipix_]
                v[s] *= 1 - t[s]
                v[s] += t[s]*m[ipix]
            del v, m, m_

        # compute reduced shear in place
        g /= 1 - k

        # compute lensed ellipticities
        g = (e + g)/(1 + g.conj()*e)

        # clean up
        del z, lon, lat, e, k


@generator(receives=GAL_Z, yields=GAL_PHZ)
def gen_phz_gausserr(sigma_0, rng=None):
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
