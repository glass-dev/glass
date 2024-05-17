# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''
Galaxies (:mod:`glass.galaxies`)
================================

.. currentmodule:: glass.galaxies

The :mod:`glass.galaxies` module provides functionality for simulating galaxies
as typically observed in a cosmological galaxy survey.

Functions
---------

.. autofunction:: redshifts
.. autofunction:: redshifts_from_nz
.. autofunction:: galaxy_shear
.. autofunction:: gaussian_phz


Intrinsic alignments
--------------------
.. autofunction:: kappa_ia_nla

'''

from __future__ import annotations

import numpy as np
import healpix

from numpy.typing import ArrayLike

from .core.array import broadcast_leading_axes, cumtrapz
from .shells import RadialWindow


def redshifts(n: int | ArrayLike, w: RadialWindow, *,
              rng: np.random.Generator | None = None
              ) -> np.ndarray:
    '''Sample redshifts from a radial window function.

    This function samples *n* redshifts from a distribution that follows
    the given radial window function *w*.

    Parameters
    ----------
    n : int or array_like
        Number of redshifts to sample.  If an array is given, the
        results are concatenated.
    w : :class:`~glass.shells.RadialWindow`
        Radial window function.
    rng : :class:`~numpy.random.Generator`, optional
        Random number generator.  If not given, a default RNG is used.

    Returns
    -------
    z : array_like
        Random redshifts following the radial window function.

    '''
    return redshifts_from_nz(n, w.za, w.wa, rng=rng)


def redshifts_from_nz(count: int | ArrayLike, z: ArrayLike, nz: ArrayLike, *,
                      rng: np.random.Generator | None = None
                      ) -> np.ndarray:
    '''Generate galaxy redshifts from a source distribution.

    The function supports sampling from multiple populations of
    redshifts if *count* is an array or if there are additional axes in
    the *z* or *nz* arrays.  In this case, the shape of *count* and the
    leading dimensions of *z* and *nz* are broadcast to a common shape,
    and redshifts are sampled independently for each extra dimension.
    The results are concatenated into a flat array.

    Parameters
    ----------
    count : int or array_like
        Number of redshifts to sample.  If an array is given, its shape
        is broadcast against the leading axes of *z* and *nz*.
    z, nz : array_like
        Source distribution.  Leading axes are broadcast against the
        shape of *count*.
    rng : :class:`~numpy.random.Generator`, optional
        Random number generator.  If not given, a default RNG is used.

    Returns
    -------
    redshifts : array_like
        Redshifts sampled from the given source distribution.  For
        inputs with extra dimensions, returns a flattened 1-D array of
        samples from all populations.

    '''

    # get default RNG if not given
    if rng is None:
        rng = np.random.default_rng()

    # bring inputs' leading axes into common shape
    dims, count, z, nz = broadcast_leading_axes((count, 0), (z, 1), (nz, 1))

    # list of results for all dimensions
    redshifts = np.empty(count.sum())

    # keep track of the number of sampled redshifts
    total = 0

    # go through extra dimensions; also works if dims is empty
    for k in np.ndindex(dims):

        # compute the CDF of each galaxy population
        cdf = cumtrapz(nz[k], z[k], dtype=float)
        cdf /= cdf[-1]

        # sample redshifts and store result
        redshifts[total:total+count[k]] = np.interp(rng.uniform(0, 1, size=count[k]), cdf, z[k])
        total += count[k]

    assert total == redshifts.size

    return redshifts


def galaxy_shear(lon: np.ndarray, lat: np.ndarray, eps: np.ndarray,
                 kappa: np.ndarray, gamma1: np.ndarray, gamma2: np.ndarray, *,
                 reduced_shear: bool = True) -> np.ndarray:
    '''Observed galaxy shears from weak lensing.

    Takes lensing maps for convergence and shear and produces a lensed
    ellipticity (shear) for each intrinsic galaxy ellipticity.

    Parameters
    ----------
    lon, lat : array_like
        Arrays for galaxy longitudes and latitudes.
    eps : array_like
        Array of galaxy :term:`ellipticity`.
    kappa, gamma1, gamma2 : array_like
        HEALPix maps for convergence and two components of shear.
    reduced_shear : bool, optional
        If ``False``, galaxy shears are not reduced by the convergence.
        Default is ``True``.

    Returns
    -------
    she : array_like
        Array of complex-valued observed galaxy shears (lensed ellipticities).

    '''

    nside = healpix.npix2nside(np.broadcast(kappa, gamma1, gamma2).shape[-1])

    size = np.broadcast(lon, lat, eps).size

    # output arrays
    k = np.empty(size)
    g = np.empty(size, dtype=complex)

    # get the lensing maps at galaxy position
    for i in range(0, size, 10000):
        s = slice(i, i+10000)
        ipix = healpix.ang2pix(nside, lon[s], lat[s], lonlat=True)
        k[s] = kappa[ipix]
        g.real[s] = gamma1[ipix]
        g.imag[s] = gamma2[ipix]

    if reduced_shear:
        # compute reduced shear in place
        g /= 1 - k

        # compute lensed ellipticities
        g = (eps + g)/(1 + g.conj()*eps)
    else:
        # simple sum of shears
        g += eps

    return g


def gaussian_phz(z: ArrayLike, sigma_0: float | ArrayLike, *,
                 lower: ArrayLike | None = None,
                 upper: ArrayLike | None = None,
                 rng: np.random.Generator | None = None) -> np.ndarray:
    r'''Photometric redshifts assuming a Gaussian error.

    A simple toy model of photometric redshift errors that assumes a
    Gaussian error with redshift-dependent standard deviation
    :math:`\sigma(z) = (1 + z) \sigma_0` [1]_.

    Parameters
    ----------
    z : array_like
        True redshifts.
    sigma_0 : float or array_like
        Redshift error in the tomographic binning at zero redshift.
    lower, upper : float or array_like, optional
        Bounds for the returned photometric redshifts.
    rng : :class:`~numpy.random.Generator`, optional
        Random number generator.  If not given, a default RNG is used.

    Returns
    -------
    phz : array_like
        Photometric redshifts assuming Gaussian errors, of the same
        shape as *z*.

    Warnings
    --------
    The *lower* and *upper* bounds are implemented using plain rejection
    sampling from the non-truncated normal distribution.  If bounds are
    used, they should always contain significant probability mass.

    See Also
    --------
    glass.observations.tomo_nz_gausserr :
        Create tomographic redshift distributions assuming the same
        model.

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

    sigma = np.add(1, z)*sigma_0
    dims = np.shape(sigma)

    zphot = rng.normal(z, sigma)

    if lower is None:
        lower = 0.
    if upper is None:
        upper = np.inf

    if not np.all(lower < upper):
        raise ValueError("requires lower < upper")

    if not dims:
        while zphot < lower or zphot > upper:
            zphot = rng.normal(z, sigma)
    else:
        z = np.broadcast_to(z, dims)
        trunc = np.where((zphot < lower) | (zphot > upper))[0]
        while trunc.size:
            znew = rng.normal(z[trunc], sigma[trunc])
            zphot[trunc] = znew
            trunc = trunc[(znew < lower) | (znew > upper)]

    return zphot


def kappa_ia_nla(delta, zeff, a_ia, cosmo, *, z0=0., eta=0., lbar=0.,
                 l0=1e-9, beta=0.):
    r'''Effective convergence from intrinsic alignments using the NLA
    model.

    Parameters
    ----------
    delta : array_like
        Matter density contrast.
    zeff : float
        Effective redshift of the matter field.
    a_ia : float
        Intrinsic alignments amplitude.
    cosmo : Cosmology
        Cosmology instance.

    Returns
    -------
    kappa_ia : array_like
        Effective convergence due to intrinsic alignments.

    Other Parameters
    ----------------
    z0 : float, optional
        Reference redshift for the redshift dependence (default: 0).
    eta : float, optional
        Power of the redshift dependence (default: 0).
    lbar : float, optional
        Mean luminosity of the galaxy sample (default: 0).
    l0 : float, optional
        Reference luminosity for the luminosity dependence (default:
        ``1e-09``).
    beta : float, optional
        Power of the luminosity dependence (default: 0).

    Notes
    -----
    The Non-linear Alignments Model (NLA) describes an effective
    convergence :math:`\kappa_{\rm IA}` that models the effect of
    intrinsic alignments.  It is computed from the matter density
    contrast :math:`\delta` as [1]_ [3]_

    .. math::

        \kappa_{\rm IA} = f_{\rm NLA} \, \delta \;,

    where the NLA factor :math:`f_{\rm NLA}` is defined as [4]_ [5]_

    .. math::

        f_{\rm{NLA}}
        = -A_{\rm IA} \, \frac{C_1 \, \bar{\rho}(z)}{D(z)} \,
            \biggl(\frac{1+z}{1+z_0}\biggr)^\eta \,
            \biggl(\frac{\bar{L}}{L_0}\biggr)^\beta \;,

    with

    * :math:`A_{\rm IA}` the intrinsic alignments amplitude,
    * :math:`C_1` a normalisation constant [2]_,
    * :math:`z` the effective redshift of the model,
    * :math:`\bar{\rho}` the mean matter density,
    * :math:`D` the growth factor,
    * :math:`\eta` the power that describes the redshift-dependence with
      respect to :math:`z_0`,
    * :math:`\bar{L}` the mean luminosity of the galaxy sample, and
    * :math:`\beta` the power that describes the luminosity-dependence
      :math:`\bar{L}` with respect to :math:`L_0`.

    References
    ----------
    .. [1] Catelan P., Kamionkowski M., Blandford R. D., 2001, MNRAS,
       320, L7. doi:10.1046/j.1365-8711.2001.04105.x
    .. [2] Hirata C. M., Seljak U., 2004, PhRvD, 70, 063526.
       doi:10.1103/PhysRevD.70.063526
    .. [3] Bridle S., King L., 2007, NJPh, 9, 444.
       doi:10.1088/1367-2630/9/12/444
    .. [4] Johnston, H., Georgiou, C., Joachimi, B., et al., 2019,
        A&A, 624, A30. doi:10.1051/0004-6361/201834714
    .. [5] Tessore, N., Loureiro, A., Joachimi, B., et al., 2023,
       OJAp, 6, 11. doi:10.21105/astro.2302.01942

    '''

    c1 = 5e-14/cosmo.h**2  # Solar masses per cubic Mpc
    rho_c1 = c1*cosmo.rho_c0

    prefactor = - a_ia * rho_c1 * cosmo.Om
    inverse_linear_growth = 1./cosmo.gf(zeff)
    redshift_dependence = ((1+zeff)/(1+z0))**eta
    luminosity_dependence = (lbar/l0)**beta

    f_nla = prefactor * inverse_linear_growth * redshift_dependence \
        * luminosity_dependence

    return delta * f_nla
