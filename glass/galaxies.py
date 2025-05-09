"""
Galaxies
========

.. currentmodule:: glass

The following functions provide functionality for simulating galaxies
as typically observed in a cosmological galaxy survey.

Functions
---------

.. autofunction:: redshifts
.. autofunction:: redshifts_from_nz
.. autofunction:: galaxy_shear
.. autofunction:: gaussian_phz

"""  # noqa: D400

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import healpix
import numpy as np

import glass
import glass.arraytools

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from glass.cosmology import Cosmology


def redshifts(
    n: int | NDArray[np.float64],
    w: glass.RadialWindow,
    *,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """
    Sample redshifts from a radial window function.

    This function samples *n* redshifts from a distribution that follows
    the given radial window function *w*.

    Parameters
    ----------
    n
        Number of redshifts to sample. If an array is given, the
        results are concatenated.
    w
        Radial window function.
    rng
        Random number generator. If not given, a default RNG is used.

    Returns
    -------
        Random redshifts following the radial window function.

    """
    return redshifts_from_nz(n, w.za, w.wa, rng=rng, warn=False)


def redshifts_from_nz(
    count: int | NDArray[np.float64],
    z: NDArray[np.float64],
    nz: NDArray[np.float64],
    *,
    rng: np.random.Generator | None = None,
    warn: bool = True,
) -> NDArray[np.float64]:
    """
    Generate galaxy redshifts from a source distribution.

    The function supports sampling from multiple populations of
    redshifts if *count* is an array or if there are additional axes in
    the *z* or *nz* arrays. In this case, the shape of *count* and the
    leading dimensions of *z* and *nz* are broadcast to a common shape,
    and redshifts are sampled independently for each extra dimension.
    The results are concatenated into a flat array.

    Parameters
    ----------
    count
        Number of redshifts to sample. If an array is given, its shape
        is broadcast against the leading axes of *z* and *nz*.
    z
        Source distribution. Leading axes are broadcast against the
        shape of *count*.
    nz
        Source distribution. Leading axes are broadcast against the
        shape of *count*.
    rng
        Random number generator. If not given, a default RNG is used.
    warn
        Throw relevant warnings.

    Returns
    -------
        Redshifts sampled from the given source distribution. For
        inputs with extra dimensions, returns a flattened 1-D array of
        samples from all populations.

    """
    if warn:
        warnings.warn(
            "when sampling galaxies, redshifts_from_nz() is often not the function you"
            " want. Try redshifts() instead. Use warn=False to suppress this warning.",
            stacklevel=2,
        )

    # get default RNG if not given
    if rng is None:
        rng = np.random.default_rng()

    # bring inputs' leading axes into common shape
    dims, *rest = glass.arraytools.broadcast_leading_axes((count, 0), (z, 1), (nz, 1))
    count_out, z_out, nz_out = rest

    # list of results for all dimensions
    redshifts = np.empty(count_out.sum())

    # keep track of the number of sampled redshifts
    total = 0

    # go through extra dimensions; also works if dims is empty
    for k in np.ndindex(dims):
        # compute the CDF of each galaxy population
        cdf = glass.arraytools.cumulative_trapezoid(nz_out[k], z_out[k], dtype=float)
        cdf /= cdf[-1]

        # sample redshifts and store result
        redshifts[total : total + count_out[k]] = np.interp(
            rng.uniform(0, 1, size=count_out[k]),
            cdf,
            z_out[k],
        )
        total += count_out[k]

    assert total == redshifts.size  # noqa: S101

    return redshifts


def galaxy_shear(  # noqa: PLR0913
    lon: NDArray[np.float64],
    lat: NDArray[np.float64],
    eps: NDArray[np.float64],
    kappa: NDArray[np.float64],
    gamma1: NDArray[np.float64],
    gamma2: NDArray[np.float64],
    *,
    reduced_shear: bool = True,
) -> NDArray[np.float64]:
    """
    Observed galaxy shears from weak lensing.

    Takes lensing maps for convergence and shear and produces a lensed
    ellipticity (shear) for each intrinsic galaxy ellipticity.

    Parameters
    ----------
    lon
        Array for galaxy longitudes.
    lat
        Array for galaxy latitudes.
    eps
        Array of galaxy :term:`ellipticity`.
    kappa
        HEALPix map for convergence.
    gamma1
        HEALPix maps for a component of shear.
    gamma2
        HEALPix maps for a component of shear.
    reduced_shear
        If ``False``, galaxy shears are not reduced
        by the convergence. Default is ``True``.

    Returns
    -------
        An array of complex-valued observed galaxy shears
        (lensed ellipticities).

    """
    nside = healpix.npix2nside(np.broadcast(kappa, gamma1, gamma2).shape[-1])

    size = np.broadcast(lon, lat, eps).size

    # output arrays
    k = np.empty(size)
    g = np.empty(size, dtype=complex)

    # get the lensing maps at galaxy position
    for i in range(0, size, 10000):
        s = slice(i, i + 10000)
        ipix = healpix.ang2pix(nside, lon[s], lat[s], lonlat=True)
        k[s] = kappa[ipix]
        g.real[s] = gamma1[ipix]
        g.imag[s] = gamma2[ipix]

    if reduced_shear:
        # compute reduced shear in place
        g /= 1 - k

        # compute lensed ellipticities
        g = (eps + g) / (1 + g.conj() * eps)
    else:
        # simple sum of shears
        g += eps

    return g


def gaussian_phz(
    z: float | NDArray[np.float64],
    sigma_0: float | NDArray[np.float64],
    *,
    lower: float | NDArray[np.float64] | None = None,
    upper: float | NDArray[np.float64] | None = None,
    rng: np.random.Generator | None = None,
) -> float | NDArray[np.float64]:
    r"""
    Photometric redshifts assuming a Gaussian error.

    A simple toy model of photometric redshift errors that assumes a
    Gaussian error with redshift-dependent standard deviation
    :math:`\sigma(z) = (1 + z) \sigma_0` [Amara07]_.

    Parameters
    ----------
    z
        True redshifts.
    sigma_0
        Redshift error in the tomographic binning at zero redshift.
    lower
        Bounds for the returned photometric redshifts.
    upper
        Bounds for the returned photometric redshifts.
    rng
        Random number generator. If not given, a default RNG is used.

    Returns
    -------
        Photometric redshifts assuming Gaussian errors, of the same
        shape as *z*.

    Raises
    ------
    ValueError
        If the bounds are not consistent.

    Warnings
    --------
    The *lower* and *upper* bounds are implemented using plain rejection
    sampling from the non-truncated normal distribution. If bounds are
    used, they should always contain significant probability mass.

    See Also
    --------
    glass.tomo_nz_gausserr:
        Create tomographic redshift distributions assuming the same model.

    Examples
    --------
    See the :doc:`/examples/1-basic/photoz` example.

    """
    # get default RNG if not given
    if rng is None:
        rng = np.random.default_rng()

    sigma = np.add(1, z) * sigma_0
    dims = np.shape(sigma)

    zphot = rng.normal(z, sigma)

    if lower is None:
        lower = 0.0
    if upper is None:
        upper = np.inf

    if not np.all(lower < upper):
        msg = "requires lower < upper"
        raise ValueError(msg)

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


def _kappa_ia_nla(  # noqa: PLR0913
    delta: NDArray[np.float64],
    zeff: float,
    a_ia: float,
    cosmo: Cosmology,
    *,
    z0: float = 0.0,
    eta: float = 0.0,
    lbar: float = 0.0,
    l0: float = 1e-9,
    beta: float = 0.0,
) -> NDArray[np.float64]:
    r"""
    Effective convergence from intrinsic alignments using the NLA model.

    Parameters
    ----------
    delta
        Matter density contrast.
    zeff
        Effective redshift of the matter field.
    a_ia
        Intrinsic alignments amplitude.
    cosmo
        Cosmology instance.
    z0
        Reference redshift for the redshift dependence.
    eta
        Power of the redshift dependence.
    lbar
        Mean luminosity of the galaxy sample.
    l0
        Reference luminosity for the luminosity dependence.
    beta
        Power of the luminosity dependence.

    Returns
    -------
        The effective convergence due to intrinsic alignments.

    Notes
    -----
    The Non-linear Alignments Model (NLA) describes an effective
    convergence :math:`\kappa_{\rm IA}` that models the effect of
    intrinsic alignments. It is computed from the matter density
    contrast :math:`\delta` as [Catelan01_] [Bridle07]_

    .. math::

        \kappa_{\rm IA} = f_{\rm NLA} \, \delta \;,

    where the NLA factor :math:`f_{\rm NLA}` is defined as [Johnston19]_ [Tessore23]_

    .. math::

        f_{\rm{NLA}}
        = -A_{\rm IA} \, \frac{C_1 \, \bar{\rho}(z)}{D(z)} \,
            \biggl(\frac{1+z}{1+z_0}\biggr)^\eta \,
            \biggl(\frac{\bar{L}}{L_0}\biggr)^\beta \;,

    with

    * :math:`A_{\rm IA}` the intrinsic alignments amplitude,
    * :math:`C_1` a normalisation constant [Hirata04]_,
    * :math:`z` the effective redshift of the model,
    * :math:`\bar{\rho}` the mean matter density,
    * :math:`D` the growth factor,
    * :math:`\eta` the power that describes the redshift-dependence with
      respect to :math:`z_0`,
    * :math:`\bar{L}` the mean luminosity of the galaxy sample, and
    * :math:`\beta` the power that describes the luminosity-dependence
      :math:`\bar{L}` with respect to :math:`L_0`.

    """
    c1 = 5e-14 / cosmo.h**2  # Solar masses per cubic Mpc
    rho_c1 = c1 * cosmo.critical_density0

    prefactor = -a_ia * rho_c1 * cosmo.Omega_m0
    inverse_linear_growth = 1.0 / cosmo.growth_factor(zeff)
    redshift_dependence = ((1 + zeff) / (1 + z0)) ** eta
    luminosity_dependence = (lbar / l0) ** beta

    f_nla = (
        prefactor * inverse_linear_growth * redshift_dependence * luminosity_dependence
    )

    return delta * f_nla  # type: ignore[no-any-return]
