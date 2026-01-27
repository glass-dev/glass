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

import math
import warnings
from typing import TYPE_CHECKING

import numpy as np

import array_api_compat

import glass.arraytools
import glass.healpix as hp
import glass.shells
from glass import _rng
from glass._array_api_utils import xp_additions as uxpx

if TYPE_CHECKING:
    from types import ModuleType

    from glass._types import FloatArray, UnifiedGenerator
    from glass.cosmology import Cosmology


def redshifts(
    n: int | FloatArray,
    w: glass.shells.RadialWindow,
    *,
    rng: UnifiedGenerator | None = None,
) -> FloatArray:
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
    count: int | FloatArray,
    z: FloatArray,
    nz: FloatArray,
    *,
    rng: UnifiedGenerator | None = None,
    warn: bool = True,
) -> FloatArray:
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
    xp = array_api_compat.array_namespace(count, z, nz, use_compat=False)

    if warn:
        warnings.warn(
            "when sampling galaxies, redshifts_from_nz() is often not the function you"
            " want. Try redshifts() instead. Use warn=False to suppress this warning.",
            stacklevel=2,
        )

    # get default RNG if not given
    if rng is None:
        rng = _rng.rng_dispatcher(xp=xp)

    # bring inputs' leading axes into common shape
    dims, *rest = glass.arraytools.broadcast_leading_axes((count, 0), (z, 1), (nz, 1))
    count_out, z_out, nz_out = rest

    # list of results for all dimensions
    redshifts = xp.empty(xp.sum(count_out))

    # keep track of the number of sampled redshifts
    total = 0

    # go through extra dimensions; also works if dims is empty
    for k in uxpx.ndindex(dims, xp=xp):
        nz_out_slice = nz_out[(*k, ...)] if k != () else nz_out
        z_out_slice = z_out[(*k, ...)] if k != () else z_out

        # compute the CDF of each galaxy population
        cdf = glass.arraytools.cumulative_trapezoid(nz_out_slice, z_out_slice)
        cdf /= cdf[-1]

        # sample redshifts and store result
        redshifts[total : total + count_out[k]] = uxpx.interp(
            rng.uniform(0, 1, size=int(count_out[k])),
            cdf,
            z_out_slice,
        )
        total += count_out[k]

    assert total == redshifts.size  # noqa: S101

    return redshifts


def galaxy_shear(  # noqa: PLR0913
    lon: FloatArray,
    lat: FloatArray,
    eps: FloatArray,
    kappa: FloatArray,
    gamma1: FloatArray,
    gamma2: FloatArray,
    *,
    reduced_shear: bool = True,
) -> FloatArray:
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
    nside = hp.npix2nside(np.broadcast(kappa, gamma1, gamma2).shape[-1])

    size = np.broadcast(lon, lat, eps).size

    # output arrays
    k = np.empty(size)
    g = np.empty(size, dtype=complex)

    # get the lensing maps at galaxy position
    for i in range(0, size, 10_000):
        s = slice(i, i + 10_000)
        ipix = hp.ang2pix(nside, lon[s], lat[s], lonlat=True, xp=np)
        k[s] = kappa[ipix]
        np.real(g)[s] = gamma1[ipix]
        np.imag(g)[s] = gamma2[ipix]

    if reduced_shear:
        # compute reduced shear in place
        g /= 1 - k

        # compute lensed ellipticities
        g = (eps + g) / (1 + g.conj() * eps)
    else:
        # simple sum of shears
        g += eps

    return g


def gaussian_phz(  # noqa: PLR0913
    z: float | FloatArray,
    sigma_0: float | FloatArray,
    *,
    lower: float | FloatArray | None = None,
    upper: float | FloatArray | None = None,
    rng: UnifiedGenerator | None = None,
    xp: ModuleType | None = None,
) -> FloatArray:
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
    xp
        The array library backend to use for array operations. If this is not
        specified, the backend will be determined from the input arrays.

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
    if xp is None:
        xp = array_api_compat.array_namespace(
            z,
            sigma_0,
            lower,
            upper,
            use_compat=False,
        )

    # Ensure inputs are arrays to allow lib utilisation
    z_arr = xp.asarray(z)
    sigma_0_arr = xp.asarray(sigma_0)

    # get default RNG if not given
    if rng is None:
        rng = _rng.rng_dispatcher(xp=xp)

    # Ensure lower and upper are arrays that have the same shape and type
    lower_arr = xp.asarray(0.0 if lower is None else lower, dtype=xp.float64)
    upper_arr = xp.asarray(xp.inf if upper is None else upper, dtype=xp.float64)
    if lower is None and upper is not None:
        lower_arr = xp.zeros_like(upper_arr, dtype=xp.float64)
    if upper is None and lower is not None:
        upper_arr = xp.full_like(lower_arr, fill_value=math.inf, dtype=xp.float64)

    sigma = xp.add(1, z_arr) * sigma_0_arr
    dims = sigma.shape
    zphot = xp.asarray(rng.normal(z_arr, sigma))

    # Check for valid user input
    if (lower_arr.ndim == upper_arr.ndim != 0) and not (
        lower_arr.shape == upper_arr.shape == zphot.shape
    ):
        msg = "lower and upper must best scalars or have the same shape as z"
        raise ValueError(msg)
    if not xp.all(lower_arr < upper_arr):
        msg = "requires lower < upper"
        raise ValueError(msg)

    if not dims:
        while zphot < lower_arr or zphot > upper_arr:
            zphot = xp.asarray(rng.normal(z_arr, sigma))
    else:
        z_arr = xp.broadcast_to(z_arr, dims)
        trunc = (zphot < lower_arr) | (zphot > upper_arr)
        while xp.count_nonzero(trunc) > 0:
            zphot = xp.where(trunc, rng.normal(z_arr, sigma), zphot)
            trunc = (zphot < lower_arr) | (zphot > upper_arr)

    return zphot


def _kappa_ia_nla(  # noqa: PLR0913
    delta: FloatArray,
    zeff: float,
    a_ia: float,
    cosmo: Cosmology,
    *,
    z0: float = 0.0,
    eta: float = 0.0,
    lbar: float = 0.0,
    l0: float = 1e-9,
    beta: float = 0.0,
) -> FloatArray:
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

    return delta * f_nla
