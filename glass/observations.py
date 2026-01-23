"""
Observations
============

.. currentmodule:: glass

The following functions provide functionality for simulating
observational effects of surveys.


Redshift distribution
---------------------

.. autofunction:: gaussian_nz
.. autofunction:: smail_nz
.. autofunction:: fixed_zbins
.. autofunction:: equal_dens_zbins
.. autofunction:: tomo_nz_gausserr


Visibility
----------

.. autofunction:: vmap_galactic_ecliptic

"""  # noqa: D400

from __future__ import annotations

import itertools
import math
from typing import TYPE_CHECKING

import numpy as np

import array_api_compat

import glass._array_api_utils as _utils
import glass.arraytools
import glass.healpix as hp

if TYPE_CHECKING:
    from types import ModuleType

    from glass._types import FloatArray


DEFAULT_XP = _utils.default_xp()


def vmap_galactic_ecliptic(
    nside: int,
    galactic: tuple[float, float] = (30, 90),
    ecliptic: tuple[float, float] = (20, 80),
    *,
    xp: ModuleType = DEFAULT_XP,
) -> FloatArray:
    """
    Visibility map masking galactic and ecliptic plane.

    This function returns a :term:`visibility map` that blocks out stripes for
    the galactic and ecliptic planes. The location of the stripes is set with
    optional parameters.

    Parameters
    ----------
    nside
        The NSIDE parameter of the resulting HEALPix map.
    galactic
        The location of the galactic plane in the respective coordinate system.
    ecliptic
        The location of the ecliptic plane in the respective coordinate system.
    xp
        The array library backend to use for array operations.

    Returns
    -------
        A HEALPix :term:`visibility map`.

    Raises
    ------
    TypeError
        If the ``galactic`` argument is not a pair of numbers.
    TypeError
        If the ``ecliptic`` argument is not a pair of numbers.

    """
    if len(galactic) != 2:
        msg = "galactic stripe must be a pair of numbers"  # type: ignore[unreachable]
        raise TypeError(msg)
    if len(ecliptic) != 2:
        msg = "ecliptic stripe must be a pair of numbers"  # type: ignore[unreachable]
        raise TypeError(msg)

    m = xp.ones(hp.nside2npix(nside))
    m *= 1 - hp.query_strip(nside, galactic, dtype=xp.float64, xp=xp)
    m = hp.Rotator(coord="GC", xp=xp).rotate_map_pixel(m)
    m *= 1 - hp.query_strip(nside, ecliptic, dtype=xp.float64, xp=xp)
    return hp.Rotator(coord="CE", xp=xp).rotate_map_pixel(m)


def gaussian_nz(
    z: FloatArray,
    mean: float | FloatArray,
    sigma: float | FloatArray,
    *,
    norm: float | FloatArray | None = None,
) -> FloatArray:
    """
    Gaussian redshift distribution.

    The redshift follows a Gaussian distribution with the given mean and
    standard deviation.

    If ``mean`` or ``sigma`` are array_like, their axes will be the leading
    axes of the redshift distribution.

    Parameters
    ----------
    z
        Redshift values of the distribution.
    mean
        Mean(s) of the redshift distribution.
    sigma
        Standard deviation(s) of the redshift distribution.
    norm
        If given, the normalisation of the distribution.

    Returns
    -------
        The redshift distribution at the given ``z`` values.

    """
    xp = array_api_compat.array_namespace(z, mean, sigma, norm, use_compat=False)
    uxpx = _utils.XPAdditions(xp)

    mean = xp.asarray(mean, dtype=xp.float64)
    sigma = xp.asarray(sigma, dtype=xp.float64)

    mean = xp.reshape(mean, mean.shape + (1,) * z.ndim)  # type: ignore[union-attr]
    sigma = xp.reshape(sigma, sigma.shape + (1,) * z.ndim)  # type: ignore[union-attr]

    nz = xp.exp(-(((z - mean) / sigma) ** 2) / 2)
    nz /= uxpx.trapezoid(nz, z, axis=-1)[..., xp.newaxis]

    if norm is not None:
        nz *= norm

    return nz


def smail_nz(
    z: FloatArray,
    z_mode: float | FloatArray,
    alpha: float | FloatArray,
    beta: float | FloatArray,
    *,
    norm: float | FloatArray | None = None,
) -> FloatArray:
    r"""
    Redshift distribution following Smail et al. (1994).

    The redshift follows the Smail et al. [Smail94]_ redshift distribution.

    Parameters
    ----------
    z
        Redshift values of the distribution.
    z_mode
        Mode of the redshift distribution, must be positive.
    alpha
        Power law exponent (z/z0)^\alpha, must be positive.
    beta
        Log-power law exponent exp[-(z/z0)^\beta], must be positive.
    norm
        If given, the normalisation of the distribution.

    Returns
    -------
        The redshift distribution at the given ``z`` values.

    Notes
    -----
    The probability distribution function :math:`p(z)` for redshift :math:`z`
    is given by Amara & Refregier [Amara07]_ as

    .. math::

        p(z) \sim \left(\frac{z}{z_0}\right)^\alpha
                    \exp\left[-\left(\frac{z}{z_0}\right)^\beta\right] \;,

    where :math:`z_0` is matched to the given mode of the distribution.

    """
    xp = array_api_compat.array_namespace(
        z,
        z_mode,
        alpha,
        beta,
        norm,
        use_compat=False,
    )
    uxpx = _utils.XPAdditions(xp)

    z_mode = xp.asarray(z_mode, dtype=xp.float64)[..., xp.newaxis]
    alpha = xp.asarray(alpha, dtype=xp.float64)[..., xp.newaxis]
    beta = xp.asarray(beta, dtype=xp.float64)[..., xp.newaxis]

    pz = z**alpha * xp.exp(-alpha / beta * (z / z_mode) ** beta)
    pz /= uxpx.trapezoid(pz, z, axis=-1)[..., xp.newaxis]

    if norm is not None:
        pz *= norm

    return pz


def fixed_zbins(
    zmin: float,
    zmax: float,
    *,
    nbins: int | None = None,
    dz: float | None = None,
    xp: ModuleType = DEFAULT_XP,
) -> list[tuple[float, float]]:
    """
    Tomographic redshift bins of fixed size.

    This function creates contiguous tomographic redshift bins of fixed size.
    It takes either the number or size of the bins.

    Parameters
    ----------
    zmin
        Extent of the redshift binning.
    zmax
        Extent of the redshift binning.
    nbins
        Number of redshift bins. Only one of ``nbins`` and ``dz`` can be given.
    dz
        Size of redshift bin. Only one of ``nbins`` and ``dz`` can be given.
    xp
        The array library backend to use for array operations.

    Returns
    -------
        A list of redshift bin edges.

    Raises
    ------
    ValueError
        If both ``nbins`` and ``dz`` are given.

    """
    if nbins is not None and dz is None:
        zbinedges = xp.linspace(zmin, zmax, nbins + 1)
    elif nbins is None and dz is not None:
        zbinedges = xp.arange(
            zmin,
            xp.nextafter(xp.asarray(zmax + dz), xp.asarray(zmax)),
            dz,
        )
    else:
        msg = "exactly one of nbins and dz must be given"
        raise ValueError(msg)

    return list(itertools.pairwise(zbinedges))


def equal_dens_zbins(
    z: FloatArray,
    nz: FloatArray,
    nbins: int,
) -> list[tuple[float, float]]:
    """
    Equal density tomographic redshift bins.

    This function subdivides a source redshift distribution into ``nbins``
    tomographic redshift bins with equal density.

    Parameters
    ----------
    z
        The source redshift distribution. Must be one-dimensional.
    nz
        The source redshift distribution. Must be one-dimensional.
    nbins
        Number of redshift bins.

    Returns
    -------
        A list of redshift bin edges.

    """
    xp = array_api_compat.array_namespace(z, nz, use_compat=False)
    uxpx = _utils.XPAdditions(xp)

    # compute the normalised cumulative distribution function
    # first compute the cumulative integral (by trapezoidal rule)
    # then normalise: the first z is at CDF = 0, the last z at CDF = 1
    # interpolate to find the z values at CDF = i/nbins for i = 0, ..., nbins
    cuml_nz = glass.arraytools.cumulative_trapezoid(nz, z)
    cuml_nz /= cuml_nz[-1]
    zbinedges = uxpx.interp(xp.linspace(0, 1, nbins + 1), cuml_nz, z)

    return list(itertools.pairwise(zbinedges))


def tomo_nz_gausserr(
    z: FloatArray,
    nz: FloatArray,
    sigma_0: float,
    zbins: list[tuple[float, float]],
) -> FloatArray:
    """
    Tomographic redshift bins with a Gaussian redshift error.

    This function takes a _true_ overall source redshift distribution ``z``,
    ``nz`` and returns tomographic source redshift distributions for the
    tomographic redshift bins given by ``zbins``. It is assumed that sources
    are assigned a tomographic redshift bin with a Gaussian error [Amara07]_. The
    standard deviation of the Gaussian depends on redshift and is given by
    ``sigma(z) = sigma_0*(1 + z)``.

    Parameters
    ----------
    z
        The true source redshift distribution. Must be one-dimensional.
    nz
        The true source redshift distribution. Must be one-dimensional.
    sigma_0
        Redshift error in the tomographic binning at zero redshift.
    zbins
        List of redshift bin edges.

    Returns
    -------
        The tomographic redshift bins convolved with a gaussian error.
        Array has a shape (nbins, len(z))

    See Also
    --------
    equal_dens_zbins:
        produce equal density redshift bins
    fixed_zbins:
        produce redshift bins of fixed size

    """
    xp = array_api_compat.array_namespace(z, nz, use_compat=False)
    uxpx = _utils.XPAdditions(xp)

    # converting zbins into an array:
    zbins_arr = xp.asarray(zbins)

    # bin edges and adds a new axis
    z_lower = zbins_arr[:, 0, xp.newaxis]
    z_upper = zbins_arr[:, 1, xp.newaxis]

    # we need a vectorised version of the error function:
    erf = uxpx.vectorize(math.erf, otypes=(float,))

    # compute the probabilities that redshifts z end up in each bin
    # then apply probability as weights to given nz
    # leading axis corresponds to the different bins
    sz = 2**0.5 * sigma_0 * (1 + z)
    # we need to call xp.asarray here because erf will return a numpy
    # array for array libs which do not implement vectorize.
    binned_nz = xp.asarray(erf((z - z_lower) / sz))
    binned_nz -= xp.asarray(erf((z - z_upper) / sz))
    binned_nz /= 1 + xp.asarray(erf(z / sz))
    binned_nz *= nz

    return binned_nz
