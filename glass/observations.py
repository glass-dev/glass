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


"""  # noqa: D205, D400, D415

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import healpy as hp
import numpy as np

from glass.core.array import cumtrapz

if TYPE_CHECKING:
    import numpy.typing as npt


def vmap_galactic_ecliptic(
    nside: int,
    galactic: tuple[float, float] = (30, 90),
    ecliptic: tuple[float, float] = (20, 80),
) -> npt.NDArray:
    """
    Visibility map masking galactic and ecliptic plane.

    This function returns a :term:`visibility map` that blocks out stripes for
    the galactic and ecliptic planes.  The location of the stripes is set with
    optional parameters.

    Parameters
    ----------
    nside : int
        The NSIDE parameter of the resulting HEALPix map.
    galactic, ecliptic : (2,) tuple of float
        The location of the galactic and ecliptic plane in their respective
        coordinate systems.

    Returns
    -------
    vis : array_like
        A HEALPix :term:`visibility map`.

    Raises
    ------
    TypeError
        If the ``galactic`` or ``ecliptic`` arguments are not pairs of numbers.

    """
    if np.ndim(galactic) != 1 or len(galactic) != 2:
        msg = "galactic stripe must be a pair of numbers"
        raise TypeError(msg)
    if np.ndim(ecliptic) != 1 or len(ecliptic) != 2:
        msg = "ecliptic stripe must be a pair of numbers"
        raise TypeError(msg)

    m = np.ones(hp.nside2npix(nside))
    m[hp.query_strip(nside, *galactic)] = 0
    m = hp.Rotator(coord="GC").rotate_map_pixel(m)
    m[hp.query_strip(nside, *ecliptic)] = 0
    return hp.Rotator(coord="CE").rotate_map_pixel(m)


def gaussian_nz(
    z: npt.NDArray,
    mean: npt.ArrayLike,
    sigma: npt.ArrayLike,
    *,
    norm: npt.ArrayLike | None = None,
) -> npt.NDArray:
    r"""
    Gaussian redshift distribution.

    The redshift follows a Gaussian distribution with the given mean and
    standard deviation.

    If ``mean`` or ``sigma`` are array_like, their axes will be the leading
    axes of the redshift distribution.

    Parameters
    ----------
    z : array_like
        Redshift values of the distribution.
    mean : float or array_like
        Mean(s) of the redshift distribution.
    sigma : float or array_like
        Standard deviation(s) of the redshift distribution.
    norm : float or array_like, optional
        If given, the normalisation of the distribution.

    Returns
    -------
    nz : array_like
        Redshift distribution at the given ``z`` values.

    """
    mean = np.reshape(mean, np.shape(mean) + (1,) * np.ndim(z))
    sigma = np.reshape(sigma, np.shape(sigma) + (1,) * np.ndim(z))

    nz = np.exp(-(((z - mean) / sigma) ** 2) / 2)
    nz /= np.trapz(nz, z, axis=-1)[..., np.newaxis]

    if norm is not None:
        nz *= norm

    return nz


def smail_nz(
    z: npt.NDArray,
    z_mode: npt.ArrayLike,
    alpha: npt.ArrayLike,
    beta: npt.ArrayLike,
    *,
    norm: npt.ArrayLike | None = None,
) -> npt.NDArray:
    r"""
    Redshift distribution following Smail et al. (1994).

    The redshift follows the Smail et al. [1]_ redshift distribution.

    Parameters
    ----------
    z : array_like
        Redshift values of the distribution.
    z_mode : float or array_like
        Mode of the redshift distribution, must be positive.
    alpha : float or array_like
        Power law exponent (z/z0)^\alpha, must be positive.
    beta : float or array_like
        Log-power law exponent exp[-(z/z0)^\beta], must be positive.
    norm : float or array_like, optional
        If given, the normalisation of the distribution.

    Returns
    -------
    pz : array_like
        Redshift distribution at the given ``z`` values.

    Notes
    -----
    The probability distribution function :math:`p(z)` for redshift :math:`z`
    is given by Amara & Refregier [2]_ as

    .. math::

        p(z) \sim \left(\frac{z}{z_0}\right)^\alpha
                    \exp\left[-\left(\frac{z}{z_0}\right)^\beta\right] \;,

    where :math:`z_0` is matched to the given mode of the distribution.

    References
    ----------
    .. [1] Smail I., Ellis R. S., Fitchett M. J., 1994, MNRAS, 270, 245
    .. [2] Amara A., Refregier A., 2007, MNRAS, 381, 1018

    """
    z_mode = np.asanyarray(z_mode)[..., np.newaxis]
    alpha = np.asanyarray(alpha)[..., np.newaxis]
    beta = np.asanyarray(beta)[..., np.newaxis]

    pz = z**alpha * np.exp(-alpha / beta * (z / z_mode) ** beta)
    pz /= np.trapz(pz, z, axis=-1)[..., np.newaxis]

    if norm is not None:
        pz *= norm

    return pz


def fixed_zbins(
    zmin: float,
    zmax: float,
    *,
    nbins: int | None = None,
    dz: float | None = None,
) -> list[tuple[float, float]]:
    """
    Tomographic redshift bins of fixed size.

    This function creates contiguous tomographic redshift bins of fixed size.
    It takes either the number or size of the bins.

    Parameters
    ----------
    zmin, zmax : float
        Extent of the redshift binning.
    nbins : int, optional
        Number of redshift bins.  Only one of ``nbins`` and ``dz`` can be given.
    dz : float, optional
        Size of redshift bin.  Only one of ``nbins`` and ``dz`` can be given.

    Returns
    -------
    zbins : list of tuple of float
        List of redshift bin edges.

    """
    if nbins is not None and dz is None:
        zbinedges = np.linspace(zmin, zmax, nbins + 1)
    if nbins is None and dz is not None:
        zbinedges = np.arange(zmin, zmax, dz)
    else:
        msg = "exactly one of nbins and dz must be given"
        raise ValueError(msg)

    return list(zip(zbinedges, zbinedges[1:]))


def equal_dens_zbins(
    z: npt.NDArray,
    nz: npt.NDArray,
    nbins: int,
) -> list[tuple[float, float]]:
    """
    Equal density tomographic redshift bins.

    This function subdivides a source redshift distribution into ``nbins``
    tomographic redshift bins with equal density.

    Parameters
    ----------
    z, nz : array_like
        The source redshift distribution. Must be one-dimensional.
    nbins : int
        Number of redshift bins.

    Returns
    -------
    zbins : list of tuple of float
        List of redshift bin edges.

    """
    # compute the normalised cumulative distribution function
    # first compute the cumulative integral (by trapezoidal rule)
    # then normalise: the first z is at CDF = 0, the last z at CDF = 1
    # interpolate to find the z values at CDF = i/nbins for i = 0, ..., nbins
    cuml_nz = cumtrapz(nz, z)
    cuml_nz /= cuml_nz[[-1]]
    zbinedges = np.interp(np.linspace(0, 1, nbins + 1), cuml_nz, z)

    return list(zip(zbinedges, zbinedges[1:]))


def tomo_nz_gausserr(
    z: npt.NDArray,
    nz: npt.NDArray,
    sigma_0: float,
    zbins: list[tuple[float, float]],
) -> npt.NDArray:
    """
    Tomographic redshift bins with a Gaussian redshift error.

    This function takes a _true_ overall source redshift distribution ``z``,
    ``nz`` and returns tomographic source redshift distributions for the
    tomographic redshift bins given by ``zbins``.  It is assumed that sources
    are assigned a tomographic redshift bin with a Gaussian error [1]_. The
    standard deviation of the Gaussian depends on redshift and is given by
    ``sigma(z) = sigma_0*(1 + z)``.

    Parameters
    ----------
    z, nz : array_like
        The true source redshift distribution. Must be one-dimensional.
    sigma_0 : float
        Redshift error in the tomographic binning at zero redshift.
    zbins : list of tuple of float
        List of redshift bin edges.

    Returns
    -------
    binned_nz : array_like
        Tomographic redshift bins convolved with a gaussian error.
        Array has a shape (nbins, len(z))

    See Also
    --------
    equal_dens_zbins :
        produce equal density redshift bins
    fixed_zbins :
        produce redshift bins of fixed size

    References
    ----------
    .. [1] Amara A., Réfrégier A., 2007, MNRAS, 381, 1018.
           doi:10.1111/j.1365-2966.2007.12271.x

    """
    # converting zbins into an array:
    zbins_arr = np.asanyarray(zbins)  # type: ignore[no-redef]

    # bin edges and adds a new axis
    z_lower = zbins_arr[:, 0, np.newaxis]
    z_upper = zbins_arr[:, 1, np.newaxis]

    # we need a vectorised version of the error function:
    erf = np.vectorize(math.erf, otypes=(float,))

    # compute the probabilities that redshifts z end up in each bin
    # then apply probability as weights to given nz
    # leading axis corresponds to the different bins
    sz = 2**0.5 * sigma_0 * (1 + z)
    binned_nz = erf((z - z_lower) / sz)
    binned_nz -= erf((z - z_upper) / sz)
    binned_nz /= 1 + erf(z / sz)
    binned_nz *= nz

    return binned_nz
