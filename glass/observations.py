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


"""  # noqa: D205, D400

from __future__ import annotations

import math

import healpy as hp
import numpy as np
import numpy.typing as npt

from glass.core.array import cumtrapz


def vmap_galactic_ecliptic(
    nside: int,
    galactic: tuple[float, float] = (30, 90),
    ecliptic: tuple[float, float] = (20, 80),
) -> npt.NDArray[np.float64]:
    """
    _summary_.

    Parameters
    ----------
    nside
        _description_
    galactic
        _description_
    ecliptic
        _description_

    Returns
    -------
        _description_

    Raises
    ------
    TypeError
        _description_
    TypeError
        _description_

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
    z: npt.NDArray[np.float64],
    mean: npt.NDArray[np.float64],
    sigma: npt.NDArray[np.float64],
    *,
    norm: npt.NDArray[np.float64] | None = None,
) -> npt.NDArray[np.float64]:
    """
    _summary_.

    Parameters
    ----------
    z
        _description_
    mean
        _description_
    sigma
        _description_
    norm
        _description_

    Returns
    -------
        _description_

    """
    mean = np.reshape(mean, np.shape(mean) + (1,) * np.ndim(z))
    sigma = np.reshape(sigma, np.shape(sigma) + (1,) * np.ndim(z))

    nz = np.exp(-(((z - mean) / sigma) ** 2) / 2)
    nz /= np.trapz(  # type: ignore[attr-defined]
        nz,
        z,
        axis=-1,
    )[..., np.newaxis]

    if norm is not None:
        nz *= norm

    return nz


def smail_nz(
    z: npt.NDArray[np.float64],
    z_mode: npt.NDArray[np.float64],
    alpha: npt.NDArray[np.float64],
    beta: npt.NDArray[np.float64],
    *,
    norm: npt.NDArray[np.float64] | None = None,
) -> npt.NDArray[np.float64]:
    """
    _summary_.

    Parameters
    ----------
    z
        _description_
    z_mode
        _description_
    alpha
        _description_
    beta
        _description_
    norm
        _description_

    Returns
    -------
        _description_

    """
    z_mode = np.asanyarray(z_mode)[..., np.newaxis]
    alpha = np.asanyarray(alpha)[..., np.newaxis]
    beta = np.asanyarray(beta)[..., np.newaxis]

    pz = z**alpha * np.exp(-alpha / beta * (z / z_mode) ** beta)
    pz /= np.trapz(  # type: ignore[attr-defined]
        pz,
        z,
        axis=-1,
    )[..., np.newaxis]

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
    _summary_.

    Parameters
    ----------
    zmin
        _description_
    zmax
        _description_
    nbins
        _description_
    dz
        _description_

    Returns
    -------
        _description_

    Raises
    ------
    ValueError
        _description_

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
    z: npt.NDArray[np.float64],
    nz: npt.NDArray[np.float64],
    nbins: int,
) -> list[tuple[float, float]]:
    """
    _summary_.

    Parameters
    ----------
    z
        _description_
    nz
        _description_
    nbins
        _description_

    Returns
    -------
        _description_

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
    z: npt.NDArray[np.float64],
    nz: npt.NDArray[np.float64],
    sigma_0: float,
    zbins: list[tuple[float, float]],
) -> npt.NDArray[np.float64]:
    """
    _summary_.

    Parameters
    ----------
    z
        _description_
    nz
        _description_
    sigma_0
        _description_
    zbins
        _description_

    Returns
    -------
        _description_

    """
    # converting zbins into an array:
    zbins_arr = np.asanyarray(zbins)

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
