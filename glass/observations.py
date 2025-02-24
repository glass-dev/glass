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
.. class:: AngularVariableDepthMask
.. class:: AngularLosVariableDepthMask

"""  # noqa: D400

from __future__ import annotations

import itertools
import math
from typing import TYPE_CHECKING

import healpy as hp
import numpy as np

import glass.arraytools

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from collections.abc import Callable


def vmap_galactic_ecliptic(
    nside: int,
    galactic: tuple[float, float] = (30, 90),
    ecliptic: tuple[float, float] = (20, 80),
) -> NDArray[np.float64]:
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
    return hp.Rotator(coord="CE").rotate_map_pixel(m)  # type: ignore[no-any-return]


def gaussian_nz(
    z: NDArray[np.float64],
    mean: float | NDArray[np.float64],
    sigma: float | NDArray[np.float64],
    *,
    norm: float | NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
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
    mean = np.reshape(mean, np.shape(mean) + (1,) * np.ndim(z))
    sigma = np.reshape(sigma, np.shape(sigma) + (1,) * np.ndim(z))

    nz = np.exp(-(((z - mean) / sigma) ** 2) / 2)
    nz /= np.trapezoid(nz, z, axis=-1)[..., np.newaxis]

    if norm is not None:
        nz *= norm

    return nz


def smail_nz(
    z: NDArray[np.float64],
    z_mode: float | NDArray[np.float64],
    alpha: float | NDArray[np.float64],
    beta: float | NDArray[np.float64],
    *,
    norm: float | NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
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
    z_mode = np.asanyarray(z_mode)[..., np.newaxis]
    alpha = np.asanyarray(alpha)[..., np.newaxis]
    beta = np.asanyarray(beta)[..., np.newaxis]

    pz = z**alpha * np.exp(-alpha / beta * (z / z_mode) ** beta)
    pz /= np.trapezoid(pz, z, axis=-1)[..., np.newaxis]

    if norm is not None:
        pz *= norm

    return pz  # type: ignore[no-any-return]


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
    zmin
        Extent of the redshift binning.
    zmax
        Extent of the redshift binning.
    nbins
        Number of redshift bins. Only one of ``nbins`` and ``dz`` can be given.
    dz
        Size of redshift bin. Only one of ``nbins`` and ``dz`` can be given.

    Returns
    -------
        A list of redshift bin edges.

    Raises
    ------
    ValueError
        If both ``nbins`` and ``dz`` are given.

    """
    if nbins is not None and dz is None:
        zbinedges = np.linspace(zmin, zmax, nbins + 1)
    elif nbins is None and dz is not None:
        zbinedges = np.arange(zmin, np.nextafter(zmax + dz, zmax), dz)
    else:
        msg = "exactly one of nbins and dz must be given"
        raise ValueError(msg)

    return list(itertools.pairwise(zbinedges))


def equal_dens_zbins(
    z: NDArray[np.float64],
    nz: NDArray[np.float64],
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
    # compute the normalised cumulative distribution function
    # first compute the cumulative integral (by trapezoidal rule)
    # then normalise: the first z is at CDF = 0, the last z at CDF = 1
    # interpolate to find the z values at CDF = i/nbins for i = 0, ..., nbins
    cuml_nz = glass.arraytools.cumulative_trapezoid(nz, z)
    cuml_nz /= cuml_nz[[-1]]
    zbinedges = np.interp(np.linspace(0, 1, nbins + 1), cuml_nz, z)

    return list(itertools.pairwise(zbinedges))


def tomo_nz_gausserr(
    z: NDArray[np.float64],
    nz: NDArray[np.float64],
    sigma_0: float,
    zbins: list[tuple[float, float]],
) -> NDArray[np.float64]:
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

    return binned_nz  # type: ignore[no-any-return]


class AngularVariableDepthMask:
    r"""Variable depth mask for tomographic bins.

    This class allows to create a mask with a different variable
    depth mask in the angular direction for each tomographic bin.

    Parameters
    ----------
    vardepth_map
        Map of variable which traces the depth per tomographic bin.
    n_bins
        Number of tomographic bins.
    zbins
        Shell redshift limits.
    """

    def __init__(
        self, vardepth_map: NDArray[np.float64], n_bins: int, zbins: NDArray[np.float64]
    ) -> None:
        self.vardepth_map = vardepth_map
        self.n_bins = n_bins
        self.zbins = zbins

    def check_index(self, index: tuple[int, int]) -> None:
        r"""Check the index for validity."""
        if not isinstance(index, tuple):
            raise TypeError("Index must be an tuple of two integers")

        if index[0] >= self.n_bins:
            raise ValueError("Leading index cannot exceed number of tomographic bins")

        if index[1] >= len(self.zbins):
            raise ValueError("Trailing index cannot exceed number of shells")

    def __getitem__(self, index: tuple[int, int]) -> NDArray[np.float64]:
        r"""Get the mask for the given index.

        Parameters
        ----------
        index
            Indices of the tomographic bin and shell pair.

        Returns
        -------
        mask
            Mask for the given index.

        Raises
        ------
        ValueError
            If the index is invalid.

        """
        self.check_index(index)

        return self.vardepth_map[index[0]]


class AngularLosVariableDepthMask(AngularVariableDepthMask):
    r"""Variable depth mask for tomographic bins.

    This class allows to create a mask with a different variable
    depth mask in both the angular and the line-of-sight directions
    for each tomographic bin.

    Parameters
    ----------
    vardepth_map
        Map of variable which traces the depth per tomographic bin.
        If vardepth_tomo_functions is not provided, the values are
        treated like a map of galaxy count ratios.
    n_bins
        Number of tomographic bins.
    zbins
        Shell redshift limits.
    ztomo
        Tomographic redshift bin limits.
    dndz
        Redshift distributions per tomographic bin.
    z
        Redshift domain of dndz.
    dndz_vardepth
        Redshift distribution affected by variable depth
        (n_bins x len(vardepth_values) x len(z)).
    vardepth_values
        Variable depth tracer domain/values of dndz_vardepth.
    vardepth_los_tracer
        Map of the variable depth tracer for line-of-sight direction. If
        provided, it is assumed to cover the same domain as vardepth_values.
    vardepth_tomo_functions
        List of functions which map the input vardepth_map to the ratio
        between the galaxy count due to the variable depth and the galaxy
        count without variable depth (for each tomographic bin). If
        provided, it is assumed that there is one vardepth_map which
        traces the variable depth for all tomographic bins.
    """

    def __init__( # noqa: PLR0913
        self,
        vardepth_map: NDArray[np.float64],
        n_bins: int,
        zbins: list[tuple[float, float]],
        ztomo: list[tuple[float, float]],
        dndz: NDArray[np.float64],
        z: NDArray[np.float64],
        dndz_vardepth: NDArray[np.float64],
        vardepth_values: NDArray[np.float64],
        vardepth_los_tracer: NDArray[np.float64] | None = None,
        vardepth_tomo_functions: list[Callable] | None = None,
    ) -> None: 
        super().__init__(vardepth_map, n_bins, zbins)
        self.ztomo = ztomo
        self.dndz = dndz
        self.z = z
        self.dndz_vardepth = dndz_vardepth
        self.vardepth_values = vardepth_values
        self.vardepth_los_tracer = vardepth_los_tracer
        self.vardepth_tomo_functions = vardepth_tomo_functions

        if vardepth_tomo_functions is not None:
            self.vardepth_map = np.atleast_2d(vardepth_map).reshape(1, -1)

    def get_los_fraction(self, index: tuple[int, int]) -> NDArray[np.float64]:
        r"""Gets the fraction of galaxies affected by variable depth in the
        line-of-sight direction for the given tomographic bin and shell.

        Parameters
        ----------
        index
            Indices of the tomographic bin and shell pair.

        Returns
        -------
        fraction_vardepth
            Fraction of galaxies affected by variable depth in the
            line-of-sight direction.
        """
        is_in_shell = (self.zbins[index[1]][0] < self.z) & (
            self.z <= self.zbins[index[1]][1]
        )

        n_gal_in_tomo_vardepth = np.trapezoid(
            self.dndz_vardepth[index[0]][:, is_in_shell], self.z[is_in_shell]
        )
        n_gal_in_tomo = np.trapezoid(
            self.dndz[index[0]][is_in_shell], self.z[is_in_shell]
        )
        return np.divide(
            n_gal_in_tomo_vardepth,
            n_gal_in_tomo,
            out=np.ones_like(n_gal_in_tomo_vardepth),
            where=n_gal_in_tomo != 0,
        )

    def __getitem__(self, index: tuple[int, int]) -> NDArray[np.float64]:
        r"""Get the mask for the given index.

        Parameters
        ----------
        index
            Indices of the tomographic bin and shell pair.

        Returns
        -------
        mask
            Mask for the given index.

        Raises
        ------
        ValueError
            If the index is invalid.

        """
        self.check_index(index)

        if self.vardepth_tomo_functions is None:
            angular_vardepth_map = angular_tracer_map = self.vardepth_map[index[0]]
        else:
            angular_vardepth_map = self.vardepth_tomo_functions[index[0]](
                self.vardepth_map[0]
            )
            angular_tracer_map = self.vardepth_map[0]

        los_fraction_vardepth = self.get_los_fraction(index)

        if self.vardepth_los_tracer is None:
            los_vardepth_map = np.interp(
                angular_tracer_map, self.vardepth_values, los_fraction_vardepth
            )
        else:
            los_vardepth_map = np.interp(
                self.vardepth_los_tracer, self.vardepth_values, los_fraction_vardepth
            )

        return np.multiply(angular_vardepth_map, los_vardepth_map)
