"""Wrapper for HEALPix operations to be Array API compatible."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import healpix
import healpy
import numpy as np

import array_api_compat

import glass._array_api_utils as _utils
from glass import _rng

if TYPE_CHECKING:
    from types import ModuleType

    from glass._types import ComplexArray, DTypeLike, FloatArray, IntArray


def alm2map(  # noqa: PLR0913
    alms: ComplexArray | Sequence[ComplexArray],
    nside: int,
    *,
    inplace: bool = False,
    lmax: int | None = None,
    pixwin: bool = False,
    pol: bool = True,
) -> FloatArray:
    """
    Computes a HEALPix map given the alm.

    Parameters
    ----------
    alms
        A complex array or a sequence of complex arrays.
    nside
        The nside of the output map.
    inplace
        If True, input alms may be modified by pixel window function and beam smoothing.
    lmax
        Explicitly define lmax.
    pixwin
        Smooth the alm using the pixel window functions.
    pol
        If True, assumes input alms are TEB.

    Returns
    -------
        A HEALPix map in RING scheme at nside or a list of T,Q,U maps.

    """
    xp = (
        array_api_compat.get_namespace(*alms, use_compat=False)
        if isinstance(alms, Sequence)
        else alms.__array_namespace__()
    )

    inputs = (
        [np.asarray(alm) for alm in alms]
        if isinstance(alms, Sequence)
        else np.asarray(alms)
    )
    return xp.asarray(
        healpy.alm2map(
            inputs,
            nside,
            inplace=inplace,
            lmax=lmax,
            pixwin=pixwin,
            pol=pol,
        ),
    )


def alm2map_spin(
    alms: Sequence[FloatArray],
    nside: int,
    spin: int,
    lmax: int,
) -> list[FloatArray]:
    """
    Computes maps from a set of 2 spinned alm.

    Parameters
    ----------
    alms
        List of 2 alms.
    nside
        Requested nside of the output map.
    spin
        Spin of the alms.
    lmax
        Maximum l of the power spectrum.

    Returns
    -------
        List of 2 out maps in RING scheme as arrays.

    """
    xp = array_api_compat.get_namespace(*alms, use_compat=False)

    inputs = [np.asarray(alm) for alm in alms]
    outputs = healpy.alm2map_spin(inputs, nside, spin, lmax)
    return [xp.asarray(out) for out in outputs]


def almxfl(
    alm: FloatArray,
    fl: FloatArray,
    *,
    inplace: bool = False,
) -> FloatArray:
    """
    Multiply alm by a function of l. The function is assumed to be zero where
    not defined.

    Parameters
    ----------
    alm
        The alm to multiply.
    fl
        The function (at l=0..fl.size-1) by which alm must be multiplied.
    inplace
        If True, modify the given alm, otherwise make a copy before multiplying.

    Returns
    -------
        The modified alm, either a new array or a reference to input alm.

    """
    xp = array_api_compat.get_namespace(alm, fl, use_compat=False)

    return xp.asarray(
        healpy.almxfl(
            np.asarray(alm),
            np.asarray(fl),
            inplace=inplace,
        ),
    )


def ang2pix(
    nside: int,
    theta: float | FloatArray,
    phi: float | FloatArray,
    *,
    lonlat: bool = False,
    xp: ModuleType | None = None,
) -> IntArray:
    """
    Converts the angle to HEALPix pixel numbers.

    Parameters
    ----------
    nside
        The HEALPix nside parameter of the map.
    theta
        Angular coordinates of a point on the sphere.
    phi
        Angular coordinates of a point on the sphere.
    lonlat
        If True, automatically adjust latitudes to be within [-90, 90] range.
    xp
        The array library backend to use for array operations.

    Returns
    -------
        The HEALPix pixel numbers.

    """
    xp = _utils.default_xp() if xp is None else xp

    return xp.asarray(
        healpix.ang2pix(
            nside,
            np.asarray(theta),
            np.asarray(phi),
            lonlat=lonlat,
        ),
    )


def ang2vec(
    theta: float | FloatArray,
    phi: float | FloatArray,
    *,
    lonlat: bool = False,
    xp: ModuleType | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """
    Convert angles to 3D position vector.

    Parameters
    ----------
    theta
        Angular coordinates of a point on the sphere.
    phi
        Angular coordinates of a point on the sphere.
    lonlat
        If True, automatically adjust latitudes to be within [-90, 90] range.
    xp
        The array library backend to use for array operations.

    Returns
    -------
        A normalised 3-vector pointing in the same direction as ``ang``.

    """
    xp = _utils.default_xp() if xp is None else xp

    x, y, z = healpix.ang2vec(
        np.asarray(theta),
        np.asarray(phi),
        lonlat=lonlat,
    )
    return xp.asarray(x), xp.asarray(y), xp.asarray(z)


def get_nside(m: FloatArray) -> int:
    """
    Return the nside of the given map.

    Parameters
    ----------
    m
        The map to get the nside from.

    Returns
    -------
        The HEALPix nside parameter of the map.

    """
    return int(healpy.get_nside(np.asarray(m)))


def map2alm(
    maps: FloatArray | Sequence[FloatArray],
    *,
    lmax: int | None = None,
    pol: bool = True,
    use_pixel_weights: bool = False,
) -> FloatArray:
    """
    Computes the alm of a HEALPix map. The input maps must all be in ring ordering.

    Parameters
    ----------
    maps
        The input map or a list of n input maps. Must be in ring ordering.
    lmax
        Maximum l of the power spectrum.
    pol
        If True, assumes input maps are TQU.
    use_pixel_weights
        If True, use pixel by pixel weighting, healpy will automatically
        download the weights, if needed.

    Returns
    -------
        alm or a tuple of 3 alm (almT, almE, almB) if polarized input.

    """
    xp = (
        array_api_compat.get_namespace(*maps, use_compat=False)
        if isinstance(maps, Sequence)
        else maps.__array_namespace__()
    )

    inputs = (
        [np.asarray(m) for m in maps]
        if isinstance(maps, Sequence)
        else np.asarray(maps)
    )
    return xp.asarray(
        healpy.map2alm(inputs, lmax=lmax, pol=pol, use_pixel_weights=use_pixel_weights),
    )


def npix2nside(npix: int) -> int:
    """
    Give the nside parameter for the given number of pixels.

    Parameters
    ----------
    npix
        The number of pixels.

    Returns
    -------
        The HEALPix nside parameter of the map.

    """
    return int(healpix.npix2nside(npix))


def nside2npix(nside: int) -> int:
    """
    Give the number of pixels for the given nside.

    Parameters
    ----------
    nside
        The HEALPix nside parameter of the map.

    Returns
    -------
        The number of pixels.

    """
    return int(healpix.nside2npix(nside))


def pixwin(
    nside: int,
    *,
    lmax: int | None = None,
    pol: bool = False,
    xp: ModuleType | None = None,
) -> FloatArray | tuple[FloatArray, ...]:
    """
    Return the pixel window function for the given nside.

    Parameters
    ----------
    nside
        The nside for which to return the pixel window function.
    lmax
        If True, return also the polar pixel window.
    pol
        Maximum l of the power spectrum.
    xp
        The array library backend to use for array operations.

    Returns
    -------
        The temperature pixel window function.

    """
    xp = _utils.default_xp() if xp is None else xp

    output = healpy.pixwin(nside, lmax=lmax, pol=pol)
    return (
        tuple(xp.asarray(out, dtype=xp.float64) for out in output)
        if pol
        else xp.asarray(output, dtype=xp.float64)
    )


def query_strip(
    nside: int,
    thetas: tuple[float, float],
    *,
    dtype: DTypeLike | None = None,
    xp: ModuleType | None = None,
) -> IntArray:
    """
    Computes a mask of the pixels whose centers lie within the colatitude range
    defined by thetas.

    Parameters
    ----------
    nside
        The nside of the HEALPix map.
    thetas
        Colatitudes in radians.
    dtype
        Desired data-type for the output array.
    xp
        The array library backend to use for array operations.

    Returns
    -------
        The mask of the pixels which lie within the given strip.

    """
    xp = _utils.default_xp() if xp is None else xp

    output = np.zeros(nside2npix(nside))
    indices = healpy.query_strip(nside, *thetas)
    output[indices] = 1

    # masks are usually integers, but this allows the user to override
    if dtype is None:
        return xp.asarray(output, dtype=xp.int64)
    return xp.asarray(output, dtype=dtype)


def randang(
    nside: int,
    ipix: IntArray,
    *,
    lonlat: bool = False,
) -> tuple[FloatArray, FloatArray]:
    """
    Sample random spherical coordinates from the given HEALPix pixels.

    ``rng`` is no longer a parameter as we must use the NumPy backend otherwise
    we run into a ``operand array with iterator write flag set is read-only``
    error coming from ``_chp.ring2ang_uv(nside, ipix, u, v, u, v)``.

    Parameters
    ----------
    nside
        The HEALPix nside parameter of the map.
    ipix
        HEALPix pixel number.
    lonlat
        If True, automatically adjust latitudes to be within [-90, 90] range.

    Returns
    -------
        A tuple ``theta, phi`` of mathematical coordinates.

    """
    xp = ipix.__array_namespace__()

    theta, phi = healpix.randang(
        nside,
        np.asarray(ipix),
        lonlat=lonlat,
        rng=_rng.rng_dispatcher(xp=np),
    )
    return xp.asarray(theta), xp.asarray(phi)


class Rotator:
    """Rotation operator, including astronomical coordinate systems."""

    def __init__(
        self,
        *,
        coord: Sequence[str] | None = None,
    ) -> None:
        """Create a rotator with given parameters.

        Parameters
        ----------
        coord
            A string or a tuple of 1 or 2 strings or a sequence of tuple.
        xp
            The array library backend to use for array operations.

        """
        self.coord = coord

    def rotate_map_pixel(self, m: FloatArray) -> FloatArray:
        """
        Rotate a HEALPix map to a new reference frame in pixel space.

        Parameters
        ----------
        m
            Input map, 1 map is considered I, 2 maps:[Q,U], 3 maps:[I,Q,U].

        Returns
        -------
            Map in the new reference frame

        """
        xp = m.__array_namespace__()

        return xp.asarray(
            healpy.Rotator(coord=self.coord).rotate_map_pixel(np.asarray(m)),
        )
