"""Wrapper for HEALPix operations to be Array API compatible."""

from __future__ import annotations

__lazy_modules__ = [
    "array_api_compat",
    "healpix",
    "healpy",
    "numpy",
]

import os
import pathlib
from typing import TYPE_CHECKING

import healpix
import healpy
import numpy as np

import glass._array_api_utils as _utils
import glass.rng
from glass._array_api_utils import numpy_fallback

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import ModuleType

    from glass._types import ComplexArray, DTypeLike, FloatArray, IntArray


def _get_healpy_datapath() -> str | None:
    healpy_datapath = os.environ.get("HEALPY_DATAPATH")
    if healpy_datapath is not None and not pathlib.Path(healpy_datapath).is_dir():
        raise ValueError(f"Healpy datapath not found at '{healpy_datapath}'")
    return healpy_datapath


@numpy_fallback
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
    return healpy.alm2map(
        alms,
        nside,
        inplace=inplace,
        lmax=lmax,
        pixwin=pixwin,
        pol=pol,
    )


@numpy_fallback
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
    inputs = [np.asarray(alm) for alm in alms]
    outputs = healpy.alm2map_spin(inputs, nside, spin, lmax)
    return list(outputs)


@numpy_fallback
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
        The function (at l=0..fl.shape[0]-1) by which alm must be multiplied.
    inplace
        If True, modify the given alm, otherwise make a copy before multiplying.

    Returns
    -------
        The modified alm, either a new array or a reference to input alm.

    """
    return healpy.almxfl(
        np.asarray(alm),
        np.asarray(fl),
        inplace=inplace,
    )


@numpy_fallback
def ang2pix(
    nside: int,
    theta: float | FloatArray,
    phi: float | FloatArray,
    *,
    lonlat: bool = False,
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
    return healpix.ang2pix(
        nside,
        np.asarray(theta),
        np.asarray(phi),
        lonlat=lonlat,
    )


@numpy_fallback
def ang2vec(
    theta: float | FloatArray,
    phi: float | FloatArray,
    *,
    lonlat: bool = False,
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
    x, y, z = healpix.ang2vec(
        np.asarray(theta),
        np.asarray(phi),
        lonlat=lonlat,
    )
    return x, y, z


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


@numpy_fallback
def map2alm(
    maps: FloatArray | Sequence[FloatArray],
    *,
    lmax: int | None = None,
    pol: bool = True,
    use_pixel_weights: bool = False,
) -> FloatArray:
    """
    Computes the alm of a HEALPix map. The input maps must all be in ring ordering.

    If you are running in an offline environment, you must provide a datapath to local
    healpy datafiles. To download these files:

        git clone --depth 1 https://github.com/healpy/healpy-data
        cd healpy-data
        bash download_weights_8192.sh

    and set datapath to the root of the repository. If this method is used, the only
    supported values of nside are 2^n for n in the range 5-13.

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
    return healpy.map2alm(
        maps,
        datapath=_get_healpy_datapath(),
        lmax=lmax,
        pol=pol,
        use_pixel_weights=use_pixel_weights,
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

    If you are running in an offline environment, you must provide a datapath to local
    healpy datafiles. To download these files:

        git clone --depth 1 https://github.com/healpy/healpy-data
        cd healpy-data
        bash download_weights_8192.sh

    and set datapath to the root of the repository. If this method is used, the only
    supported values of nside are 2^n for n in the range 5-13.

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

    output = healpy.pixwin(nside, datapath=_get_healpy_datapath(), lmax=lmax, pol=pol)
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


@numpy_fallback
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
    theta, phi = healpix.randang(
        nside,
        ipix,
        lonlat=lonlat,
        rng=glass.rng.default_rng(xp=np),
    )
    return theta, phi


class Rotator:
    """Rotation operator, including astronomical coordinate systems."""

    def __init__(
        self,
        *,
        coord: Sequence[str] | None = None,
    ) -> None:
        """
        Create a rotator with given parameters.

        Parameters
        ----------
        coord
            A string or a tuple of 1 or 2 strings or a sequence of tuple.
        xp
            The array library backend to use for array operations.

        """
        self.coord = coord

    @numpy_fallback
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
        return healpy.Rotator(coord=self.coord).rotate_map_pixel(m)
