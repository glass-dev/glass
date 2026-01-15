"""_summary_.."""

from __future__ import annotations

from typing import TYPE_CHECKING

import healpy  # noqa: F401
import numpy as np

import healpix

if TYPE_CHECKING:
    from glass._types import FloatArray, IntArray, UnifiedGenerator


def alm2map():
    pass


def alm2map_spin():
    pass


def almxfl():
    pass


def ang2pix(
    nside: int,
    theta: float,
    phi: float,
    *,
    nest: bool = False,
    lonlat: bool = False,
) -> IntArray:
    """Converts the angle to HEALPix pixel numbers.

    Parameters
    ----------
    nside
        The HEALPix nside parameter of the map.
    theta
        Angular coordinates of a point on the sphere.
    phi
        Angular coordinates of a point on the sphere.
    nest
        If True return the map in NEST ordering.
    lonlat
        If True, automatically adjust latitudes to be within [-90, 90] range.

    Returns
    -------
        The HEALPix pixel numbers.

    """
    return healpix.ang2pix(nside, theta, phi, nest=nest, lonlat=lonlat)


def ang2vec(
    theta: float, phi: float, *, lonlat: bool = False
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Convert angles to 3D position vector.

    Parameters
    ----------
    theta
        Angular coordinates of a point on the sphere.
    phi
        Angular coordinates of a point on the sphere.
    lonlat
        If True, automatically adjust latitudes to be within [-90, 90] range.

    Returns
    -------
        A normalised 3-vector pointing in the same direction as ``ang``.

    """
    return healpix.ang2vec(theta, phi, lonlat=lonlat)


def get_nside():
    pass


def map2alm():
    pass


def npix2nside(npix: int) -> int:
    """Give the nside parameter for the given number of pixels.

    Parameters
    ----------
    npix
        The number of pixels.

    Returns
    -------
        The HEALPix nside parameter of the map.

    """
    return healpix.npix2nside(npix)


def nside2npix(nside: int) -> int:
    """Give the number of pixels for the given nside.

    Parameters
    ----------
    nside
        The HEALPix nside parameter of the map.

    Returns
    -------
        The number of pixels.

    """
    return healpix.nside2npix(nside)


def pixwin():
    pass


def query_strip():
    pass


def randang(
    nside: int,
    ipix: IntArray,
    *,
    nest: bool = False,
    lonlat: bool = False,
    rng: UnifiedGenerator | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Sample random spherical coordinates from the given HEALPix pixels.

    Parameters
    ----------
    nside
        The HEALPix nside parameter of the map.
    ipix
        HEALPix pixel number.
    nest
        If True return the map in NEST ordering.
    lonlat
        If True, automatically adjust latitudes to be within [-90, 90] range.
    rng
        Random number generator. If not given, a default RNG is used.

    Returns
    -------
        A tuple ``theta, phi`` of mathematical coordinates.

    """
    xp = ipix.__array_namespace__()
    return xp.asarray(
        healpix.randang(
            nside,
            np.asarray(ipix),
            nest=nest,
            lonlat=lonlat,
            rng=rng,
        )
    )


class Rotator:
    def rotate_map_pixel(self):
        pass
