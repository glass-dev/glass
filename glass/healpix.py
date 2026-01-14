"""_summary_."""

from __future__ import annotations

from typing import TYPE_CHECKING

import healpix

if TYPE_CHECKING:
    from glass._types import IntArray, UnifiedGenerator


def ang2pix(
    nside: int,
    theta: float,
    phi: float,
    *,
    nest: bool = False,
    lonlat: bool = False,
):
    return healpix.ang2pix(nside, theta, phi, nest=nest, lonlat=lonlat)


def ang2vec(theta: float, phi: float, *, lonlat: bool = False):
    return healpix.ang2vec(theta, phi, lonlat=lonlat)


def npix2nside(npix: int) -> int:
    """_summary_."""
    return healpix.npix2nside(npix)


def nside2npix(nside: int) -> int:
    """_summary_."""
    return healpix.nside2npix(nside)


def randang(
    nside: int,
    ipix: IntArray,
    *,
    nest: bool = False,
    lonlat: bool = False,
    rng: UnifiedGenerator | None = None,
):
    """Sample random spherical coordinates from the given HEALPix pixels."""
    return healpix.randang(nside, ipix, nest=nest, lonlat=lonlat, rng=rng)
