"""Module for spherical harmonic utilities."""

from __future__ import annotations

__lazy_modules__ = [
    "array_api_compat",
]

import typing
from typing import TYPE_CHECKING

import array_api_compat

import glass.healpix as hp

if TYPE_CHECKING:
    from glass._types import ComplexArray, FloatArray


def multalm(
    alm: ComplexArray,
    bl: FloatArray,
) -> ComplexArray:
    """
    Multiply alm by bl.

    The alm should be ordered by increasing ``m`` within each ``m`` block::

        [
            00, 10, 20, 30,
            11, 21, 31,
            22, 32,
            33
            ...
        ]

    Parameters
    ----------
    alm
        The alm to multiply.
    bl
        The bl to multiply.

    Returns
    -------
        The product of alm and bl.

    """
    xp = array_api_compat.array_namespace(alm, bl, use_compat=False)
    if bl.size == 0:
        return alm

    factors = xp.concat(tuple(bl[m:] for m in range(bl.size)))
    return alm * factors


@typing.overload
def inverse_transform(
    alm: ComplexArray,
    *,
    lmax: int,
    nside: int,
    pixwin: bool = False,
    pol: bool = True,
    spin: typing.Literal[0] = 0,
) -> FloatArray:
    # returns a real scalar map
    ...


@typing.overload
def inverse_transform(
    alm: ComplexArray,
    *,
    lmax: int,
    nside: int,
    pixwin: bool = False,
    pol: bool = True,
    spin: typing.Literal[1, 2],
) -> ComplexArray:
    # returns a complex map for spin 1 or 2
    ...


@typing.overload
def inverse_transform(
    alm: ComplexArray,
    *,
    lmax: int,
    nside: int,
    pixwin: bool = False,
    pol: bool = True,
    spin: int,
) -> FloatArray | ComplexArray:
    # returns a real or complex map depending on the spin
    ...


def inverse_transform(  # noqa: PLR0913
    alm: ComplexArray,
    *,
    lmax: int,
    nside: int,
    pixwin: bool = False,
    pol: bool = True,
    spin: int = 0,
) -> FloatArray | ComplexArray:
    """
    Compute the inverse spherical harmonic transform of alm.

    Convert HEALPix harmonic coefficients into a map without pixel-window
    smoothing. For non-zero spin, ``alm`` contains E modes and B modes are set
    to zero.

    Parameters
    ----------
    alm
        The spherical harmonic coefficients to transform.
    lmax
        The maximum multipole of the spherical harmonic transform.
    nside
        The nside parameter of the output map.
    pixwin
        Whether to apply the pixel window function.
    pol
        Whether to compute polarization.
    spin
        Spin of the output map. Zero produces a real scalar map; non-zero spin
        produces a complex map whose real and imaginary parts are the two
        spin components.

    Returns
    -------
        The map resulting from the inverse spherical harmonic transform.

    """
    xp = alm.__array_namespace__()

    if spin == 0:
        return hp.alm2map(alm, nside, lmax=lmax, pixwin=pixwin, pol=pol)

    maps = hp.alm2map_spin([alm, xp.zeros_like(alm)], nside, spin, lmax)
    return maps[0] + 1j * maps[1]


def transform(
    maps: FloatArray,
    *,
    lmax: int,
    nside: int,  # noqa: ARG001
    pol: bool = True,
    use_pixel_weights: bool = False,
) -> ComplexArray:
    """
    Compute the spherical harmonic transform of a map.

    Parameters
    ----------
    maps
        The real-space map to transform.
    lmax
        The maximum multipole of the spherical harmonic transform.
    nside
        The nside parameter of the input map.
    pol
        Whether to compute polarization.
    use_pixel_weights
        Whether to use pixel weights in the transform.

    Returns
    -------
        The spherical harmonic coefficients resulting from the transform.

    """
    return hp.map2alm(
        maps,
        lmax=lmax,
        pol=pol,
        use_pixel_weights=use_pixel_weights,
    )
