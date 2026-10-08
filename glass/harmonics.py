"""Module for spherical harmonic utilities."""

from __future__ import annotations

__lazy_modules__ = [
    "array_api_compat",
]

import typing
from typing import TYPE_CHECKING

import array_api_compat

from glass.healpix import _alm2map, _alm2map_spin, _map2alm

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
    spin: int,
) -> FloatArray | ComplexArray:
    # returns a real or complex map depending on the spin
    ...


def inverse_transform(
    alm: ComplexArray,
    *,
    lmax: int,
    nside: int,
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
    spin
        Spin of the output map. Zero produces a real scalar map; non-zero spin
        produces a complex map whose real and imaginary parts are the two
        spin components.

    Returns
    -------
        The map resulting from the inverse spherical harmonic transform.

    """
    xp = alm.__array_namespace__()

    if xp.__name__ != "jax.numpy":
        if spin == 0:
            return hp._alm2map(alm, nside, lmax=lmax)

        maps = hp._alm2map_spin([alm, xp.zeros_like(alm)], nside, spin, lmax)
        return maps[0] + 1j * maps[1]

    import s2fft  # noqa: PLC0415
    import s2fft.sampling  # noqa: PLC0415

    bandlimit = lmax + 1
    flm = s2fft.sampling.reindex.flm_hp_to_2d_fast(alm, bandlimit)

    if spin:
        # Spin-weighted harmonics have no modes with ell < spin.
        flm = flm.at[: abs(spin)].set(0)

    # S2FFT implementation requires L >= 2 * nside
    if bandlimit < 2 * nside:
        padding = 2 * nside - bandlimit
        # pad missing modes with zeros
        flm = xp.pad(flm, ((0, padding), (padding, padding)))
        bandlimit = 2 * nside

    maps = s2fft.inverse(
        flm,
        bandlimit,
        method="jax",
        nside=nside,
        reality=spin == 0,
        sampling="healpix",
        spin=spin,
    )

    if spin:
        # Healpy's E-only convention has the opposite sign to S2FFT's.
        return -maps
    # S2FFT returns complex values, but the scalar map should be real-valued.
    # https://github.com/astro-informatics/s2fft/issues/411
    return xp.real(maps)


def transform(
    maps: FloatArray,
    *,
    lmax: int,
    nside: int,
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

    Returns
    -------
        The spherical harmonic coefficients resulting from the transform.

    """
    xp = maps.__array_namespace__()

    if xp.__name__ != "jax.numpy":
        # the only time GLASS used map2alm was with pixel weights
        return hp._map2alm(maps, lmax=lmax, use_pixel_weights=True)

    import s2fft  # noqa: PLC0415
    import s2fft.sampling  # noqa: PLC0415

    bandlimit = lmax + 1
    # the S2FFT implementation needs at least two modes per nside
    effective_bandlimit = max(bandlimit, 2 * nside)

    flm = s2fft.forward(
        maps,
        effective_bandlimit,
        # iterations are set here to match the healpy_jax method's behaviour,
        # without this there are large numerical errors in the transform
        iter=3,
        method="jax",
        nside=nside,
        reality=True,
        sampling="healpix",
    )

    if effective_bandlimit != bandlimit:
        # keep the requested ell rows and the corresponding centered m columns
        offset = effective_bandlimit - bandlimit
        flm = flm[:bandlimit, offset : offset + 2 * bandlimit - 1]

    return s2fft.sampling.reindex.flm_2d_to_hp_fast(flm, bandlimit)
