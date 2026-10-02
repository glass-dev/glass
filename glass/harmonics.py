"""Module for spherical harmonic utilities."""

from __future__ import annotations

__lazy_modules__ = [
    "array_api_compat",
]

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


def inverse_transform(
    alm: ComplexArray,
    *,
    lmax: int,
    nside: int,
) -> FloatArray:
    """
    Compute the inverse spherical harmonic transform of alm.

    Convert scalar HEALPix harmonic coefficients into a map without pixel-window
    smoothing. Use the S2FFT healpy wrapper for JAX arrays.

    Parameters
    ----------
    alm
        The spherical harmonic coefficients to transform.
    lmax
        The maximum multipole of the spherical harmonic transform.
    nside
        The nside parameter of the output map.

    Returns
    -------
        The real-space map resulting from the inverse spherical harmonic transform.

    """
    xp = alm.__array_namespace__()

    if xp.__name__ != "jax.numpy":
        return hp.alm2map(alm, nside, lmax=lmax)

    import s2fft  # noqa: PLC0415
    import s2fft.sampling  # noqa: PLC0415

    bandlimit = lmax + 1
    flm = s2fft.sampling.reindex.flm_hp_to_2d_fast(alm, bandlimit)
    return s2fft.inverse(
        flm,
        bandlimit,
        method="jax_healpy",
        nside=nside,
        sampling="healpix",
    )


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
        return hp.map2alm(maps, lmax=lmax)

    import s2fft  # noqa: PLC0415
    import s2fft.sampling  # noqa: PLC0415

    bandlimit = lmax + 1
    flm = s2fft.forward(
        maps,
        bandlimit,
        method="jax_healpy",
        nside=nside,
        sampling="healpix",
    )
    return s2fft.sampling.reindex.flm_2d_to_hp_fast(flm, bandlimit)
