"""Module for spherical harmonic utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import healpy as hp

import array_api_compat

if TYPE_CHECKING:
    from glass._types import ComplexArray, FloatArray


def multalm(
    alm: ComplexArray,
    bl: FloatArray,
) -> ComplexArray:
    """
    Multiply alm by bl.

    The alm should be in GLASS order:

    [
        00,
        10, 11,
        20, 21, 22,
        30, 31, 32, 33,
        ...
    ]

    Parameters
    ----------
    alm
        The alm to multiply.
    bl
        The bl to multiply.
    inplace
        Whether to perform the operation in place.

    Returns
    -------
        The product of alm and bl.

    """
    xp = array_api_compat.array_namespace(alm, bl, use_compat=False)
    return alm * xp.repeat(bl, xp.arange(bl.size) + 1)


def transform(map: FloatArray) -> ComplexArray:
    """Transform between map and alm representations."""


def inverse_transform(
    alm: ComplexArray,
    *,
    nside: int,
    inplace: bool = False,
    lmax: int | None = None,
    polarised_input: bool = False,
) -> FloatArray:
    """
    Computes the inverse spherical harmonic transform.

    Parameters
    ----------
    alm
        The spherical harmonic coefficients.
    nside
        The nside of the output map if using HEALPix.
    inplace
        Whether to perform the operation in place.
    lmax
        The maximum multipole to use.
    polarised_input
        Whether the input alm represents polarised data.

    Returns
    -------
        The output map.
    """
    return hp.alm2map(
        alm,
        nside,
        inplace=inplace,
        lmax=lmax,
        pol=polarised_input,
    )
