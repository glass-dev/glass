"""Module for spherical harmonic utilities."""

from __future__ import annotations

from collections.abc import Sequence
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

    The alm should be in GLASS order::

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

    Returns
    -------
        The product of alm and bl.

    """
    xp = array_api_compat.array_namespace(alm, bl, use_compat=False)
    return alm * xp.repeat(bl, xp.arange(bl.size) + 1)


def transform(
    maps: FloatArray,
    *,
    lmax: int | None = None,
    polarised_input: bool = True,
    use_pixel_weights: bool = False,
) -> ComplexArray:
    """
    Computes the spherical harmonic transform.

    Parameters
    ----------
    maps
        The input map(s).
    lmax
        The maximum multipole to use.
    polarised_input
        Whether the input maps represent polarised data.
    use_pixel_weights
        Whether to use pixel weights in the transform.

    Returns
    -------
        The spherical harmonic coefficients.

    """
    return hp.map2alm(
        maps,
        lmax=lmax,
        pol=polarised_input,
        use_pixel_weights=use_pixel_weights,
    )


def inverse_transform(  # noqa: PLR0913
    alm: ComplexArray | Sequence[ComplexArray],
    *,
    nside: int,
    inplace: bool = False,
    lmax: int | None = None,
    polarised_input: bool = True,
    spin: int | None = None,
) -> FloatArray | list[FloatArray]:
    """
    Computes the inverse spherical harmonic transform.

    Parameters
    ----------
    alm
        The spherical harmonic coefficients. If computing a spin-s map (spin is
        not None), this should be a sequence of two coefficient arrays:
        [alm, blm]. Otherwise, it is the single alm array.
    nside
        The nside of the output map if using HEALPix.
    inplace
        Whether to perform the operation in place (only applicable when
        spin=None).
    lmax
        The maximum multipole to use.
    polarised_input
        Whether the input alm represents polarised data (only applicable when
        spin=None).
    spin
        The spin s of the field being transformed.

    Returns
    -------
        The output map(s).

    """
    if spin is None:
        # alm is the single alm array or a sequence/array of 3 for polarised data
        return hp.alm2map(
            alm,
            nside,
            inplace=inplace,
            lmax=lmax,
            pol=polarised_input,
        )

    # check if alm is of the correct form for spin transforms
    if not (hasattr(alm, "shape") and alm.shape[0] == 2) and not (
        isinstance(alm, Sequence) and len(alm) == 2
    ):
        # alm must be a sequence [alm, blm]
        msg = "for spin transforms, alm must be of the form [alm, blm]"
        raise ValueError(msg)
    # does not accept "inplace" or "polarised_input" arguments
    return hp.alm2map_spin(alm, nside, spin, lmax)
