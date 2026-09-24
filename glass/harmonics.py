"""Module for spherical harmonic utilities."""

from __future__ import annotations

__lazy_modules__ = [
    "array_api_compat",
]

from typing import TYPE_CHECKING

import array_api_compat

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
