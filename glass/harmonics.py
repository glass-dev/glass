"""Module for spherical harmonic utilities."""

from __future__ import annotations

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
    return alm * xp.repeat(bl, xp.arange(bl.shape[0]) + 1)
