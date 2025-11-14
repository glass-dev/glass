"""Module for spherical harmonic utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import array_api_compat

if TYPE_CHECKING:
    from glass._types import ComplexArray, FloatArray


def multalm(
    alm: ComplexArray,
    bl: FloatArray,
    *,
    inplace: bool = False,
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

    n = bl.size
    # Ideally would be xp.asanyarray but this does not yet exist. The key difference
    # between the two in numpy is that asanyarray maintains subclasses of NDArray
    # whereas asarray will return the base class NDArray. Currently, we don't seem
    # to pass a subclass of NDArray so this, so it might be okay
    out = xp.asarray(alm) if inplace else xp.asarray(alm, copy=True)
    for ell in range(n):
        out[ell * (ell + 1) // 2 : (ell + 1) * (ell + 2) // 2] *= bl[ell]

    return out
