"""Module for array utilities."""

from __future__ import annotations

import itertools
from functools import partial
from typing import TYPE_CHECKING

import array_api_compat
import array_api_extra as xpx

import glass._array_api_utils as _utils

if TYPE_CHECKING:
    from types import ModuleType

    from typing_extensions import Unpack

    from glass._types import AnyArray, FloatArray, IntArray


def broadcast_first(
    *arrays: FloatArray,
) -> tuple[FloatArray, ...]:
    """
    Broadcast arrays, treating the first axis as common.

    Parameters
    ----------
    arrays
        The arrays to broadcast.

    Returns
    -------
        The broadcasted arrays.

    """
    xp = array_api_compat.array_namespace(*arrays, use_compat=False)

    arrays = tuple(xp.moveaxis(a, 0, -1) if a.ndim else a for a in arrays)
    arrays = xp.broadcast_arrays(*arrays)
    return tuple(xp.moveaxis(a, -1, 0) if a.ndim else a for a in arrays)


def broadcast_leading_axes(
    *args: tuple[
        float | FloatArray,
        int,
    ],
    xp: ModuleType | None = None,
) -> tuple[
    tuple[int, ...],
    Unpack[tuple[FloatArray, ...]],
]:
    """
    Broadcast all but the last N axes.

    Parameters
    ----------
    args
        The arrays and the number of axes to keep.
    xp
        The array library backend to use for array operations.

    Returns
    -------
        The shape of the broadcast dimensions, and all input arrays
        with leading axes matching that shape.

    Examples
    --------
    Broadcast all dimensions of ``a``, all except the last dimension of
    ``b``, and all except the last two dimensions of ``c``.

    >>> import numpy as np
    >>> a = 0
    >>> b = np.zeros((4, 10))
    >>> c = np.zeros((3, 1, 5, 6))
    >>> dims, a, b, c = broadcast_leading_axes((a, 0), (b, 1), (c, 2))
    >>> dims
    (3, 4)
    >>> a.shape
    (3, 4)
    >>> b.shape
    (3, 4, 10)
    >>> c.shape
    (3, 4, 5, 6)

    """
    if xp is None:
        xp = array_api_compat.array_namespace(
            *[arg[0] for arg in args],
            use_compat=False,
        )

    shapes, trails = [], []
    for a, n in args:
        a_arr = xp.asarray(a)
        s = a_arr.shape
        i = len(s) - n
        shapes.append(s[:i])
        trails.append(s[i:])
    dims = xpx.broadcast_shapes(*shapes)
    arrs = (
        xp.broadcast_to(xp.asarray(a), dims + t)
        for (a, _), t in zip(args, trails, strict=False)
    )
    return (dims, *arrs)


def ndinterp(  # noqa: PLR0913
    x: float | FloatArray,
    xq: FloatArray,
    fq: FloatArray,
    axis: int = -1,
    left: float | None = None,
    right: float | None = None,
    period: float | None = None,
) -> FloatArray:
    """
    Interpolate multi-dimensional array over axis.

    Parameters
    ----------
    x
        The x-coordinates.
    xq
        The x-coordinates of the data points.
    fq
        The function values corresponding to the x-coordinates in *xq*.
    axis
        The axis to interpolate over.
    left
        The value to return for x < xq[0].
    right
        The value to return for x > xq[-1].
    period
        The period of the function, used for interpolating periodic data.

    Returns
    -------
        The interpolated array.

    """
    xp = array_api_compat.array_namespace(x, xq, fq, use_compat=False)
    uxpx = _utils.XPAdditions(xp)

    return uxpx.apply_along_axis(
        partial(uxpx.interp, x, xq),
        axis,
        fq,
        left=left,
        right=right,
        period=period,
    )


def trapezoid_product(
    f: tuple[FloatArray, FloatArray],
    *ff: tuple[FloatArray, FloatArray],
    axis: int = -1,
) -> float | FloatArray:
    """
    Trapezoidal rule for a product of functions.

    Parameters
    ----------
    f
        The first function.
    ff
        The other functions.
    axis
        The axis along which to integrate.

    Returns
    -------
        The integral of the product of the functions.

    """
    # Flatten ff into a 1D tuple of all ff inputs and then expand to get the namespace
    xp = array_api_compat.array_namespace(
        *f,
        *tuple(itertools.chain(*ff)),
        use_compat=False,
    )
    uxpx = _utils.XPAdditions(xp)

    x: FloatArray
    x, _ = f
    for x_, _ in ff:
        x = xpx.union1d(
            x[(x >= x_[0]) & (x <= x_[-1])],
            x_[(x_ >= x[0]) & (x_ <= x[-1])],
        )
    y = uxpx.interp(x, *f)
    for f_ in ff:
        y *= uxpx.interp(x, *f_)
    return uxpx.trapezoid(y, x, axis=axis)


def cumulative_trapezoid(
    f: IntArray | FloatArray,
    x: IntArray | FloatArray,
) -> AnyArray:
    """
    Cumulative trapezoidal rule along last axis.

    Parameters
    ----------
    f
        The function values.
    x
        The x-coordinates.

    Returns
    -------
        The cumulative integral of the function.

    """
    xp = array_api_compat.array_namespace(f, x, use_compat=False)

    f = xp.asarray(f, dtype=xp.float64)
    x = xp.asarray(x, dtype=xp.float64)

    # Compute the cumulative trapezoid without mutating any arrays
    return xp.cumulative_sum(
        (f[..., 1:] + f[..., :-1]) * 0.5 * xp.diff(x),
        axis=-1,
        include_initial=True,
    )
