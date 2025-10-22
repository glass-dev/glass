"""Module for array utilities."""

from __future__ import annotations

import itertools
from functools import partial
from typing import TYPE_CHECKING

import numpy as np

import glass._array_api_utils as _utils

if TYPE_CHECKING:
    from typing import Unpack

    from numpy.typing import DTypeLike, NDArray

    from glass._array_api_utils import FloatArray


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
    xp = _utils.get_namespace(*arrays)

    arrays = tuple(xp.moveaxis(a, 0, -1) if a.ndim else a for a in arrays)
    arrays = xp.broadcast_arrays(*arrays)
    return tuple(xp.moveaxis(a, -1, 0) if a.ndim else a for a in arrays)


def broadcast_leading_axes(
    *args: tuple[
        float | NDArray[np.float64],
        int,
    ],
) -> tuple[
    tuple[int, ...],
    Unpack[tuple[NDArray[np.float64], ...]],
]:
    """
    Broadcast all but the last N axes.

    Parameters
    ----------
    args
        The arrays and the number of axes to keep.

    Returns
    -------
        The shape of the broadcast dimensions, and all input arrays
        with leading axes matching that shape.

    Examples
    --------
    Broadcast all dimensions of ``a``, all except the last dimension of
    ``b``, and all except the last two dimensions of ``c``.

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
    shapes, trails = [], []
    for a, n in args:
        s = np.shape(a)
        i = len(s) - n
        shapes.append(s[:i])
        trails.append(s[i:])
    dims = np.broadcast_shapes(*shapes)
    arrs = (
        np.broadcast_to(a, dims + t) for (a, _), t in zip(args, trails, strict=False)
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
    arrays_to_check = (xq, fq) if type(x) is float else (x, xq, fq)
    xp = _utils.get_namespace(*arrays_to_check)
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
    xp = _utils.get_namespace(*f, *tuple(itertools.chain(*ff)))
    uxpx = _utils.XPAdditions(xp)

    x: FloatArray
    x, _ = f
    for x_, _ in ff:
        x = uxpx.union1d(
            x[(x >= x_[0]) & (x <= x_[-1])],
            x_[(x_ >= x[0]) & (x_ <= x[-1])],
        )
    y = uxpx.interp(x, *f)
    for f_ in ff:
        y *= uxpx.interp(x, *f_)
    return uxpx.trapezoid(y, x, axis=axis)


def cumulative_trapezoid(
    f: NDArray[np.int_] | NDArray[np.float64],
    x: NDArray[np.int_] | NDArray[np.float64],
    dtype: DTypeLike | None = None,
    out: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """
    Cumulative trapezoidal rule along last axis.

    Parameters
    ----------
    f
        The function values.
    x
        The x-coordinates.
    dtype
        The output data type.
    out
        The output array.

    Returns
    -------
        The cumulative integral of the function.

    """
    if out is None:
        out = np.empty_like(f, dtype=dtype)

    np.cumsum((f[..., 1:] + f[..., :-1]) / 2 * np.diff(x), axis=-1, out=out[..., 1:])
    out[..., 0] = 0
    return out
