"""Module for array utilities."""

from __future__ import annotations

import typing
from functools import partial

import numpy as np
import numpy.typing as npt


def broadcast_first(
    *arrays: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], ...]:
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
    arrays = tuple(np.moveaxis(a, 0, -1) if np.ndim(a) else a for a in arrays)
    arrays = np.broadcast_arrays(*arrays)
    return tuple(np.moveaxis(a, -1, 0) if np.ndim(a) else a for a in arrays)


def broadcast_leading_axes(
    *args: tuple[
        float | npt.NDArray[np.float64],
        int,
    ],
) -> tuple[
    tuple[int, ...],
    typing.Unpack[tuple[npt.NDArray[np.float64], ...]],
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
    arrs = (np.broadcast_to(a, dims + t) for (a, _), t in zip(args, trails))
    return (dims, *arrs)


def ndinterp(  # noqa: PLR0913
    x: float | npt.NDArray[np.float64],
    xp: npt.NDArray[np.float64],
    fp: npt.NDArray[np.float64],
    axis: int = -1,
    left: float | None = None,
    right: float | None = None,
    period: float | None = None,
) -> npt.NDArray[np.float64]:
    """
    Interpolate multi-dimensional array over axis.

    Parameters
    ----------
    x
        The x-coordinates.
    xp
        The x-coordinates of the data points.
    fp
        The function values corresponding to the x-coordinates in *xp*.
    axis
        The axis to interpolate over.
    left
        The value to return for x < xp[0].
    right
        The value to return for x > xp[-1].
    period
        The period of the function, used for interpolating periodic data.

    Returns
    -------
        The interpolated array.

    """
    return np.apply_along_axis(
        partial(np.interp, x, xp),
        axis,
        fp,
        left=left,
        right=right,
        period=period,
    )


def trapz_product(
    f: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
    *ff: tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ],
    axis: int = -1,
) -> npt.NDArray[np.float64]:
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
    x: npt.NDArray[np.float64]
    x, _ = f
    for x_, _ in ff:
        x = np.union1d(
            x[(x >= x_[0]) & (x <= x_[-1])],
            x_[(x_ >= x[0]) & (x_ <= x[-1])],
        )
    y = np.interp(x, *f)
    for f_ in ff:
        y *= np.interp(x, *f_)
    return np.trapz(  # type: ignore[attr-defined, no-any-return]
        y,
        x,
        axis=axis,
    )


def cumtrapz(
    f: npt.NDArray[np.int_] | npt.NDArray[np.float64],
    x: npt.NDArray[np.int_] | npt.NDArray[np.float64],
    dtype: npt.DTypeLike | None = None,
    out: npt.NDArray[np.float64] | None = None,
) -> npt.NDArray[np.float64]:
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
