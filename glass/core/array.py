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
    _summary_.

    Parameters
    ----------
    arrays
        _description_

    Returns
    -------
        _description_

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
    _summary_.

    Parameters
    ----------
    args
        _description_

    Returns
    -------
        _description_

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
    _summary_.

    Parameters
    ----------
    x
        _description_
    xp
        _description_
    fp
        _description_
    axis
        _description_
    left
        _description_
    right
        _description_
    period
        _description_

    Returns
    -------
        _description_

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
    _summary_.

    Parameters
    ----------
    f
        _description_
    ff
        _description_
    axis
        _description_

    Returns
    -------
        _description_

    """
    x, _ = f
    for x_, _ in ff:
        x = np.union1d(
            x[(x >= x_[0]) & (x <= x_[-1])],
            x_[(x_ >= x[0]) & (x_ <= x[-1])],
        )
    y = np.interp(x, *f)
    for f_ in ff:
        y *= np.interp(x, *f_)
    return np.trapz(  # type: ignore[attr-defined]
        y,
        x,
        axis=axis,
    )


def cumtrapz(
    f: npt.NDArray[np.int_] | npt.NDArray[np.float64],
    x: npt.NDArray[np.int_] | npt.NDArray[np.float64],
    dtype: type | None = None,
    out: npt.NDArray[np.float64] | None = None,
) -> npt.NDArray[np.float64]:
    """
    Cumulative trapezoidal rule along last axis.

    Parameters
    ----------
    f
        _description_
    x
        _description_
    dtype
        _description_
    out
        _description_

    Returns
    -------
        _description_

    """
    if out is None:
        out = np.empty_like(f, dtype=dtype)

    np.cumsum((f[..., 1:] + f[..., :-1]) / 2 * np.diff(x), axis=-1, out=out[..., 1:])
    out[..., 0] = 0
    return out
