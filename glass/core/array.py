# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''module for array utilities'''

import numpy as np
from functools import partial


def broadcast_leading_axes(*args):
    '''Broadcast all but the last N axes.

    Returns the shape of the broadcast dimensions, and all input arrays
    with leading axes matching that shape.

    Example
    -------
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

    '''

    shapes, trails = [], []
    for a, n in args:
        s = np.shape(a)
        i = len(s) - n
        shapes.append(s[:i])
        trails.append(s[i:])
    dims = np.broadcast_shapes(*shapes)
    arrs = (np.broadcast_to(a, dims + t) for (a, _), t in zip(args, trails))
    return (dims, *arrs)


def ndinterp(x, xp, fp, axis=-1, left=None, right=None, period=None):
    '''interpolate multi-dimensional array over axis'''
    return np.apply_along_axis(partial(np.interp, x, xp), axis, fp,
                               left=left, right=right, period=period)


def trapz_product(f, *ff, axis=-1):
    '''trapezoidal rule for a product of functions'''
    x, _ = f
    for x_, _ in ff:
        x = np.union1d(x, x_[(x_ > x[0]) & (x_ < x[-1])])
    y = np.interp(x, *f)
    for f_ in ff:
        y *= np.interp(x, *f_)
    return np.trapz(y, x, axis=axis)


def cumtrapz(f, x, dtype=None, out=None):
    '''cumulative trapezoidal rule along last axis'''

    if out is None:
        out = np.empty_like(f, dtype=dtype)

    np.cumsum((f[..., 1:] + f[..., :-1])/2*np.diff(x), axis=-1, out=out[..., 1:])
    out[..., 0] = 0
    return out
