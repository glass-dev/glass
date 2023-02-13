# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''module for mathematical utilities'''

import numpy as np
from functools import partial

# constants
DEGREE2_SPHERE = 60**4//100/np.pi
ARCMIN2_SPHERE = 60**6//100/np.pi
ARCSEC2_SPHERE = 60**8//100/np.pi


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


def cumtrapz(f, x, out=None):
    '''cumulative trapezoidal rule along last axis'''

    if out is None:
        out = np.empty_like(f)

    np.cumsum((f[..., 1:] + f[..., :-1])/2*np.diff(x), axis=-1, out=out[..., 1:])
    out[..., 0] = 0
    return out
