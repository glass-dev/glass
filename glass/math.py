# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''module for mathematical utilities'''

import numpy as np

# constants
DEGREE2_SPHERE = 60**4//100/np.pi
ARCMIN2_SPHERE = 60**6//100/np.pi
ARCSEC2_SPHERE = 60**8//100/np.pi


def restrict_interval(f, x, xmin, xmax):
    '''restrict a function to an interval using interpolation'''

    # get the extra axes of the function
    *a, n = np.shape(f)

    # the x grid might not coincide with the interval
    # get points which are strictly in the interior of the interval
    interior = np.greater(x, xmin) & np.less(x, xmax)

    # create an array for the restricted function
    # length of last axis is number of interior points plus 2 boundary points
    f_ = np.empty_like(f, shape=(*a, np.sum(interior)+2))

    # restrict function on each extra axis
    # first, fill in the strict interior of the function
    # then interpolate on the boundary for each extra function axis
    np.compress(interior, f, axis=-1, out=f_[..., 1:-1])
    for i in np.ndindex(*a):
        f_[i][[0, -1]] = np.interp([xmin, xmax], x, f[i])

    # get the x values of the restriction
    x_ = np.concatenate([[xmin], np.extract(interior, x), [xmax]])

    return f_, x_


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
