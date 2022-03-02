# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''internal module for implementation utilities'''

import numpy as np


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
        f_[i, [0, -1]] = np.interp([xmin, xmax], x, f[i])

    # get the x values of the restriction
    x_ = np.concatenate([[xmin], np.extract(interior, x), [xmax]])

    return f_, x_
