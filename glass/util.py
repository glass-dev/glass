# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''module for implementation utilities'''

import numpy as np
from numba import njit


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


def cumtrapz(f, x, out=None):
    '''cumulative trapezoidal rule along last axis'''

    if out is None:
        out = np.empty_like(f)

    np.cumsum((f[..., 1:] + f[..., :-1])/2*np.diff(x), axis=-1, out=out[..., 1:])
    out[..., 0] = 0
    return out


def triaxial_axis_ratio(zeta, xi, size=None, *, rng=None):
    r'''axis ratio of a randomly projected triaxial ellipsoid

    Given the two axis ratios `1 >= zeta >= xi` of a randomly oriented triaxial
    ellipsoid, computes the axis ratio `q` of the projection.

    Parameters
    ----------
    zeta : array_like
        Axis ratio of intermediate and major axis.
    xi : array_like
        Axis ratio of minor and major axis.
    size : tuple of int or None
        Size of the random draw. If `None` is given, size is inferred from
        other inputs.
    rng : :class:`~numpy.random.Generator`, optional
        Random number generator.  If not given, a default RNG will be used.

    Returns
    -------
    q : array_like
        Axis ratio of the randomly projected ellipsoid.

    Notes
    -----
    See equations (11) and (12) in [1]_ for details.

    References
    ----------
    .. [1] Binney J., 1985, MNRAS, 212, 767. doi:10.1093/mnras/212.4.767

    '''

    # default RNG if not provided
    if rng is None:
        rng = np.random.default_rng()

    # get size from inputs if not explicitly provided
    if size is None:
        size = np.broadcast(zeta, xi).shape

    # draw random viewing angle (theta, phi)
    cos2_theta = rng.uniform(low=-1., high=1., size=size)
    cos2_theta *= cos2_theta
    sin2_theta = 1 - cos2_theta
    cos2_phi = np.cos(rng.uniform(low=0., high=2*np.pi, size=size))
    cos2_phi *= cos2_phi
    sin2_phi = 1 - cos2_phi

    # transform arrays to quantities that are used in eq. (11)
    z2m1 = np.square(zeta)
    z2m1 -= 1
    x2 = np.square(xi)

    # eq. (11) multiplied by xi^2 zeta^2
    A = (1 + z2m1*sin2_phi)*cos2_theta + x2*sin2_theta
    B2 = 4*z2m1**2*cos2_theta*sin2_phi*cos2_phi
    C = 1 + z2m1*cos2_phi

    # eq. (12)
    q = np.sqrt((A+C-np.sqrt((A-C)**2+B2))/(A+C+np.sqrt((A-C)**2+B2)))

    return q


@njit(nogil=True)
def _tally(count, indices):
    '''compiled function to tally with unit weights'''
    for i in indices:
        count[i] += 1


@njit(nogil=True)
def _tallyw(count, indices, weights):
    '''compiled function to tally with given weights'''
    for i, w in zip(indices, weights):
        count[i] += w


def tally(count, indices, weights=None):
    '''tally indices into array of counts

    Tally given indices into an array of counts.  The counts are optionally
    weighted.

    Counts are not zeroed on input to enable continued tallying.

    Parameters
    ----------
    count : (N,) array_like
        Total count for each index.
    indices : (M,) array_like
        Indices to be tallied.  Each index ``i`` must be ``0 <= i < N``.
    weights : (M,) array_like, optional
        Weight for each index.  If not given, unit weights are assumed.

    See Also
    --------
    accumulate : similar function to sum values by indices

    '''

    if weights is None:
        _tally(count, indices)
    else:
        _tallyw(count, indices, weights)


@njit(nogil=True)
def _accumulate(total, count, indices, values):
    '''compiled function to accumulate values with unit weights'''
    for i, v in zip(indices, values):
        total[i] += v
        count[i] += 1


@njit(nogil=True)
def _accumulatew(total, count, indices, values, weights):
    '''compiled function to accumulate values with given weights'''
    for i, v, w in zip(indices, values, weights):
        total[i] += w*v
        count[i] += w


def accumulate(total, count, values, indices, weights=None):
    '''accumulate values into array of totals

    Accumulate given values and indices into arrays of totals and counts.  The
    values are optionally weighted.

    Totals and counts are not zeroed on input to enable continued accumulation.

    Parameters
    ----------
    total : (N,) array_like
        Array of accumulated values.
    count : (N,) array_like
        Total count for each index.
    values : (M,) array_like
        Values to be accumulated.
    indices : (M,) array_like
        Indices for values.  Each index ``i`` must be ``0 <= i < N``.
    weights : (npos,) array_like, optional
        Weight for each value.  If not given, unit weights are assumed.

    See Also
    --------
    tally : similar function to sum counts by indices

    '''

    if weights is None:
        _accumulate(total, count, indices, values)
    else:
        _accumulatew(total, count, indices, values, weights)
