# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''
Shells (:mod:`glass.shells`)
============================

.. currentmodule:: glass.shells

The :mod:`glass.shells` module provides functions for the definition of
matter shells, i.e. the radial discretisation of the light cone.


.. _reference-window-functions:

Window functions
----------------

.. autoclass:: RadialWindow
.. autofunction:: tophat_windows
.. autofunction:: linear_windows
.. autofunction:: cubic_windows


Window function tools
---------------------

.. autofunction:: restrict
.. autofunction:: partition


Redshift grids
--------------

.. autofunction:: redshift_grid
.. autofunction:: distance_grid


Weight functions
----------------

.. autofunction:: distance_weight
.. autofunction:: volume_weight
.. autofunction:: density_weight

'''

import warnings
from collections import namedtuple
import numpy as np

from .core.array import ndinterp

# type checking
from typing import (Union, Sequence, List, Tuple, Optional, Callable,
                    TYPE_CHECKING)
from numpy.typing import ArrayLike
if TYPE_CHECKING:
    from cosmology import Cosmology

# types
ArrayLike1D = Union[Sequence[float], np.ndarray]
WeightFunc = Callable[[ArrayLike1D], np.ndarray]


def distance_weight(z: ArrayLike, cosmo: 'Cosmology') -> np.ndarray:
    '''Uniform weight in comoving distance.'''
    return 1/cosmo.ef(z)


def volume_weight(z: ArrayLike, cosmo: 'Cosmology') -> np.ndarray:
    '''Uniform weight in comoving volume.'''
    return cosmo.xm(z)**2/cosmo.ef(z)


def density_weight(z: ArrayLike, cosmo: 'Cosmology') -> np.ndarray:
    '''Uniform weight in matter density.'''
    return cosmo.rho_m_z(z)*cosmo.xm(z)**2/cosmo.ef(z)


RadialWindow = namedtuple('RadialWindow', 'za, wa, zeff')
RadialWindow.__doc__ = '''A radial window, defined by a window function.

    The radial window is defined by a window function in redshift, which
    is given by a pair of arrays ``za``, ``wa``.

    The radial window also has an effective redshift, stored in the
    ``zeff`` attribute, which should be a representative redshift for
    the window function.

    To prevent accidental inconsistencies, instances of this type are
    immutable (however, the array entries may **not** be immutable; do
    not change them in place)::

        >>> from glass.shells import RadialWindow
        >>> w1 = RadialWindow(..., ..., zeff=0.1)
        >>> w1.zeff = 0.15
        Traceback (most recent call last):
          File "<stdin>", line 1, in <module>
        AttributeError: can't set attribute

    To create a new instance with a changed attribute value, use the
    ``._replace`` method::

        >>> w1 = w1._replace(zeff=0.15)
        >>> w1
        RadialWindow(za=..., wa=..., zeff=0.15)

    Attributes
    ----------
    za : (N,) array_like
        Redshift array; the abscissae of the window function.
    wa : (N,) array_like
        Weight array; the values (ordinates) of the window function.
    zeff : float
        Effective redshift of the window.

    Methods
    -------
    _replace

    '''
RadialWindow.za.__doc__ = '''Redshift array; the abscissae of the window function.'''
RadialWindow.wa.__doc__ = '''Weight array; the values (ordinates) of the window function.'''
RadialWindow.zeff.__doc__ = '''Effective redshift of the window.'''


def tophat_windows(zbins: ArrayLike1D, dz: float = 1e-3,
                   weight: Optional[WeightFunc] = None
                   ) -> List[RadialWindow]:
    '''Tophat window functions from the given redshift bin edges.

    Uses the *N+1* given redshifts as bin edges to construct *N* tophat
    window functions.  The redshifts of the windows have linear spacing
    approximately equal to ``dz``.

    An optional weight function :math:`w(z)` can be given using
    ``weight``; it is applied to the tophat windows.

    The resulting windows functions are :class:`RadialWindow` instances.
    Their effective redshifts are the mean redshifts of the (weighted)
    tophat bins.

    Parameters
    ----------
    zbins : (N+1,) array_like
        Redshift bin edges for the tophat window functions.
    dz : float, optional
        Approximate spacing of the redshift grid.
    weight : callable, optional
        If given, a weight function to be applied to the window
        functions.

    Returns
    -------
    ws : (N,) list of :class:`RadialWindow`
        List of window functions.

    See Also
    --------
    :ref:`user-window-functions`

    '''
    if len(zbins) < 2:
        raise ValueError('zbins must have at least two entries')
    if zbins[0] != 0:
        warnings.warn('first tophat window does not start at redshift zero')

    wht: WeightFunc
    if weight is not None:
        wht = weight
    else:
        wht = np.ones_like

    ws = []
    for zmin, zmax in zip(zbins, zbins[1:]):
        n = max(round((zmax - zmin)/dz), 2)
        z = np.linspace(zmin, zmax, n)
        w = wht(z)
        zeff = np.trapz(w*z, z)/np.trapz(w, z)
        ws.append(RadialWindow(z, w, zeff))
    return ws


def linear_windows(zgrid: ArrayLike1D, dz: float = 1e-3,
                   weight: Optional[WeightFunc] = None,
                   ) -> List[RadialWindow]:
    '''Linear interpolation window functions.

    Uses the *N+2* given redshifts as nodes to construct *N* triangular
    window functions between the first and last node.  These correspond
    to linear interpolation of radial functions.  The redshift spacing
    of the windows is approximately equal to ``dz``.

    An optional weight function :math:`w(z)` can be given using
    ``weight``; it is applied to the triangular windows.

    The resulting windows functions are :class:`RadialWindow` instances.
    Their effective redshifts correspond to the given nodes.

    Parameters
    ----------
    zgrid : (N+2,) array_like
        Redshift grid for the triangular window functions.
    dz : float, optional
        Approximate spacing of the redshift grid.
    weight : callable, optional
        If given, a weight function to be applied to the window
        functions.

    Returns
    -------
    ws : (N,) list of :class:`RadialWindow`
        List of window functions.

    See Also
    --------
    :ref:`user-window-functions`

    '''
    if len(zgrid) < 3:
        raise ValueError('nodes must have at least 3 entries')
    if zgrid[0] != 0:
        warnings.warn('first triangular window does not start at z=0')

    ws = []
    for zmin, zmid, zmax in zip(zgrid, zgrid[1:], zgrid[2:]):
        n = max(round((zmid - zmin)/dz), 2) - 1
        m = max(round((zmax - zmid)/dz), 2)
        z = np.concatenate([np.linspace(zmin, zmid, n, endpoint=False),
                            np.linspace(zmid, zmax, m)])
        w = np.concatenate([np.linspace(0., 1., n, endpoint=False),
                            np.linspace(1., 0., m)])
        if weight is not None:
            w *= weight(z)
        ws.append(RadialWindow(z, w, zmid))
    return ws


def cubic_windows(zgrid: ArrayLike1D, dz: float = 1e-3,
                  weight: Optional[WeightFunc] = None,
                  ) -> List[RadialWindow]:
    '''Cubic interpolation window functions.

    Uses the *N+2* given redshifts as nodes to construct *N* cubic
    Hermite spline window functions between the first and last node.
    These correspond to cubic spline interpolation of radial functions.
    The redshift spacing of the windows is approximately equal to
    ``dz``.

    An optional weight function :math:`w(z)` can be given using
    ``weight``; it is applied to the cubic spline windows.

    The resulting windows functions are :class:`RadialWindow` instances.
    Their effective redshifts correspond to the given nodes.

    Parameters
    ----------
    zgrid : (N+2,) array_like
        Redshift grid for the cubic spline window functions.
    dz : float, optional
        Approximate spacing of the redshift grid.
    weight : callable, optional
        If given, a weight function to be applied to the window
        functions.

    Returns
    -------
    ws : (N,) list of :class:`RadialWindow`
        List of window functions.

    See Also
    --------
    :ref:`user-window-functions`

    '''
    if len(zgrid) < 3:
        raise ValueError('nodes must have at least 3 entries')
    if zgrid[0] != 0:
        warnings.warn('first cubic spline window does not start at z=0')

    ws = []
    for zmin, zmid, zmax in zip(zgrid, zgrid[1:], zgrid[2:]):
        n = max(round((zmid - zmin)/dz), 2) - 1
        m = max(round((zmax - zmid)/dz), 2)
        z = np.concatenate([np.linspace(zmin, zmid, n, endpoint=False),
                            np.linspace(zmid, zmax, m)])
        u = np.linspace(0., 1., n, endpoint=False)
        v = np.linspace(1., 0., m)
        w = np.concatenate([u**2*(3-2*u), v**2*(3-2*v)])
        if weight is not None:
            w *= weight(z)
        ws.append(RadialWindow(z, w, zmid))
    return ws


def restrict(z: ArrayLike1D, f: ArrayLike1D, w: RadialWindow
             ) -> Tuple[np.ndarray, np.ndarray]:
    '''Restrict a function to a redshift window.

    Multiply the function :math:`f(z)` by a window function :math:`w(z)`
    to produce :math:`w(z) f(z)` over the support of :math:`w`.

    The function :math:`f(z)` is given by redshifts ``z`` of shape
    *(N,)* and function values ``f`` of shape *(..., N)*, with any
    number of leading axes allowed.

    The window function :math:`w(z)` is given by ``w``, which must be a
    :class:`RadialWindow` instance or compatible with it.

    The restriction has redshifts that are the union of the redshifts of
    the function and window over the support of the window.
    Intermediate function values are found by linear interpolation

    Parameters
    ----------
    z, f : array_like
        The function to be restricted.
    w : :class:`RadialWindow`
        The window function for the restriction.

    Returns
    -------
    zr, fr : array
        The restricted function.

    '''

    z_ = np.compress(np.greater(z, w.za[0]) & np.less(z, w.za[-1]), z)
    zr = np.union1d(w.za, z_)
    fr = ndinterp(zr, z, f, left=0., right=0.) * ndinterp(zr, w.za, w.wa)
    return zr, fr


def partition(z: ArrayLike,
              f: ArrayLike,
              ws: Sequence[RadialWindow],
              *,
              method: str = "lstsq",
              ) -> ArrayLike:
    """Partition a function by a sequence of windows.

    Returns a vector of weights :math:`x_1, x_2, \\ldots` such that the
    weighted sum of normalised radial window functions :math:`x_1 \\,
    w_1(z) + x_2 \\, w_2(z) + \\ldots` approximates the given function
    :math:`f(z)`.

    The function :math:`f(z)` is given by redshifts *z* of shape *(N,)*
    and function values *f* of shape *(..., N)*, with any number of
    leading axes allowed.

    The window functions are given by the sequence *ws* of
    :class:`RadialWindow` or compatible entries.

    Parameters
    ----------
    z, f : array_like
        The function to be partitioned.  If *f* is multi-dimensional,
        its last axis must agree with *z*.
    ws : sequence of :class:`RadialWindow`
        Ordered sequence of window functions for the partition.
    method : {"lstsq", "restrict"}
        Method for the partition.  See notes for description.

    Returns
    -------
    x : array_like
        Weights of the partition.  If *f* is multi-dimensional, the
        leading axes of *x* match those of *f*.

    Notes
    -----
    Formally, if :math:`w_i` are the normalised window functions,
    :math:`f` is the target function, and :math:`z_i` is a redshift grid
    with intervals :math:`\\Delta z_i`, the partition problem seeks an
    approximate solution of

    .. math::
        \\begin{pmatrix}
        w_1(z_1) \\Delta z_1 & w_2(z_1) \\, \\Delta z_1 & \\cdots \\\\
        w_1(z_2) \\Delta z_2 & w_2(z_2) \\, \\Delta z_2 & \\cdots \\\\
        \\vdots & \\vdots & \\ddots
        \\end{pmatrix} \\, \\begin{pmatrix}
        x_1 \\\\ x_2 \\\\ \\vdots
        \\end{pmatrix} = \\begin{pmatrix}
        f(z_1) \\, \\Delta z_1 \\\\ f(z_2) \\, \\Delta z_2 \\\\ \\vdots
        \\end{pmatrix} \\;.

    The redshift grid is the union of the given array *z* and the
    redshift arrays of all window functions.  Intermediate function
    values are found by linear interpolation.

    If ``method="lstsq"``, obtain a partition from a least-squares
    solution.  This will more closely match the shape of the input
    function, but the normalisation might differ.

    If ``method="restrict"``, obtain a partition by integrating the
    restriction (using :func:`restrict`) of the function :math:`f` to
    each window.  This will more closely match the normalisation of the
    input function, but the shape might differ.

    """
    try:
        partition_method = globals()[f"partition_{method}"]
    except KeyError:
        raise ValueError(f"invalid method: {method}") from None
    return partition_method(z, f, ws)


def partition_lstsq(z: ArrayLike, f: ArrayLike, ws: Sequence[RadialWindow]
                    ) -> ArrayLike:
    """Least-squares partition."""

    # compute the union of all given redshift grids
    zp = z
    for w in ws:
        zp = np.union1d(zp, w.za)

    # compute grid spacing
    dz = np.gradient(zp)

    # create the window function matrix
    a = [np.interp(zp, za, wa, left=0., right=0.) for za, wa, _ in ws]
    a = a/np.trapz(a, zp, axis=-1)[..., None]
    a = a*dz

    # create the target vector of distribution values
    b = ndinterp(zp, z, f, left=0., right=0.)
    b = b*dz

    # return least-squares fit
    return np.linalg.lstsq(a.T, b.T, rcond=None)[0].T


def partition_restrict(z: ArrayLike, f: ArrayLike, ws: Sequence[RadialWindow]
                       ) -> ArrayLike:
    """Partition by restriction and integration."""

    ngal = []
    for w in ws:
        zr, fr = restrict(z, f, w)
        ngal.append(np.trapz(fr, zr, axis=-1))
    return np.transpose(ngal)


def redshift_grid(zmin, zmax, *, dz=None, num=None):
    '''Redshift grid with uniform spacing in redshift.'''
    if dz is not None and num is None:
        z = np.arange(zmin, np.nextafter(zmax+dz, zmax), dz)
    elif dz is None and num is not None:
        z = np.linspace(zmin, zmax, num+1)
    else:
        raise ValueError('exactly one of "dz" or "num" must be given')
    return z


def distance_grid(cosmo, zmin, zmax, *, dx=None, num=None):
    '''Redshift grid with uniform spacing in comoving distance.'''
    xmin, xmax = cosmo.dc(zmin), cosmo.dc(zmax)
    if dx is not None and num is None:
        x = np.arange(xmin, np.nextafter(xmax+dx, xmax), dx)
    elif dx is None and num is not None:
        x = np.linspace(xmin, xmax, num+1)
    else:
        raise ValueError('exactly one of "dx" or "num" must be given')
    return cosmo.dc_inv(x)
