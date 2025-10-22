"""
Shells
======

.. currentmodule:: glass

The following functions provide functionality for the definition of
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
.. autofunction:: combine


Redshift grids
--------------

.. autofunction:: redshift_grid
.. autofunction:: distance_grid


Weight functions
----------------

.. autoclass:: DistanceWeight
.. autoclass:: VolumeWeight
.. autoclass:: DensityWeight

"""  # noqa: D400

from __future__ import annotations

import dataclasses
import itertools
import math
import warnings
from typing import TYPE_CHECKING

import numpy as np

import glass._array_api_utils as _utils
import glass.algorithm
import glass.arraytools
from glass._array_api_utils import XPAdditions

if TYPE_CHECKING:
    import types
    from collections.abc import Callable, Iterator, Sequence

    from numpy.typing import NDArray

    from glass._array_api_utils import FloatArray
    from glass.cosmology import Cosmology

    ArrayLike1D = Sequence[float] | NDArray[np.float64]
    WeightFunc = Callable[[ArrayLike1D], NDArray[np.float64]]


@dataclasses.dataclass
class DistanceWeight:
    """Uniform weight in comoving distance.

    Attributes
    ----------
    cosmo
        Cosmology instance.

    """

    cosmo: Cosmology

    def __call__(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Uniform weight in comoving distance.

        Parameters
        ----------
        z
            The redshifts at which to evaluate the weight.

        Returns
        -------
            The weight function evaluated at redshifts *z*.

        """
        return 1 / self.cosmo.H_over_H0(z)  # type: ignore[no-any-return]


@dataclasses.dataclass
class VolumeWeight:
    """Uniform weight in comoving volume.

    Attributes
    ----------
    cosmo
        Cosmology instance.

    """

    cosmo: Cosmology

    def __call__(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Uniform weight in comoving distance.

        Parameters
        ----------
        z
            The redshifts at which to evaluate the weight.

        Returns
        -------
            The weight function evaluated at redshifts *z*.

        """
        return (  # type: ignore[no-any-return]
            (self.cosmo.transverse_comoving_distance(z) / self.cosmo.hubble_distance)
            ** 2
            / self.cosmo.H_over_H0(z)
        )


@dataclasses.dataclass
class DensityWeight:
    """
    Uniform weight in matter density.

    Attributes
    ----------
    cosmo
        Cosmology instance.

    """

    cosmo: Cosmology

    def __call__(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Uniform weight in comoving distance.

        Parameters
        ----------
        z
            The redshifts at which to evaluate the weight.

        Returns
        -------
            The weight function evaluated at redshifts *z*.

        """
        return (  # type: ignore[no-any-return]
            self.cosmo.critical_density0
            * self.cosmo.Omega_m(z)
            * (self.cosmo.transverse_comoving_distance(z) / self.cosmo.hubble_distance)
            ** 2
            / self.cosmo.H_over_H0(z)
        )


@dataclasses.dataclass(frozen=True)
class RadialWindow:
    """
    A radial window, defined by a window function.

    The radial window is defined by a window function in redshift, which
    is given by a pair of arrays ``za``, ``wa``.

    The radial window also has an effective redshift, stored in the
    ``zeff`` attribute, which should be a representative redshift for
    the window function.

    To prevent accidental inconsistencies, instances of this type are
    immutable (however, the array entries may **not** be immutable; do
    not change them in place)::

        >>> import glass
        >>> import numpy as np
        >>> za = np.asarray([0.0, 1.0])
        >>> wa = np.asarray([1.0, 0.0])
        >>> w1 = glass.RadialWindow(za, wa, zeff=0.1)
        >>> w1.zeff = 0.15
        Traceback (most recent call last):
          File "<string>", line 4, in __setattr__
        dataclasses.FrozenInstanceError: cannot assign to field 'zeff'

    To create a new instance with a changed attribute value, use the
    ``dataclasses.replace`` method::

        >>> w1 = dataclasses.replace(w1, zeff=0.15)
        >>> w1.zeff
        0.15

    Attributes
    ----------
    za
        Redshift array; the abscissae of the window function.
    wa
        Weight array; the values (ordinates) of the window function.
    zeff
        Effective redshift of the window.

    """

    za: FloatArray
    wa: FloatArray
    zeff: float = math.nan
    xp: types.ModuleType | None = None

    def __post_init__(self) -> None:
        """
        Magic method to setup optional inputs
        - Calculates the effective redshift if not given.
        - Determines xp from za and wa.
        """
        if self.xp is None:
            object.__setattr__(self, "xp", _utils.get_namespace(self.za, self.wa))
        if math.isnan(self.zeff):
            object.__setattr__(self, "zeff", self._calculate_zeff())

    def __iter__(self) -> Iterator[FloatArray]:
        """
        Iterate over the window function and effective redshift.

        To be removed upon deprecation of ``glass.ext.camb``.
        """
        yield from (self.za, self.wa, self.zeff)

    def _calculate_zeff(self) -> float:
        """Calculate ``zeff`` if not given.

        Returns
        -------
            The effective redshift depending on the size of ``za``.

        """
        uxpx = XPAdditions(self.xp)  # type: ignore[arg-type]
        if self.za.size > 0:
            return uxpx.trapezoid(  # type: ignore[return-value]
                self.za * self.wa,
                self.za,
            ) / uxpx.trapezoid(self.wa, self.za)
        return math.nan


def tophat_windows(
    zbins: FloatArray,
    dz: float = 1e-3,
    weight: WeightFunc | None = None,
) -> list[RadialWindow]:
    """
    Tophat window functions from the given redshift bin edges.

    Uses the *N+1* given redshifts as bin edges to construct *N* tophat
    window functions. The redshifts of the windows have linear spacing
    approximately equal to ``dz``.

    An optional weight function :math:`w(z)` can be given using
    ``weight``; it is applied to the tophat windows.

    The resulting windows functions are :class:`RadialWindow` instances.
    Their effective redshifts are the mean redshifts of the (weighted)
    tophat bins.

    Parameters
    ----------
    zbins
        Redshift bin edges for the tophat window functions.
    dz
        Approximate spacing of the redshift grid.
    weight
        If given, a weight function to be applied to the window functions.

    Returns
    -------
        A list of window functions.

    Raises
    ------
    ValueError
        If the number of redshift bins is less than 2.

    See Also
    --------
    :ref:`user-window-functions`

    """
    if zbins.ndim != 1:
        msg = "zbins must be a 1D array"
        raise ValueError(msg)
    if zbins.size < 2:
        msg = "zbins must have at least two entries"
        raise ValueError(msg)
    if zbins[0] != 0:
        warnings.warn(
            "first tophat window does not start at redshift zero",
            stacklevel=2,
        )

    xp = _utils.get_namespace(zbins)
    uxpx = XPAdditions(xp)

    wht: WeightFunc
    wht = weight if weight is not None else xp.ones_like
    ws = []
    for zmin, zmax in itertools.pairwise(zbins):
        n = int(max(xp.round((zmax - zmin) / dz), 2))
        z = xp.linspace(zmin, zmax, n, dtype=xp.float64)
        w = wht(z)
        zeff = uxpx.trapezoid(w * z, z) / uxpx.trapezoid(w, z)
        ws.append(RadialWindow(z, w, float(zeff)))
    return ws


def linear_windows(
    zgrid: FloatArray,
    dz: float = 1e-3,
    weight: WeightFunc | None = None,
) -> list[RadialWindow]:
    """
    Linear interpolation window functions.

    Uses the *N+2* given redshifts as nodes to construct *N* triangular
    window functions between the first and last node. These correspond
    to linear interpolation of radial functions. The redshift spacing
    of the windows is approximately equal to ``dz``.

    An optional weight function :math:`w(z)` can be given using
    ``weight``; it is applied to the triangular windows.

    The resulting windows functions are :class:`RadialWindow` instances.
    Their effective redshifts correspond to the given nodes.

    Parameters
    ----------
    zgrid
        Redshift grid for the triangular window functions.
    dz
        Approximate spacing of the redshift grid.
    weight
        If given, a weight function to be applied to the window functions.

    Returns
    -------
        A list of window functions.

    Raises
    ------
    ValueError
        If the number of nodes is less than 3.

    See Also
    --------
    :ref:`user-window-functions`

    """
    if zgrid.ndim != 1:
        msg = "zgrid must be a 1D array"
        raise ValueError(msg)
    if zgrid.size < 3:
        msg = "nodes must have at least 3 entries"
        raise ValueError(msg)
    if zgrid[0] != 0:
        warnings.warn("first triangular window does not start at z=0", stacklevel=2)

    xp = _utils.get_namespace(zgrid)

    ws = []
    for zmin, zmid, zmax in zip(zgrid, zgrid[1:], zgrid[2:], strict=False):
        n = int(max(xp.round((zmid - zmin) / dz), 2)) - 1
        m = int(max(xp.round((zmax - zmid) / dz), 2))
        z = xp.concat(
            (xp.linspace(zmin, zmid, n, endpoint=False), xp.linspace(zmid, zmax, m)),
        )
        w = xp.concat(
            (xp.linspace(0.0, 1.0, n, endpoint=False), xp.linspace(1.0, 0.0, m)),
        )
        if weight is not None:
            w *= weight(z)
        ws.append(RadialWindow(z, w, zmid))
    return ws


def cubic_windows(
    zgrid: FloatArray,
    dz: float = 1e-3,
    weight: WeightFunc | None = None,
) -> list[RadialWindow]:
    """
    Cubic interpolation window functions.

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
    zgrid
        Redshift grid for the cubic spline window functions.
    dz
        Approximate spacing of the redshift grid.
    weight
        If given, a weight function to be applied to the window functions.

    Returns
    -------
        A list of window functions.

    Raises
    ------
    ValueError
        If the number of nodes is less than 3.

    See Also
    --------
    :ref:`user-window-functions`

    """
    if zgrid.ndim != 1:
        msg = "zgrid must be a 1D array"
        raise ValueError(msg)
    if zgrid.size < 3:
        msg = "nodes must have at least 3 entries"
        raise ValueError(msg)
    if zgrid[0] != 0:
        warnings.warn("first cubic spline window does not start at z=0", stacklevel=2)

    xp = _utils.get_namespace(zgrid)

    ws = []
    for zmin, zmid, zmax in zip(zgrid, zgrid[1:], zgrid[2:], strict=False):
        n = int(max(xp.round((zmid - zmin) / dz), 2)) - 1
        m = int(max(xp.round((zmax - zmid) / dz), 2))
        z = xp.concat(
            (xp.linspace(zmin, zmid, n, endpoint=False), xp.linspace(zmid, zmax, m)),
        )
        u = xp.linspace(0.0, 1.0, n, endpoint=False)
        v = xp.linspace(1.0, 0.0, m)
        w = xp.concat([u**2 * (3 - 2 * u), v**2 * (3 - 2 * v)])
        if weight is not None:
            w *= weight(z)
        ws.append(RadialWindow(z, w, zmid))
    return ws


def restrict(
    z: FloatArray,
    f: FloatArray,
    w: RadialWindow,
) -> tuple[FloatArray, FloatArray]:
    """
    Restrict a function to a redshift window.

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
    z
        The function to be restricted.
    f
        The function to be restricted.
    w
        The window function for the restriction.

    Returns
    -------
        The restricted function

    """
    if z.ndim != 1:
        msg = "z must be 1D arrays"
        raise ValueError(msg)
    xp = _utils.get_namespace(z, f)
    uxpx = XPAdditions(xp)

    z_ = z[xp.greater(z, w.za[0]) & xp.less(z, w.za[-1])]
    zr = uxpx.union1d(w.za, z_)

    fr = glass.arraytools.ndinterp(
        zr, z, f, left=0.0, right=0.0
    ) * glass.arraytools.ndinterp(zr, w.za, w.wa)
    return zr, fr


def partition(
    z: FloatArray,
    fz: FloatArray,
    shells: Sequence[RadialWindow],
    *,
    method: str = "nnls",
) -> FloatArray:
    r"""
    Partition a function by a sequence of windows.

    Returns a vector of weights :math:`x_1, x_2, \ldots` such that the
    weighted sum of normalised radial window functions :math:`x_1 \,
    w_1(z) + x_2 \, w_2(z) + \ldots` approximates the given function
    :math:`f(z)`.

    The function :math:`f(z)` is given by redshifts *z* of shape *(N,)*
    and function values *fz* of shape *(..., N)*, with any number of
    leading axes allowed.

    The window functions are given by the sequence *shells* of
    :class:`RadialWindow` or compatible entries.

    Parameters
    ----------
    z
        The function to be partitioned. If *f* is multi-dimensional,
        its last axis must agree with *z*.
    fz
        The function to be partitioned. If *f* is multi-dimensional,
        its last axis must agree with *z*.
    shells
        Ordered sequence of window functions for the partition.
    method
        Method for the partition. See notes for description. The
        options are "lstsq", "nnls", "restrict".

    Returns
    -------
        The weights of the partition, where the leading axis corresponds to
        *shells*.

    Raises
    ------
    ValueError
        If the method is not recognised.

    Notes
    -----
    Formally, if :math:`w_i` are the normalised window functions,
    :math:`f` is the target function, and :math:`z_i` is a redshift grid
    with intervals :math:`\Delta z_i`, the partition problem seeks an
    approximate solution of

    .. math::

        \begin{pmatrix}
        w_1(z_1) \Delta z_1 & w_2(z_1) \, \Delta z_1 & \cdots \\
        w_1(z_2) \Delta z_2 & w_2(z_2) \, \Delta z_2 & \cdots \\
        \vdots & \vdots & \ddots
        \end{pmatrix} \, \begin{pmatrix}
        x_1 \\ x_2 \\ \vdots
        \end{pmatrix} = \begin{pmatrix}
        f(z_1) \, \Delta z_1 \\ f(z_2) \, \Delta z_2 \\ \vdots
        \end{pmatrix} \;.

    The redshift grid is the union of the given array *z* and the
    redshift arrays of all window functions. Intermediate function
    values are found by linear interpolation.

    When partitioning a density function, it is usually desirable to
    keep the normalisation fixed. In that case, the problem can be
    enhanced with the further constraint that the sum of the solution
    equals the integral of the target function,

    .. math::

        \begin{pmatrix}
        w_1(z_1) \Delta z_1 & w_2(z_1) \, \Delta z_1 & \cdots \\
        w_1(z_2) \Delta z_2 & w_2(z_2) \, \Delta z_2 & \cdots \\
        \vdots & \vdots & \ddots \\
        \hline
        \lambda & \lambda & \cdots
        \end{pmatrix} \, \begin{pmatrix}
        x_1 \\ x_2 \\ \vdots
        \end{pmatrix} = \begin{pmatrix}
        f(z_1) \, \Delta z_1 \\ f(z_2) \, \Delta z_2 \\ \vdots
        \\ \hline \lambda \int \! f(z) \, dz
        \end{pmatrix} \;,

    where :math:`\lambda` is a multiplier to enforce the integral
    constraints.

    The :func:`glass.partition()` function implements a number of methods to
    obtain a solution:

    If ``method="nnls"`` (the default), obtain a partition from a
    non-negative least-squares solution. This will usually match the
    shape of the input function closely. The contribution from each
    shell is a positive number, which is required to partition e.g.
    density functions.

    If ``method="lstsq"``, obtain a partition from an unconstrained
    least-squares solution. This will more closely match the shape of
    the input function, but might lead to shells with negative
    contributions.

    If ``method="restrict"``, obtain a partition by integrating the
    restriction (using :func:`glass.restrict`) of the function :math:`f` to
    each window. For overlapping shells, this method might produce
    results which are far from the input function.

    """
    try:
        partition_method = globals()[f"partition_{method}"]
    except KeyError:
        msg = f"invalid method: {method}"
        raise ValueError(msg) from None
    return partition_method(z, fz, shells)


def partition_lstsq(
    z: FloatArray,
    fz: FloatArray,
    shells: Sequence[RadialWindow],
    *,
    sumtol: float = 0.01,
) -> FloatArray:
    """
    Least-squares partition.

    Parameters
    ----------
    z
        The function to be partitioned.
    fz
        The function to be partitioned.
    shells
        Ordered sequence of window functions.
    sumtol
        Tolerance for the sum of the partition.

    Returns
    -------
        The partition.

    """
    xp = _utils.get_namespace(z, fz)
    uxpx = XPAdditions(xp)

    # make sure nothing breaks
    sumtol = max(sumtol, 1e-4)

    # compute the union of all given redshift grids
    zp = z
    for w in shells:
        zp = uxpx.union1d(zp, w.za)

    # get extra leading axes of fz
    *dims, _ = fz.shape

    # compute grid spacing
    dz = uxpx.gradient(zp)

    # create the window function matrix
    a = xp.stack([uxpx.interp(zp, za, wa, left=0.0, right=0.0) for za, wa, _ in shells])
    a /= uxpx.trapezoid(a, zp, axis=-1)[..., None]
    a = a * dz

    # create the target vector of distribution values
    b = glass.arraytools.ndinterp(zp, z, fz, left=0.0, right=0.0)
    b = b * dz

    # append a constraint for the integral
    mult = 1 / sumtol
    a = xp.concat([a, mult * xp.ones((len(shells), 1))], axis=-1)
    b = xp.concat([b, mult * xp.reshape(uxpx.trapezoid(fz, z), (*dims, 1))], axis=-1)

    # now a is a matrix of shape (len(shells), len(zp) + 1)
    # and b is a matrix of shape (*dims, len(zp) + 1)
    # need to find weights x such that b == x @ a over all axes of b
    # do the least-squares fit over partially flattened b, then reshape
    x = uxpx.linalg_lstsq(
        xp.matrix_transpose(a),
        xp.matrix_transpose(xp.reshape(b, (-1, zp.size + 1))),
        rcond=None,
    )[0]
    x = xp.reshape(xp.matrix_transpose(x), (*dims, len(shells)))
    # roll the last axis of size len(shells) to the front
    return xp.moveaxis(x, -1, 0)


def partition_nnls(
    z: FloatArray,
    fz: FloatArray,
    shells: Sequence[RadialWindow],
    *,
    sumtol: float = 0.01,
) -> FloatArray:
    """
    Non-negative least-squares partition.

    Parameters
    ----------
    z
        The function to be partitioned.
    fz
        The function to be partitioned.
    shells
        Ordered sequence of window functions.
    sumtol
        Tolerance for the sum of the partition.

    Returns
    -------
        The partition.

    """
    xp = _utils.get_namespace(z, fz)
    uxpx = XPAdditions(xp)

    # make sure nothing breaks
    sumtol = max(sumtol, 1e-4)

    # compute the union of all given redshift grids
    zp = z
    for w in shells:
        zp = uxpx.union1d(zp, w.za)

    # get extra leading axes of fz
    *dims, _ = fz.shape

    # compute grid spacing
    dz = uxpx.gradient(zp)

    # create the window function matrix
    a = xp.stack(
        [uxpx.interp(zp, za, wa, left=0.0, right=0.0) for za, wa, _ in shells],
    )
    a /= uxpx.trapezoid(a, zp, axis=-1)[..., None]
    a = a * dz

    # create the target vector of distribution values
    b = glass.arraytools.ndinterp(zp, z, fz, left=0.0, right=0.0)
    b = b * dz

    # append a constraint for the integral
    mult = 1 / sumtol
    a = xp.concat([a, mult * xp.ones((len(shells), 1))], axis=-1)
    b = xp.concat([b, mult * xp.reshape(uxpx.trapezoid(fz, z), (*dims, 1))], axis=-1)

    # now a is a matrix of shape (len(shells), len(zp) + 1)
    # and b is a matrix of shape (*dims, len(zp) + 1)
    # for each dim, find non-negative weights x such that b == a.T @ x

    # reduce the dimensionality of the problem using a thin QR decomposition
    q, r = xp.linalg.qr(a.T)
    y = uxpx.einsum("ji,...j", q, b)

    x = xp.stack(
        [
            glass.algorithm.nnls(r, y[(*i, ...)])  # type: ignore[arg-type]
            for i in itertools.product(*(range(d) for d in dims))
        ],
        axis=0,
    )

    # all done
    return xp.reshape(xp.moveaxis(x, 0, -1), (len(shells), *dims))


def partition_restrict(
    z: FloatArray,
    fz: FloatArray,
    shells: Sequence[RadialWindow],
) -> FloatArray:
    """
    Partition by restriction and integration.

    Parameters
    ----------
    z
        The function to be partitioned.
    fz
        The function to be partitioned.
    shells
        Ordered sequence of window functions.

    Returns
    -------
        The partition.

    """
    xp = _utils.get_namespace(z, fz)
    uxpx = XPAdditions(xp)

    parts = []
    for _, w in enumerate(shells):
        zr, fr = restrict(z, fz, w)
        parts.append(uxpx.trapezoid(fr, zr, axis=-1))
    return xp.stack(parts)


def _uniform_grid(
    start: float,
    stop: float,
    *,
    step: float | None = None,
    num: int | None = None,
    xp: types.ModuleType | None = None,
) -> FloatArray:
    """
    Create a uniform grid.

    Parameters
    ----------
    start
        The minimum value.
    stop
        The maximum value.
    step
        The spacing.
    num
        The number of samples.
    xp
        The array library to use. If None, defaults to numpy

    Returns
    -------
        The uniform grid.

    Raises
    ------
    ValueError
        If both ``step`` and ``num`` are given.

    """
    if xp is None:
        xp = np
    if step is not None and num is None:
        return xp.arange(start, stop + step, step)
    if step is None and num is not None:
        return xp.linspace(start, stop, num + 1, dtype=xp.float64)
    msg = "exactly one of grid step size or number of steps must be given"
    raise ValueError(msg)


def redshift_grid(
    zmin: float,
    zmax: float,
    *,
    dz: float | None = None,
    num: int | None = None,
    xp: types.ModuleType | None = None,
) -> FloatArray:
    """
    Redshift grid with uniform spacing in redshift.

    Parameters
    ----------
    zmin
        The minimum redshift.
    zmax
        The maximum redshift.
    dz
        The redshift spacing.
    num
        The number redshift samples.
    xp
        The array library to use. If None, defaults to numpy

    Returns
    -------
        The redshift grid.

    """
    return _uniform_grid(zmin, zmax, step=dz, num=num, xp=xp)


def distance_grid(
    cosmo: Cosmology,
    zmin: float,
    zmax: float,
    *,
    dx: float | None = None,
    num: int | None = None,
) -> NDArray[np.float64]:
    """
    Redshift grid with uniform spacing in comoving distance.

    Parameters
    ----------
    cosmo
        Cosmology instance.
    zmin
        The minimum redshift.
    zmax
        The maximum redshift.
    dx
        The comoving distance spacing.
    num
        The number of samples.

    Returns
    -------
        The redshift grid.

    """
    xmin, xmax = cosmo.comoving_distance(zmin), cosmo.comoving_distance(zmax)
    x = _uniform_grid(xmin, xmax, step=dx, num=num)
    return cosmo.inv_comoving_distance(x)  # type: ignore[no-any-return]


def combine(
    z: FloatArray,
    weights: FloatArray,
    shells: Sequence[RadialWindow],
) -> FloatArray:
    r"""
    Evaluate a linear combination of window functions.

    Takes a vector of weights :math:`x_1, x_2, \ldots` and computes the
    weighted sum of normalised radial window functions :math:`f(z) = x_1
    \, w_1(z) + x_2 \, w_2(z) + \ldots` in the given redshifts
    :math:`z`.

    The window functions are given by the sequence *shells* of
    :class:`RadialWindow` or compatible entries.

    Parameters
    ----------
    z
        Redshifts *z* in which to evaluate the combined function.
    weights
        Weights of the linear combination, where the leading axis
        corresponds to *shells*.
    shells
        Ordered sequence of window functions to be combined.

    Returns
    -------
        A linear combination of window functions, evaluated in *z*.

    See Also
    --------
    partition:
        Find weights for a given function.

    """
    xp = _utils.get_namespace(
        z, weights, *(arr for shell in shells for arr in (shell.za, shell.wa))
    )
    uxpx = XPAdditions(xp)

    return xp.sum(
        xp.stack(
            [
                xp.expand_dims(weight, axis=-1)
                * uxpx.interp(
                    z,
                    shell.za,
                    shell.wa / uxpx.trapezoid(shell.wa, shell.za),
                    left=0.0,
                    right=0.0,
                )
                for shell, weight in zip(shells, weights, strict=False)
            ],
        ),
        axis=0,
    )
