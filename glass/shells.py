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

.. autofunction:: distance_weight
.. autofunction:: volume_weight
.. autofunction:: density_weight
"""  # noqa: D205, D400

from __future__ import annotations

import typing
import warnings

import numpy as np
import numpy.typing as npt

from glass.core.array import ndinterp

if typing.TYPE_CHECKING:
    from cosmology import Cosmology


def distance_weight(
    z: npt.NDArray[np.float64],
    cosmo: Cosmology,
) -> npt.NDArray[np.float64]:
    """Uniform weight in comoving distance."""
    return 1 / cosmo.ef(z)


def volume_weight(
    z: npt.NDArray[np.float64],
    cosmo: Cosmology,
) -> npt.NDArray[np.float64]:
    """Uniform weight in comoving volume."""
    return cosmo.xm(z) ** 2 / cosmo.ef(z)


def density_weight(
    z: npt.NDArray[np.float64],
    cosmo: Cosmology,
) -> npt.NDArray[np.float64]:
    """Uniform weight in matter density."""
    return cosmo.rho_m_z(z) * cosmo.xm(z) ** 2 / cosmo.ef(z)


class RadialWindow(typing.NamedTuple):
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

        >>> from glass import RadialWindow
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
    za:
        Redshift array; the abscissae of the window function.
    wa:
        Weight array; the values (ordinates) of the window function.
    zeff:
        Effective redshift of the window.

    Methods
    -------
    _replace

    """

    za: npt.NDArray[np.float64]
    wa: npt.NDArray[np.float64]
    zeff: float = 0


def tophat_windows(
    zbins: npt.NDArray[np.float64],
    dz: float = 1e-3,
    weight: typing.Callable[
        [list[float] | npt.NDArray[np.float64]],
        npt.NDArray[np.float64],
    ]
    | None = None,
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

    Returns a list of window functions.

    Parameters
    ----------
    zbins:
        Redshift bin edges for the tophat window functions.
    dz:
        Approximate spacing of the redshift grid.
    weight:
        If given, a weight function to be applied to the window functions.

    See Also
    --------
    :ref:`user-window-functions`

    """
    if len(zbins) < 2:
        msg = "zbins must have at least two entries"
        raise ValueError(msg)
    if zbins[0] != 0:
        warnings.warn(
            "first tophat window does not start at redshift zero",
            stacklevel=2,
        )

    wht: (
        npt.NDArray[np.float64]
        | typing.Callable[
            [list[float] | npt.NDArray[np.float64]],
            npt.NDArray[np.float64],
        ]
    )
    wht = weight if weight is not None else np.ones_like
    ws = []
    for zmin, zmax in zip(zbins, zbins[1:]):
        n = max(round((zmax - zmin) / dz), 2)
        z = np.linspace(zmin, zmax, n)
        w = wht(z)
        zeff = np.trapz(  # type: ignore[attr-defined]
            w * z,
            z,
        ) / np.trapz(  # type: ignore[attr-defined]
            w,
            z,
        )
        ws.append(RadialWindow(z, w, zeff))
    return ws


def linear_windows(
    zgrid: npt.NDArray[np.float64],
    dz: float = 1e-3,
    weight: typing.Callable[
        [list[float] | npt.NDArray[np.float64]],
        npt.NDArray[np.float64],
    ]
    | None = None,
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

    Returns a list of window functions.

    Parameters
    ----------
    zgrid:
        Redshift grid for the triangular window functions.
    dz:
        Approximate spacing of the redshift grid.
    weight:
        If given, a weight function to be applied to the window functions.

    See Also
    --------
    :ref:`user-window-functions`

    """
    if len(zgrid) < 3:
        msg = "nodes must have at least 3 entries"
        raise ValueError(msg)
    if zgrid[0] != 0:
        warnings.warn("first triangular window does not start at z=0", stacklevel=2)

    ws = []
    for zmin, zmid, zmax in zip(zgrid, zgrid[1:], zgrid[2:]):
        n = max(round((zmid - zmin) / dz), 2) - 1
        m = max(round((zmax - zmid) / dz), 2)
        z = np.concatenate(
            [np.linspace(zmin, zmid, n, endpoint=False), np.linspace(zmid, zmax, m)],
        )
        w = np.concatenate(
            [np.linspace(0.0, 1.0, n, endpoint=False), np.linspace(1.0, 0.0, m)],
        )
        if weight is not None:
            w *= weight(z)
        ws.append(RadialWindow(z, w, zmid))
    return ws


def cubic_windows(
    zgrid: npt.NDArray[np.float64],
    dz: float = 1e-3,
    weight: typing.Callable[
        [list[float] | npt.NDArray[np.float64]],
        npt.NDArray[np.float64],
    ]
    | None = None,
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

    Returns a list of window functions.

    Parameters
    ----------
    zgrid:
        Redshift grid for the cubic spline window functions.
    dz:
        Approximate spacing of the redshift grid.
    weight:
        If given, a weight function to be applied to the window functions.

    See Also
    --------
    :ref:`user-window-functions`

    """
    if len(zgrid) < 3:
        msg = "nodes must have at least 3 entries"
        raise ValueError(msg)
    if zgrid[0] != 0:
        warnings.warn("first cubic spline window does not start at z=0", stacklevel=2)

    ws = []
    for zmin, zmid, zmax in zip(zgrid, zgrid[1:], zgrid[2:]):
        n = max(round((zmid - zmin) / dz), 2) - 1
        m = max(round((zmax - zmid) / dz), 2)
        z = np.concatenate(
            [np.linspace(zmin, zmid, n, endpoint=False), np.linspace(zmid, zmax, m)],
        )
        u = np.linspace(0.0, 1.0, n, endpoint=False)
        v = np.linspace(1.0, 0.0, m)
        w = np.concatenate([u**2 * (3 - 2 * u), v**2 * (3 - 2 * v)])
        if weight is not None:
            w *= weight(z)
        ws.append(RadialWindow(z, w, zmid))
    return ws


def restrict(
    z: npt.NDArray[np.float64],
    f: npt.NDArray[np.float64],
    w: RadialWindow,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
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

    Returns the restricted function

    Parameters
    ----------
    z:
        The function to be restricted.
    f:
        The function to be restricted.
    w:
        The window function for the restriction.

    """
    z_ = np.compress(np.greater(z, w.za[0]) & np.less(z, w.za[-1]), z)
    zr = np.union1d(w.za, z_)
    fr = ndinterp(zr, z, f, left=0.0, right=0.0) * ndinterp(zr, w.za, w.wa)
    return zr, fr


def partition(
    z: npt.NDArray[np.float64],
    fz: npt.NDArray[np.float64],
    shells: list[RadialWindow],
    *,
    method: str = "nnls",
) -> npt.NDArray[np.float64]:
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

    Returns the weights of the partition, where the leading axis corresponds to
    *shells*.

    Parameters
    ----------
    z:
        The function to be partitioned. If *f* is multi-dimensional,
        its last axis must agree with *z*.
    fz:
        The function to be partitioned. If *f* is multi-dimensional,
        its last axis must agree with *z*.
    shells:
        Ordered sequence of window functions for the partition.
    method:
        Method for the partition. See notes for description. The
        options are "lstsq", "nnls", "restrict".

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

    The :func:`partition()` function implements a number of methods to
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
    restriction (using :func:`restrict`) of the function :math:`f` to
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
    z: npt.NDArray[np.float64],
    fz: npt.NDArray[np.float64],
    shells: list[RadialWindow],
    *,
    sumtol: float = 0.01,
) -> npt.NDArray[np.float64]:
    """Least-squares partition."""
    # make sure nothing breaks
    sumtol = max(sumtol, 1e-4)

    # compute the union of all given redshift grids
    zp = z
    for w in shells:
        zp = np.union1d(zp, w.za)

    # get extra leading axes of fz
    *dims, _ = np.shape(fz)

    # compute grid spacing
    dz = np.gradient(zp)

    # create the window function matrix
    a = np.array([np.interp(zp, za, wa, left=0.0, right=0.0) for za, wa, _ in shells])
    a /= np.trapz(  # type: ignore[attr-defined]
        a,
        zp,
        axis=-1,
    )[..., None]
    a = a * dz

    # create the target vector of distribution values
    b = ndinterp(zp, z, fz, left=0.0, right=0.0)
    b = b * dz

    # append a constraint for the integral
    mult = 1 / sumtol
    a = np.concatenate([a, mult * np.ones((len(shells), 1))], axis=-1)
    b = np.concatenate(
        [
            b,
            mult
            * np.reshape(
                np.trapz(  # type: ignore[attr-defined]
                    fz,
                    z,
                ),
                (*dims, 1),
            ),
        ],
        axis=-1,
    )

    # now a is a matrix of shape (len(shells), len(zp) + 1)
    # and b is a matrix of shape (*dims, len(zp) + 1)
    # need to find weights x such that b == x @ a over all axes of b
    # do the least-squares fit over partially flattened b, then reshape
    x = np.linalg.lstsq(a.T, b.reshape(-1, zp.size + 1).T, rcond=None)[0]
    x = x.T.reshape(*dims, len(shells))
    # roll the last axis of size len(shells) to the front
    return np.moveaxis(x, -1, 0)


def partition_nnls(
    z: npt.NDArray[np.float64],
    fz: npt.NDArray[np.float64],
    shells: list[RadialWindow],
    *,
    sumtol: float = 0.01,
) -> npt.NDArray[np.float64]:
    """
    Non-negative least-squares partition.

    Uses the ``nnls()`` algorithm from ``scipy.optimize`` and thus
    requires SciPy.

    """
    from glass.core.algorithm import nnls

    # make sure nothing breaks
    sumtol = max(sumtol, 1e-4)

    # compute the union of all given redshift grids
    zp = z
    for w in shells:
        zp = np.union1d(zp, w.za)

    # get extra leading axes of fz
    *dims, _ = np.shape(fz)

    # compute grid spacing
    dz = np.gradient(zp)

    # create the window function matrix
    a = np.array(
        [
            np.interp(
                zp,
                za,
                wa,
                left=0.0,
                right=0.0,
            )
            for za, wa, _ in shells
        ],
    )
    a /= np.trapz(  # type: ignore[attr-defined]
        a,
        zp,
        axis=-1,
    )[..., None]
    a = a * dz

    # create the target vector of distribution values
    b = ndinterp(zp, z, fz, left=0.0, right=0.0)
    b = b * dz

    # append a constraint for the integral
    mult = 1 / sumtol
    a = np.concatenate([a, mult * np.ones((len(shells), 1))], axis=-1)
    b = np.concatenate(
        [
            b,
            mult
            * np.reshape(
                np.trapz(  # type: ignore[attr-defined]
                    fz,
                    z,
                ),
                (*dims, 1),
            ),
        ],
        axis=-1,
    )

    # now a is a matrix of shape (len(shells), len(zp) + 1)
    # and b is a matrix of shape (*dims, len(zp) + 1)
    # for each dim, find non-negative weights x such that b == a.T @ x

    # reduce the dimensionality of the problem using a thin QR decomposition
    q, r = np.linalg.qr(a.T)
    y = np.einsum("ji,...j", q, b)

    # for each dim, find non-negative weights x such that y == r @ x
    x = np.empty([len(shells), *dims])
    for i in np.ndindex(*dims):
        x[(..., *i)] = nnls(r, y[i])

    # all done
    return x


def partition_restrict(
    z: npt.NDArray[np.float64],
    fz: npt.NDArray[np.float64],
    shells: list[RadialWindow],
) -> npt.NDArray[np.float64]:
    """Partition by restriction and integration."""
    part = np.empty((len(shells),) + np.shape(fz)[:-1])
    for i, w in enumerate(shells):
        zr, fr = restrict(z, fz, w)
        part[i] = np.trapz(  # type: ignore[attr-defined]
            fr,
            zr,
            axis=-1,
        )
    return part


def redshift_grid(
    zmin: float,
    zmax: float,
    *,
    dz: float | None = None,
    num: int | None = None,
) -> npt.NDArray[np.float64]:
    """Redshift grid with uniform spacing in redshift."""
    if dz is not None and num is None:
        z = np.arange(zmin, np.nextafter(zmax + dz, zmax), dz)
    elif dz is None and num is not None:
        z = np.linspace(zmin, zmax, num + 1)
    else:
        msg = 'exactly one of "dz" or "num" must be given'
        raise ValueError(msg)
    return z


def distance_grid(
    cosmo: Cosmology,
    zmin: float,
    zmax: float,
    *,
    dx: float | None = None,
    num: int | None = None,
) -> npt.NDArray[np.float64]:
    """Redshift grid with uniform spacing in comoving distance."""
    xmin, xmax = cosmo.dc(zmin), cosmo.dc(zmax)
    if dx is not None and num is None:
        x = np.arange(xmin, np.nextafter(xmax + dx, xmax), dx)
    elif dx is None and num is not None:
        x = np.linspace(xmin, xmax, num + 1)
    else:
        msg = 'exactly one of "dx" or "num" must be given'
        raise ValueError(msg)
    return cosmo.dc_inv(x)


def combine(
    z: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    shells: list[RadialWindow],
) -> npt.NDArray[np.float64]:
    r"""
    Evaluate a linear combination of window functions.

    Takes a vector of weights :math:`x_1, x_2, \ldots` and computes the
    weighted sum of normalised radial window functions :math:`f(z) = x_1
    \, w_1(z) + x_2 \, w_2(z) + \ldots` in the given redshifts
    :math:`z`.

    The window functions are given by the sequence *shells* of
    :class:`RadialWindow` or compatible entries.

    Returns a linear combination of window functions, evaluated in *z*.

    Parameters
    ----------
    z:
        Redshifts *z* in which to evaluate the combined function.
    weights:
        Weights of the linear combination, where the leading axis
        corresponds to *shells*.
    shells:
        Ordered sequence of window functions to be combined.

    See Also
    --------
    partition:
        Find weights for a given function.

    """
    return (
        np.expand_dims(weight, -1)
        * np.interp(
            z,
            shell.za,
            shell.wa
            / np.trapz(  # type: ignore[attr-defined]
                shell.wa,
                shell.za,
            ),
            left=0.0,
            right=0.0,
        )
        for shell, weight in zip(shells, weights)
    ).sum(axis=0)
