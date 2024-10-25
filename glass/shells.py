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
    """
    _summary_.

    Parameters
    ----------
    z
        _description_
    cosmo
        _description_

    Returns
    -------
        _description_

    """
    return 1 / cosmo.ef(z)


def volume_weight(
    z: npt.NDArray[np.float64],
    cosmo: Cosmology,
) -> npt.NDArray[np.float64]:
    """
    _summary_.

    Parameters
    ----------
    z
        _description_
    cosmo
        _description_

    Returns
    -------
        _description_

    """
    return cosmo.xm(z) ** 2 / cosmo.ef(z)


def density_weight(
    z: npt.NDArray[np.float64],
    cosmo: Cosmology,
) -> npt.NDArray[np.float64]:
    """
    _summary_.

    Parameters
    ----------
    z
        _description_
    cosmo
        _description_

    Returns
    -------
        _description_

    """
    return cosmo.rho_m_z(z) * cosmo.xm(z) ** 2 / cosmo.ef(z)


class RadialWindow(typing.NamedTuple):
    """_summary_."""

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
    _summary_.

    Parameters
    ----------
    zbins
        _description_
    dz
        _description_
    weight
        _description_

    Returns
    -------
        _description_

    Raises
    ------
    ValueError
        _description_

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
    _summary_.

    Parameters
    ----------
    zgrid
        _description_
    dz
        _description_
    weight
        _description_

    Returns
    -------
        _description_

    Raises
    ------
    ValueError
        _description_

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
    _summary_.

    Parameters
    ----------
    zgrid
        _description_
    dz
        _description_
    weight
        _description_

    Returns
    -------
        _description_

    Raises
    ------
    ValueError
        _description_

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
    _summary_.

    Parameters
    ----------
    z
        _description_
    f
        _description_
    w
        _description_

    Returns
    -------
        _description_

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
    """
    _summary_.

    Parameters
    ----------
    z
        _description_
    fz
        _description_
    shells
        _description_
    method
        _description_

    Returns
    -------
        _description_

    Raises
    ------
    ValueError
        _description_

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
    """
    _summary_.

    Parameters
    ----------
    z
        _description_
    fz
        _description_
    shells
        _description_
    sumtol
        _description_

    Returns
    -------
        _description_

    """
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
    _summary_.

    Parameters
    ----------
    z
        _description_
    fz
        _description_
    shells
        _description_
    sumtol
        _description_

    Returns
    -------
        _description_

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
    """
    _summary_.

    Parameters
    ----------
    z
        _description_
    fz
        _description_
    shells
        _description_

    Returns
    -------
        _description_

    """
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
    """
    _summary_.

    Parameters
    ----------
    zmin
        _description_
    zmax
        _description_
    dz
        _description_
    num
        _description_

    Returns
    -------
        _description_

    Raises
    ------
    ValueError
        _description_

    """
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
    """
    _summary_.

    Parameters
    ----------
    cosmo
        _description_
    zmin
        _description_
    zmax
        _description_
    dx
        _description_
    num
        _description_

    Returns
    -------
        _description_

    Raises
    ------
    ValueError
        _description_

    """
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
    """
    _summary_.

    Parameters
    ----------
    z
        _description_
    weights
        _description_
    shells
        _description_

    Returns
    -------
        _description_

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
