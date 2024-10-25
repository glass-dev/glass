"""
Random fields
=============

.. currentmodule:: glass

The following functions provide functionality for simulating random
fields on the sphere. This is done in the form of HEALPix maps.

Functions
---------

.. autofunction:: discretized_cls
.. autofunction:: lognormal_gls
.. autofunction:: generate_gaussian
.. autofunction:: generate_lognormal
.. autofunction:: effective_cls


Utility functions
-----------------

.. autofunction:: getcl

"""  # noqa: D205, D400

from __future__ import annotations

import typing
import warnings

import healpy as hp
import numpy as np
import numpy.typing as npt
from gaussiancl import gaussiancl

if typing.TYPE_CHECKING:
    import collections.abc


def iternorm(
    k: int,
    cov: collections.abc.Iterable[npt.NDArray[np.float64]],
    size: tuple[int, ...] = (),
) -> collections.abc.Generator[
    tuple[int | None, npt.NDArray[np.float64], npt.NDArray[np.float64]]
]:
    """
    _summary_.

    Parameters
    ----------
    k
        _description_
    cov
        _description_
    size
        _description_

    Yields
    ------
        _description_

    Raises
    ------
    TypeError
        _description_
    ValueError
        _description_

    """
    m = np.zeros((*size, k, k))
    a = np.zeros((*size, k))
    s = np.zeros((*size,))
    q = (*size, k + 1)
    j = 0 if k > 0 else None

    for i, x in enumerate(cov):
        x = np.asanyarray(x)  # noqa: PLW2901
        if x.shape != q:
            try:
                x = np.broadcast_to(x, q)  # noqa: PLW2901
            except ValueError:
                msg = f"covariance row {i}: shape {x.shape} cannot be broadcast to {q}"
                raise TypeError(msg) from None

        # only need to update matrix A if there are correlations
        if j is not None:
            # compute new entries of matrix A
            m[..., :, j] = 0
            m[..., j : j + 1, :] = np.matmul(a[..., np.newaxis, :], m)
            m[..., j, j] = np.where(s != 0, -1, 0)
            np.divide(
                m[..., j, :],
                -s[..., np.newaxis],
                where=(m[..., j, :] != 0),
                out=m[..., j, :],
            )

            # compute new vector a
            c = x[..., 1:, np.newaxis]
            a = np.matmul(m[..., :j], c[..., k - j :, :])
            a += np.matmul(m[..., j:], c[..., : k - j, :])
            a = a.reshape(*size, k)

            # next rolling index
            j = (j - 1) % k

        # compute new standard deviation
        s = x[..., 0] - np.einsum("...i,...i", a, a)
        if np.any(s < 0):
            msg = "covariance matrix is not positive definite"
            raise ValueError(msg)
        s = np.sqrt(s)

        # yield the next index, vector a, and standard deviation s
        yield j, a, s


def cls2cov(
    cls: npt.NDArray[np.float64],
    nl: int,
    nf: int,
    nc: int,
) -> collections.abc.Generator[npt.NDArray[np.float64]]:
    """
    _summary_.

    Parameters
    ----------
    cls
        _description_
    nl
        _description_
    nf
        _description_
    nc
        _description_

    Yields
    ------
        _description_

    Raises
    ------
    ValueError
        _description_

    """
    cov = np.zeros((nl, nc + 1))
    end = 0
    for j in range(nf):
        begin, end = end, end + j + 1
        for i, cl in enumerate(cls[begin:end][: nc + 1]):
            if i == 0 and np.any(np.less(cl, 0)):
                msg = "negative values in cl"
                raise ValueError(msg)
            n = len(cl)
            cov[:n, i] = cl
            cov[n:, i] = 0
        cov /= 2
        yield cov


def multalm(
    alm: npt.NDArray[np.complex128],
    bl: npt.NDArray[np.float64],
    *,
    inplace: bool = False,
) -> npt.NDArray[np.complex128]:
    """
    _summary_.

    Parameters
    ----------
    alm
        _description_
    bl
        _description_
    inplace
        _description_

    Returns
    -------
        _description_

    """
    n = len(bl)
    out = np.asanyarray(alm) if inplace else np.copy(alm)
    for m in range(n):
        out[m * n - m * (m - 1) // 2 : (m + 1) * n - m * (m + 1) // 2] *= bl[m:]
    return out


def transform_cls(
    cls: npt.NDArray[np.float64],
    tfm: str | typing.Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    pars: tuple[typing.Any, ...] = (),
) -> list[list[float] | npt.NDArray[np.float64]]:
    """
    _summary_.

    Parameters
    ----------
    cls
        _description_
    tfm
        _description_
    pars
        _description_

    Returns
    -------
        _description_

    """
    gls = []
    for cl in cls:
        if len(cl) > 0:
            monopole = 0.0 if cl[0] == 0 else None
            gl, info, _, _ = gaussiancl(cl, tfm, pars, monopole=monopole)
            if info == 0:
                warnings.warn(
                    "Gaussian cl did not converge, inexact transform",
                    stacklevel=2,
                )
        else:
            gl = []
        gls.append(gl)
    return gls


def discretized_cls(
    cls: list[list[float]] | npt.NDArray[np.float64],
    *,
    lmax: int | None = None,
    ncorr: int | None = None,
    nside: int | None = None,
) -> list[list[float] | npt.NDArray[np.float64]]:
    """
    _summary_.

    Parameters
    ----------
    cls
        _description_
    lmax
        _description_
    ncorr
        _description_
    nside
        _description_

    Returns
    -------
        _description_

    Raises
    ------
    ValueError
        _description_

    """
    if ncorr is not None:
        n = int((2 * len(cls)) ** 0.5)
        if n * (n + 1) // 2 != len(cls):
            msg = "length of cls array is not a triangle number"
            raise ValueError(msg)
        cls = [
            cls[i * (i + 1) // 2 + j] if j <= ncorr else []
            for i in range(n)
            for j in range(i + 1)
        ]

    if nside is not None:
        pw = hp.pixwin(nside, lmax=lmax)

    gls = []
    for cl in cls:
        if cl is not None and len(cl) > 0:
            if lmax is not None:
                cl = cl[: lmax + 1]  # noqa: PLW2901
            if nside is not None:
                n = min(len(cl), len(pw))
                cl = cl[:n] * pw[:n] ** 2  # noqa: PLW2901
        gls.append(cl)
    return gls


def lognormal_gls(
    cls: npt.NDArray[np.float64],
    shift: float = 1.0,
) -> list[list[float] | npt.NDArray[np.float64]]:
    """
    _summary_.

    Parameters
    ----------
    cls
        _description_
    shift
        _description_

    Returns
    -------
        _description_

    """
    return transform_cls(cls, "lognormal", (shift,))


def generate_gaussian(
    gls: npt.NDArray[np.float64],
    nside: int,
    *,
    ncorr: int | None = None,
    rng: np.random.Generator | None = None,
) -> collections.abc.Generator[npt.NDArray[np.complex128]]:
    """
    _summary_.

    Parameters
    ----------
    gls
        _description_
    nside
        _description_
    ncorr
        _description_
    rng
        _description_

    Yields
    ------
        _description_

    Raises
    ------
    ValueError
        _description_

    """
    # get the default RNG if not given
    if rng is None:
        rng = np.random.default_rng()

    # number of gls and number of fields
    ngls = len(gls)
    ngrf = int((2 * ngls) ** 0.5)

    # number of correlated fields if not specified
    if ncorr is None:
        ncorr = ngrf - 1

    # number of modes
    n = max((len(gl) for gl in gls), default=0)
    if n == 0:
        msg = "all gls are empty"
        raise ValueError(msg)

    # generates the covariance matrix for the iterative sampler
    cov = cls2cov(gls, n, ngrf, ncorr)

    # working arrays for the iterative sampling
    z = np.zeros(n * (n + 1) // 2, dtype=np.complex128)
    y = np.zeros((n * (n + 1) // 2, ncorr), dtype=np.complex128)

    # generate the conditional normal distribution for iterative sampling
    conditional_dist = iternorm(ncorr, cov, size=(n,))

    # sample the fields from the conditional distribution
    for j, a, s in conditional_dist:
        # standard normal random variates for alm
        # sample real and imaginary parts, then view as complex number
        rng.standard_normal(n * (n + 1), np.float64, z.view(np.float64))

        # scale by standard deviation of the conditional distribution
        # variance is distributed over real and imaginary part
        alm = multalm(z, s)

        # add the mean of the conditional distribution
        for i in range(ncorr):
            alm += multalm(y[:, i], a[:, i])

        # store the standard normal in y array at the indicated index
        if j is not None:
            y[:, j] = z

        # modes with m = 0 are real-valued and come first in array
        alm[:n].real += alm[:n].imag
        alm[:n].imag[:] = 0

        # transform alm to maps
        # can be performed in place on the temporary alm array
        yield hp.alm2map(alm, nside, pixwin=False, pol=False, inplace=True)


def generate_lognormal(
    gls: npt.NDArray[np.float64],
    nside: int,
    shift: float = 1.0,
    *,
    ncorr: int | None = None,
    rng: np.random.Generator | None = None,
) -> collections.abc.Generator[npt.NDArray[np.complex128]]:
    """
    _summary_.

    Parameters
    ----------
    gls
        _description_
    nside
        _description_
    shift
        _description_
    ncorr
        _description_
    rng
        _description_

    Yields
    ------
        _description_

    """
    for i, m in enumerate(generate_gaussian(gls, nside, ncorr=ncorr, rng=rng)):
        # compute the variance of the auto-correlation
        gl = gls[i * (i + 1) // 2]
        ell = np.arange(len(gl))
        var = np.sum((2 * ell + 1) * gl) / (4 * np.pi)

        # fix mean of the Gaussian random field for lognormal transformation
        m -= var / 2  # noqa: PLW2901

        # exponentiate values in place and subtract 1 in one operation
        np.expm1(m, out=m)

        # lognormal shift, unless unity
        if shift != 1:
            m *= shift  # noqa: PLW2901

        # yield the lognormal map
        yield m


def getcl(
    cls: npt.NDArray[np.float64],
    i: int,
    j: int,
    lmax: int | None = None,
) -> npt.NDArray[np.float64]:
    """
    _summary_.

    Parameters
    ----------
    cls
        _description_
    i
        _description_
    j
        _description_
    lmax
        _description_

    Returns
    -------
        _description_

    """
    if j > i:
        i, j = j, i
    cl = cls[i * (i + 1) // 2 + i - j]
    if lmax is not None:
        if len(cl) > lmax + 1:
            cl = cl[: lmax + 1]
        else:
            cl = np.pad(cl, (0, lmax + 1 - len(cl)))
    return cl


def effective_cls(
    cls: npt.NDArray[np.float64],
    weights1: npt.NDArray[np.float64],
    weights2: npt.NDArray[np.float64] | None = None,
    *,
    lmax: int | None = None,
) -> npt.NDArray[np.float64]:
    """
    _summary_.

    Parameters
    ----------
    cls
        _description_
    weights1
        _description_
    weights2
        _description_
    lmax
        _description_

    Returns
    -------
        _description_

    Raises
    ------
    ValueError
        _description_
    ValueError
        _description_

    """
    from itertools import combinations_with_replacement, product

    # this is the number of fields
    n = int((2 * len(cls)) ** 0.5)
    if n * (n + 1) // 2 != len(cls):
        msg = "length of cls is not a triangle number"
        raise ValueError(msg)

    # find lmax if not given
    if lmax is None:
        lmax = max(map(len, cls), default=0) - 1

    # broadcast weights1 such that its shape ends in n
    weights1 = np.asanyarray(weights1)
    weights2 = np.asanyarray(weights2) if weights2 is not None else weights1

    shape1, shape2 = weights1.shape, weights2.shape
    for i, shape in enumerate((shape1, shape2)):
        if not shape or shape[0] != n:
            msg = f"shape mismatch between fields and weights{i+1}"
            raise ValueError(msg)

    # get the iterator over leading weight axes
    # auto-spectra do not repeat identical computations
    pairs = (
        combinations_with_replacement(np.ndindex(shape1[1:]), 2)
        if weights2 is weights1
        else product(np.ndindex(shape1[1:]), np.ndindex(shape2[1:]))
    )

    # create the output array: axes for all input axes plus lmax+1
    out = np.empty(shape1[1:] + shape2[1:] + (lmax + 1,))

    # helper that will grab the entire first column (i.e. shells)
    c = (slice(None),)

    # compute all combined cls from pairs
    # if weights2 is weights1, set the transpose elements in one pass
    for j1, j2 in pairs:
        w1, w2 = weights1[c + j1], weights2[c + j2]
        cl = sum(
            w1[i1] * w2[i2] * getcl(cls, i1, i2, lmax=lmax)
            for i1, i2 in np.ndindex(n, n)
        )
        out[j1 + j2] = cl
        if weights2 is weights1 and j1 != j2:
            out[j2 + j1] = cl
    return out
