"""Functions for random fields."""

from __future__ import annotations

import itertools
import math
import warnings
from collections.abc import Sequence
from itertools import combinations_with_replacement, product
from typing import TYPE_CHECKING

import numpy as np
from transformcl import cltovar

import array_api_compat
import array_api_extra as xpx

import glass._array_api_utils as _utils
import glass.grf
import glass.harmonics
import glass.healpix as hp
import glass.shells
from glass import _rng

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable, Iterator, Sequence
    from types import ModuleType
    from typing import Literal

    from glass._types import (
        AngularPowerSpectra,
        AnyArray,
        ComplexArray,
        FloatArray,
        IntArray,
        T,
        UnifiedGenerator,
    )


try:
    from warnings import deprecated
except ImportError:
    if TYPE_CHECKING:
        from glass._types import P, R

    def deprecated(msg: str, /) -> Callable[[Callable[P, R]], Callable[P, R]]:  # type: ignore[no-redef]
        """Backport of Python's warnings.deprecated()."""
        from functools import wraps  # noqa: PLC0415
        from warnings import warn  # noqa: PLC0415

        def decorator(func: Callable[P, R], /) -> Callable[P, R]:
            @wraps(func)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                warn(msg, category=DeprecationWarning, stacklevel=2)
                return func(*args, **kwargs)

            return wrapper

        return decorator


def _inv_triangle_number(triangle_number: int) -> int:
    r"""
    The :math:`n`-th triangle number is :math:`T_n = n \, (n+1)/2`. If
    the argument is :math:`T_n`, then :math:`n` is returned. Otherwise,
    a :class:`ValueError` is raised.
    """
    n = math.floor(math.sqrt(2 * triangle_number))
    if n * (n + 1) // 2 != triangle_number:
        msg = f"not a triangle number: {triangle_number}"
        raise ValueError(msg)
    return n


def nfields_from_nspectra(nspectra: int) -> int:
    r"""
    Returns the number of fields for a number of spectra.

    Given the number of spectra *nspectra*, returns the number of
    fields *n* such that ``n * (n + 1) // 2 == nspectra`` or raises
    a :class:`ValueError` if the number of spectra is invalid.
    """
    try:
        n = _inv_triangle_number(nspectra)
    except ValueError:
        msg = f"invalid number of spectra: {nspectra}"
        raise ValueError(msg) from None
    return n


def iternorm(
    k: int,
    cov: Iterable[FloatArray],
    size: int | tuple[int, ...] = (),
) -> Generator[tuple[int | None, FloatArray, FloatArray]]:
    """
    Return the vector a and variance sigma^2 for iterative normal sampling.

    Parameters
    ----------
    k
        The number of fields.
    cov
        The covariance matrix for the fields.
    size
        The size of the covariance matrix.

    Yields
    ------
    index
        The index for iterative sampling.
    vector
        The vector for iterative sampling.
    standard_deviation
        The standard deviation for iterative sampling.

    Raises
    ------
    TypeError
        If the covariance matrix is not broadcastable to the given size.
    ValueError
        If the covariance matrix is not positive definite.

    """
    # Convert to list here to allow determining the namespace
    first = next(cov)  # type: ignore[call-overload]
    xp = first.__array_namespace__()
    uxpx = _utils.XPAdditions(xp)

    n = (size,) if isinstance(size, int) else size

    m = xp.zeros((*n, k, k))
    a = xp.zeros((*n, k))
    s = xp.zeros((*n,))
    q = (*n, k + 1)
    j = 0 if k > 0 else None

    # We must use cov_expanded here as cov has been consumed to determine the namespace
    for i, x in enumerate(itertools.chain([first], cov)):
        # Ideally would be xp.asanyarray but this does not yet exist. The key difference
        # between the two in numpy is that asanyarray maintains subclasses of NDArray
        # whereas asarray will return the base class NDArray. Currently, we don't seem
        # to pass a subclass of NDArray so this, so it might be okay
        x = xp.asarray(x)  # noqa: PLW2901
        if x.shape != q:
            try:
                x = xp.broadcast_to(x, q)  # noqa: PLW2901
            except ValueError:
                msg = f"covariance row {i}: shape {x.shape} cannot be broadcast to {q}"
                raise TypeError(msg) from None

        # only need to update matrix A if there are correlations
        if j is not None:
            # compute new entries of matrix A
            m[..., :, j] = 0
            m[..., j : j + 1, :] = xp.matmul(a[..., xp.newaxis, :], m)
            m[..., j, j] = xp.where(s != 0, -1, s)
            # To ensure we don't divide by zero or nan we use a mask to only divide the
            # appropriate values of m and s
            m_j = m[..., j, :]
            s_broadcast = xp.broadcast_to(s[..., xp.newaxis], m_j.shape)
            mask = (m_j != 0) & (s_broadcast != 0) & ~xp.isnan(s_broadcast)
            m_j[mask] = xp.divide(m_j[mask], -s_broadcast[mask])
            m[..., j, :] = m_j

            # compute new vector a
            c = x[..., 1:, xp.newaxis]
            a = xp.matmul(m[..., :j], c[..., k - j :, :])
            a += xp.matmul(m[..., j:], c[..., : k - j, :])
            a = xp.reshape(a, (*n, k))

            # next rolling index
            j = (j - 1) % k

        # compute new standard deviation
        einsum_result_np = uxpx.einsum("...i,...i", a, a)
        s = x[..., 0] - einsum_result_np
        if xp.any(s < 0):
            msg = "covariance matrix is not positive definite"
            raise ValueError(msg)
        s = xp.sqrt(s)

        # yield the next index, vector a, and standard deviation s
        yield j, a, s


def cls2cov(
    cls: AngularPowerSpectra,
    nl: int,
    nf: int,
    nc: int,
) -> Generator[FloatArray]:
    """
    Return array of Cls as a covariance matrix for iterative sampling.

    Parameters
    ----------
    cls
        Angular matter power spectra in *GLASS* ordering.
    nl
        The number of modes.
    nf
        The number of fields.
    nc
        The number of correlated fields.

    Yields
    ------
    matrix
        The covariance matrix for iterative sampling.

    Raises
    ------
    ValueError
        If negative values are found in the Cls.

    """
    xp = array_api_compat.array_namespace(*cls, use_compat=False)

    cov = xp.zeros((nl, nc + 1))
    end = 0
    for j in range(nf):
        begin, end = end, end + j + 1
        for i, cl in enumerate(cls[begin:end][: nc + 1]):
            if i == 0 and xp.any(xp.less(cl, 0)):
                msg = "negative values in cl"
                raise ValueError(msg)
            n = cl.shape[0]
            cov = xpx.at(cov)[:n, i].set(cl)
            cov = xpx.at(cov)[n:, i].set(0.0)
        cov /= 2
        yield cov


def discretized_cls(
    cls: AngularPowerSpectra,
    *,
    lmax: int | None = None,
    ncorr: int | None = None,
    nside: int | None = None,
) -> AngularPowerSpectra:
    """
    Apply discretisation effects to angular power spectra.

    Depending on the given arguments, this truncates the angular power spectra
    to ``lmax``, removes all but ``ncorr`` correlations between fields, and
    applies the HEALPix pixel window function of the given ``nside``. If no
    arguments are given, no action is performed.

    Parameters
    ----------
    cls
        Angular matter power spectra in *GLASS* ordering.
    lmax
        The maximum mode number to truncate the spectra.
    ncorr
        The number of correlated fields to keep.
    nside
        The resolution parameter for the HEALPix maps.

    Returns
    -------
        The discretised angular power spectra.

    Raises
    ------
    ValueError
        If the length of the Cls array is not a triangle number.

    """
    if len(cls) == 0:  # type: ignore[arg-type]
        return []

    xp = array_api_compat.array_namespace(*cls, use_compat=False)

    if ncorr is not None:
        n = nfields_from_nspectra(len(cls))
        cls = [
            cls[i * (i + 1) // 2 + j] if j <= ncorr else xp.asarray([])
            for i in range(n)
            for j in range(i + 1)
        ]

    if nside is not None:
        pw = hp.pixwin(nside, lmax=lmax, xp=xp)

    gls = []
    for cl in cls:
        if cl.shape[0] > 0:
            if lmax is not None:
                cl = cl[: lmax + 1]  # noqa: PLW2901
            if nside is not None:
                n = min(cl.shape[0], pw.shape[0])
                cl = cl[:n] * pw[:n] ** 2  # noqa: PLW2901
        gls.append(cl)
    return gls


@deprecated("use glass.solve_gaussian_spectra() instead")
def lognormal_gls(
    cls: AngularPowerSpectra,
    shift: float = 1.0,
) -> AngularPowerSpectra:
    """
    Compute Gaussian Cls for a lognormal random field.

    .. deprecated:: 2025.1

       Use :func:`glass.lognormal_fields` and
       :func:`glass.compute_gaussian_spectra` or
       :func:`glass.solve_gaussian_spectra` instead.

    Parameters
    ----------
    cls
        Angular matter power spectra in *GLASS* ordering.
    shift
        The shift parameter for the lognormal transformation.

    Returns
    -------
        The Gaussian angular power spectra for a lognormal random field.

    """
    n = nfields_from_nspectra(len(cls))
    fields = [glass.grf.Lognormal(shift) for _ in range(n)]
    return solve_gaussian_spectra(fields, cls)


def _generate_grf(
    gls: AngularPowerSpectra,
    nside: int,
    *,
    ncorr: int | None = None,
    rng: UnifiedGenerator | None = None,
) -> Generator[FloatArray]:
    """
    Iteratively sample Gaussian random fields (internal use).

    A generator that iteratively samples HEALPix maps of Gaussian random fields
    with the given angular power spectra ``gls`` and resolution parameter
    ``nside``.

    The optional argument ``ncorr`` can be used to artificially limit now many
    realised fields are correlated. This saves memory, as only `ncorr` previous
    fields need to be kept.

    The ``gls`` array must contain the angular power spectra of the
    Gaussian random fields in :ref:`standard order <twopoint_order>`.

    Parameters
    ----------
    gls
        The Gaussian angular power spectra for a random field.
    nside
        The resolution parameter for the HEALPix maps.
    ncorr
        The number of correlated fields. If not given, all fields are correlated.
    rng
        Random number generator. If not given, a default RNG is used.

    Yields
    ------
    fields
        The Gaussian random fields.

    Raises
    ------
    ValueError
        If all gls are empty.

    """
    if rng is None:
        rng = _rng.rng_dispatcher(xp=np)

    # number of gls and number of fields
    ngls = len(gls)
    ngrf = nfields_from_nspectra(ngls)

    # number of correlated fields if not specified
    if ncorr is None:
        ncorr = ngrf - 1

    # number of modes
    n = max((len(gl) for gl in gls), default=0)  # type: ignore[arg-type]
    if n == 0:
        msg = "all gls are empty"
        raise ValueError(msg)

    # generates the covariance matrix for the iterative sampler
    cov = cls2cov(gls, n, ngrf, ncorr)

    # working arrays for the iterative sampling
    z = np.zeros(n * (n + 1) // 2, dtype=np.complex128)
    y = np.zeros((n * (n + 1) // 2, ncorr), dtype=np.complex128)

    # generate the conditional normal distribution for iterative sampling
    conditional_dist = iternorm(ncorr, cov, size=n)

    # sample the fields from the conditional distribution
    for j, a, s in conditional_dist:
        # standard normal random variates for alm
        # sample real and imaginary parts, then view as complex number
        rng.standard_normal(n * (n + 1), np.float64, z.view(np.float64))  # type: ignore[call-arg]

        # scale by standard deviation of the conditional distribution
        # variance is distributed over real and imaginary part
        alm = glass.harmonics.multalm(z, s)

        # add the mean of the conditional distribution
        for i in range(ncorr):
            alm += glass.harmonics.multalm(y[:, i], a[:, i])

        # store the standard normal in y array at the indicated index
        if j is not None:
            y[:, j] = z

        alm = _glass_to_healpix_alm(alm)

        # modes with m = 0 are real-valued and come first in array
        np.real(alm[:n])[:] += np.imag(alm[:n])
        np.imag(alm[:n])[:] = 0

        # transform alm to maps
        # can be performed in place on the temporary alm array
        yield hp.alm2map(alm, nside, pixwin=False, pol=False, inplace=True)


@deprecated("use glass.generate() instead")
def generate_gaussian(
    gls: AngularPowerSpectra,
    nside: int,
    *,
    ncorr: int | None = None,
    rng: np.random.Generator | None = None,
) -> Generator[FloatArray]:
    """
    Sample Gaussian random fields from Cls iteratively.

    .. deprecated:: 2025.1

       Use :func:`glass.generate` instead.

    A generator that iteratively samples HEALPix maps of Gaussian random fields
    with the given angular power spectra ``gls`` and resolution parameter
    ``nside``.

    The optional argument ``ncorr`` can be used to artificially limit now many
    realised fields are correlated. This saves memory, as only `ncorr` previous
    fields need to be kept.

    The ``gls`` array must contain the angular power spectra of the
    Gaussian random fields in :ref:`standard order <twopoint_order>`.

    Parameters
    ----------
    gls
        The Gaussian angular power spectra for a random field.
    nside
        The resolution parameter for the HEALPix maps.
    ncorr
        The number of correlated fields. If not given, all fields are correlated.
    rng
        Random number generator. If not given, a default RNG is used.

    Yields
    ------
    fields
        The Gaussian random fields.

    Raises
    ------
    ValueError
        If all gls are empty.

    """
    n = nfields_from_nspectra(len(gls))
    fields = [glass.grf.Normal() for _ in range(n)]
    yield from generate(fields, gls, nside, ncorr=ncorr, rng=rng)


@deprecated("use glass.generate() instead")
def generate_lognormal(
    gls: AngularPowerSpectra,
    nside: int,
    shift: float = 1.0,
    *,
    ncorr: int | None = None,
    rng: np.random.Generator | None = None,
) -> Generator[FloatArray]:
    """
    Sample lognormal random fields from Gaussian Cls iteratively.

    .. deprecated:: 2025.1

       Use :func:`glass.generate` instead.

    Parameters
    ----------
    gls
        The Gaussian angular power spectra for a lognormal random field.
    nside
        The resolution parameter for the HEALPix maps.
    shift
        The shift parameter for the lognormal transformation.
    ncorr
        The number of correlated fields. If not given, all fields are correlated.
    rng
        Random number generator. If not given, a default RNG is used.

    Yields
    ------
    fields
        The lognormal random fields.

    """
    n = nfields_from_nspectra(len(gls))
    fields = [glass.grf.Lognormal(shift) for _ in range(n)]
    yield from generate(fields, gls, nside, ncorr=ncorr, rng=rng)


def getcl(
    cls: AngularPowerSpectra,
    i: int,
    j: int,
    lmax: int | None = None,
) -> FloatArray:
    """
    Return a specific angular power spectrum from an array in
    :ref:`standard order <twopoint_order>`.

    Parameters
    ----------
    cls
        Angular matter power spectra in *GLASS* ordering.
    i
        Indices to return.
    j
        Indices to return.
    lmax
        Truncate the returned spectrum at this mode number.

    Returns
    -------
        The angular power spectrum for indices *i* and *j* from an
        array in *GLASS* ordering.

    """
    if j > i:
        i, j = j, i
    cl = cls[i * (i + 1) // 2 + i - j]
    if lmax is not None:
        if cl.size > lmax + 1:
            cl = cl[: lmax + 1]
        else:
            cl = xpx.pad(cl, (0, lmax + 1 - cl.size))
    return cl


def enumerate_spectra(
    entries: AngularPowerSpectra,
) -> Iterator[tuple[int, int, AnyArray]]:
    """
    Iterate over a set of two-point functions in :ref:`standard order
    <twopoint_order>`, yielding a tuple of indices and their associated
    entry from the input.

    Examples
    --------
    >>> spectra = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    >>> list(enumerate_spectra(spectra))
    [(0, 0, [1, 2, 3]), (1, 1, [4, 5, 6]), (1, 0, [7, 8, 9])]

    """
    for k, cl in enumerate(entries):
        i = int((2 * k + 0.25) ** 0.5 - 0.5)
        j = i * (i + 3) // 2 - k
        yield i, j, cl


def spectra_indices(n: int, *, xp: ModuleType | None = None) -> IntArray:
    """
    Return an array of indices in :ref:`standard order <twopoint_order>`
    for a set of two-point functions for *n* fields.  Each row is a pair
    of indices *i*, *j*.

    Examples
    --------
    >>> spectra_indices(3)
    array([[0, 0],
           [1, 1],
           [1, 0],
           [2, 2],
           [2, 1],
           [2, 0]])

    """
    xp = _utils.default_xp() if xp is None else xp

    i, j = xp.tril_indices(n)
    return xp.asarray([i, i - j]).T


def effective_cls(
    cls: AngularPowerSpectra,
    weights1: FloatArray,
    weights2: FloatArray | None = None,
    *,
    lmax: int | None = None,
) -> FloatArray:
    """
    Compute effective angular power spectra from weights.

    Computes a linear combination of the angular power spectra *cls*
    using the factors provided by *weights1* and *weights2*. Additional
    axes in *weights1* and *weights2* produce arrays of spectra.

    Parameters
    ----------
    cls
        Angular matter power spectra in *GLASS* ordering.
    weights1
        Weight factors for spectra. The first axis must be equal to
        the number of fields.
    weights2
        Second set of weights. If not given, *weights1* is used.
    lmax
        Truncate the angular power spectra at this mode number. If not
        given, the longest input in *cls* will be used.

    Returns
    -------
        A dictionary of effective angular power spectra, where keys
        correspond to the leading axes of *weights1* and *weights2*.

    Raises
    ------
    ValueError
        If the length of *cls* is not a triangle number.
    ValueError
        If the shapes of *weights1* and *weights2* are incompatible.

    """
    xp = array_api_compat.array_namespace(*cls, weights1, weights2, use_compat=False)
    uxpx = _utils.XPAdditions(xp)

    # this is the number of fields
    n = nfields_from_nspectra(len(cls))

    # find lmax if not given
    if lmax is None:
        lmax = max((cl.shape[0] for cl in cls), default=0) - 1

    # broadcast weights1 such that its shape ends in n
    weights1 = xp.asarray(weights1)
    weights2 = xp.asarray(weights2) if weights2 is not None else weights1

    shape1, shape2 = weights1.shape, weights2.shape
    for i, shape in enumerate((shape1, shape2)):
        if not shape or shape[0] != n:
            msg = f"shape mismatch between fields and weights{i + 1}"
            raise ValueError(msg)

    # get the iterator over leading weight axes
    # auto-spectra do not repeat identical computations
    pairs = (
        combinations_with_replacement(uxpx.ndindex(shape1[1:]), 2)
        if weights2 is weights1
        else product(uxpx.ndindex(shape1[1:]), uxpx.ndindex(shape2[1:]))
    )

    # create the output array: axes for all input axes plus lmax+1
    out = xp.empty(shape1[1:] + shape2[1:] + (lmax + 1,))

    # helper that will grab the entire first column (i.e. shells)
    c = (slice(None),)

    # compute all combined cls from pairs
    # if weights2 is weights1, set the transpose elements in one pass
    for j1, j2 in pairs:
        w1, w2 = weights1[c + j1], weights2[c + j2]
        cl = sum(
            w1[i1] * w2[i2] * getcl(cls, i1, i2, lmax=lmax)
            for i1 in range(n)
            for i2 in range(n)
        )
        out[j1 + j2 + (...,)] = cl
        if weights2 is weights1 and j1 != j2:
            out[j2 + j1 + (...,)] = cl
    return out


def gaussian_fields(
    shells: Sequence[glass.shells.RadialWindow],
) -> Sequence[glass.grf.Normal]:
    """
    Create Gaussian random fields for radial windows *shells*.

    Parameters
    ----------
    shells
        Window functions for the simulated shells.

    Returns
    -------
        A sequence describing the Gaussian random fields.

    """
    return [glass.grf.Normal() for _shell in shells]


def lognormal_fields(
    shells: Sequence[glass.shells.RadialWindow],
    shift: Callable[[float], float] | None = None,
) -> Sequence[glass.grf.Lognormal]:
    """
    Create lognormal fields for radial windows *shells*.  If *shifts* is
    given, it must be a callable that returns a lognormal shift (i.e.
    the scale parameter) at the nominal redshift of each shell.

    Parameters
    ----------
    shells
        Window functions for the simulated shells.
    shift
        Callable that returns the lognormal shift for each field.

    Returns
    -------
        A sequence describing the lognormal fields.

    """
    if shift is None:
        shift = lambda _z: 1.0  # noqa: E731

    return [glass.grf.Lognormal(shift(shell.zeff)) for shell in shells]


def compute_gaussian_spectra(
    fields: Sequence[glass.grf.Transformation],
    spectra: AngularPowerSpectra,
) -> AngularPowerSpectra:
    """
    Compute a sequence of Gaussian angular power spectra.

    After transformation by *fields*, the expected two-point statistics
    should recover *spectra* when using a band-limited transform
    [Tessore23]_.

    Parameters
    ----------
    fields
        The fields to be simulated.
    spectra
        The desired angular power spectra of the fields.

    Returns
    -------
        Gaussian angular power spectra for simulation.

    """
    n = len(fields)
    if len(spectra) != n * (n + 1) // 2:
        msg = "mismatch between number of fields and spectra"
        raise ValueError(msg)

    gls = []
    for i, j, cl in enumerate_spectra(spectra):
        gl = glass.grf.compute(cl, fields[i], fields[j]) if cl.size > 0 else 0 * cl
        gls.append(gl)
    return gls


def solve_gaussian_spectra(
    fields: Sequence[glass.grf.Transformation],
    spectra: AngularPowerSpectra,
) -> AngularPowerSpectra:
    """
    Solve a sequence of Gaussian angular power spectra.

    After transformation by *fields*, the expected two-point statistics
    should recover *spectra* when using a non-band-limited transform
    [Tessore23]_.

    Parameters
    ----------
    fields
        The fields to be simulated.
    spectra
        The desired angular power spectra of the fields.

    Returns
    -------
        Gaussian angular power spectra for simulation.

    """
    n = len(fields)
    if len(spectra) != n * (n + 1) // 2:
        msg = "mismatch between number of fields and spectra"
        raise ValueError(msg)

    gls = []
    for i, j, cl in enumerate_spectra(spectra):
        if cl.size > 0:
            # transformation pair
            t1, t2 = fields[i], fields[j]

            # set zero-padding of solver to 2N
            pad = 2 * cl.size

            # if the desired monopole is zero, that is most likely
            # and artefact of the theory spectra -- the variance of the
            # matter density in a finite shell is not zero
            # -> set output monopole to zero, which ignores cl[0]
            monopole = 0.0 if cl[0] == 0 else None

            # call solver
            gl, _cl_out, info = glass.grf.solve(cl, t1, t2, pad=pad, monopole=monopole)

            # warn if solver didn't converge
            if info == 0:
                warnings.warn(
                    f"Gaussian spectrum for fields ({i}, {j}) did not converge",
                    stacklevel=2,
                )
        else:
            gl = 0 * cl  # makes a copy of the empty array
        gls.append(gl)
    return gls


def generate(
    fields: Sequence[glass.grf.Transformation],
    gls: AngularPowerSpectra,
    nside: int,
    *,
    ncorr: int | None = None,
    rng: np.random.Generator | None = None,
) -> Iterator[AnyArray]:
    """
    Sample random fields from Gaussian angular power spectra.

    Iteratively sample HEALPix maps of transformed Gaussian random
    fields with the given angular power spectra *gls* and resolution
    parameter *nside*.

    The random fields are sampled from Gaussian random fields using the
    transformations in *fields*.

    The *gls* array must contain the angular power spectra of the
    Gaussian random fields in :ref:`standard order <twopoint_order>`.

    The optional number *ncorr* limits how many realised fields are
    correlated. This saves memory, as only *ncorr* previous fields are
    kept.

    Parameters
    ----------
    fields
        Transformations for the random fields.
    gls
        Gaussian angular power spectra.
    nside
        Resolution parameter for the HEALPix maps.
    ncorr
        Number of correlated fields. If not given, all fields are
        correlated.
    rng
        Random number generator. If not given, a default RNG is used.

    Yields
    ------
    x
        Sampled random fields.

    """
    n = len(fields)
    if len(gls) != n * (n + 1) // 2:
        msg = "mismatch between number of fields and gls"
        raise ValueError(msg)

    variances = (cltovar(getcl(gls, i, i)) for i in range(n))
    grf = _generate_grf(gls, nside, ncorr=ncorr, rng=rng)

    for t, x, var in zip(fields, grf, variances, strict=True):
        yield t(x, var)


def glass_to_healpix_spectra(spectra: Sequence[T]) -> list[T]:
    """
    Reorder spectra from GLASS to HEALPix order.

    Reorder spectra in :ref:`GLASS order <twopoint_order>` to conform to
    (new) HEALPix order.

    Parameters
    ----------
    spectra
        Sequence of spectra in GLASS order.

    Returns
    -------
        Sequence of spectra in HEALPix order.

    """
    n = nfields_from_nspectra(len(spectra))

    comb = {tuple(idx_tuple): pos for pos, idx_tuple in enumerate(spectra_indices(n))}
    return [spectra[comb[(i + k, i)]] for k in range(n) for i in range(n - k)]


def healpix_to_glass_spectra(spectra: Sequence[T]) -> list[T]:
    """
    Reorder spectra from HEALPix to GLASS order.

    Reorder HEALPix spectra (in new order) to conform to :ref:`GLASS
    order <twopoint_order>`.

    Parameters
    ----------
    spectra
        Sequence of spectra in HEALPix order.

    Returns
    -------
        Sequence of spectra in GLASS order.

    """
    n = nfields_from_nspectra(len(spectra))

    comb = [(i + k, i) for k in range(n) for i in range(n - k)]
    return [spectra[comb.index((i, j))] for i, j in spectra_indices(n)]  # type: ignore[arg-type]


def _glass_to_healpix_alm(alm: ComplexArray) -> ComplexArray:
    """
    Reorder alms in GLASS order to conform to (new) HEALPix order.

    Parameters
    ----------
    alm
        alm in GLASS order.

    Returns
    -------
        alm in HEALPix order.

    """
    xp = alm.__array_namespace__()

    n = _inv_triangle_number(alm.size)
    ell = xp.arange(n)
    out = [alm[ell[m:] * (ell[m:] + 1) // 2 + m] for m in ell]
    return xp.concat(out)


def lognormal_shift_hilbert2011(z: float) -> float:
    """
    Lognormal shift of Hilbert et al. (2011) for convergence fields.

    Lognormal shift parameter for the weak lensing convergence
    from the fitting formula of [Hilbert11]_.

    Parameters
    ----------
    z
        Redshift.

    Returns
    -------
        Lognormal shift.

    """
    return z * (8e-3 + z * (2.9e-2 + z * (-7.9e-3 + z * 6.5e-4)))


def cov_from_spectra(
    spectra: AngularPowerSpectra,
    *,
    lmax: int | None = None,
) -> AnyArray:
    """
    Construct covariance matrix from spectra.

    Construct a covariance matrix from the angular power spectra in
    *spectra*.

    Parameters
    ----------
    spectra
        Sequence of angular power spectra.
    lmax
        Maximum angular mode number. If not given, the maximum is taken
        from the provided spectra.

    Returns
    -------
        Covariance matrix from the given spectra.

    """
    # recover the number of fields from the number of spectra
    n = nfields_from_nspectra(len(spectra))

    # first case: maximum length in input spectra
    k = max((cl.size for cl in spectra), default=0) if lmax is None else lmax + 1

    # this is the covariance matrix of the spectra
    # the leading dimension is k, then it is a n-by-n covariance matrix
    # missing entries are zero, which is the default value
    cov = np.zeros((k, n, n))

    # fill the matrix up by going through the spectra in order
    # skip over entries that are None
    # if the spectra are ragged, some entries at high ell may remain zero
    # only fill the lower triangular part, everything is symmetric
    for i, j, cl in enumerate_spectra(spectra):
        cov[: cl.size, i, j] = cov[: cl.size, j, i] = cl.reshape(-1)[:k]  # type: ignore[union-attr]

    return cov


def check_posdef_spectra(spectra: AngularPowerSpectra) -> bool:
    """
    Test whether angular power spectra are positive semi-definite.

    Parameters
    ----------
    spectra
        Spectra to be tested.

    Returns
    -------
        Whether spectra are positive semi-definite or not.

    """
    cov = cov_from_spectra(spectra)
    xp = cov.__array_namespace__()
    is_positive_semi_definite: bool = xp.all(xp.linalg.eigvalsh(cov) >= 0)
    return is_positive_semi_definite


def regularized_spectra(
    spectra: AngularPowerSpectra,
    *,
    lmax: int | None = None,
    method: Literal["nearest", "clip"] = "nearest",
    **method_kwargs: float | None,
) -> AngularPowerSpectra:
    r"""
    Regularise a set of angular power spectra.

    Regularises a complete set *spectra* of angular power spectra in
    :ref:`standard order <twopoint_order>` such that at every angular
    mode number :math:`\ell`, the matrix :math:`C_\ell^{ij}` is a
    valid positive semi-definite covariance matrix.

    The length of the returned spectra is set by *lmax*, or the maximum
    length of the given spectra if *lmax* is not given.  Shorter input
    spectra are padded with zeros as necessary.  Missing spectra can be
    set to :data:`None` or, preferably, an empty array.

    The *method* parameter determines how the regularisation is carried
    out.

    Parameters
    ----------
    spectra
        Spectra to be regularised.
    lmax
        Maximum angular mode number. If not given, the maximum is taken
        from the provided spectra.
    method
        Regularisation method.

    Returns
    -------
        Regularised angular power spectra.

    """
    # regularise the cov matrix using the chosen method
    cov_method: Callable[..., AnyArray]
    if method == "clip":
        from glass.algorithm import cov_clip as cov_method  # noqa: PLC0415
    elif method == "nearest":
        from glass.algorithm import cov_nearest as cov_method  # noqa: PLC0415
    else:
        msg = f"unknown method '{method}'"  # type: ignore[unreachable]
        raise ValueError(msg)

    # get the cov matrix from spectra
    cov = cov_from_spectra(spectra, lmax=lmax)

    # regularise the cov matrix using the chosen method
    cov = cov_method(cov, **method_kwargs)

    # return regularised spectra from cov matrix array
    return [cov[:, i, j] for i, j in spectra_indices(cov.shape[-1])]
