"""Functions for random fields."""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from itertools import combinations_with_replacement, product
from typing import TYPE_CHECKING

import healpy as hp
import numpy as np
from transformcl import cltovar

import glass
import glass.grf

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable, Iterator, Sequence
    from typing import Any, Literal

    from numpy.typing import NDArray

    Fields = Sequence[glass.grf.Transformation]
    Cls = Sequence[NDArray[Any]]

try:
    from warnings import deprecated  # type: ignore[attr-defined]
except ImportError:
    if TYPE_CHECKING:
        from typing import ParamSpec, TypeVar

        _P = ParamSpec("_P")
        _R = TypeVar("_R")

    def deprecated(msg: str, /) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
        """Backport of Python's warnings.deprecated()."""
        from functools import wraps
        from warnings import warn

        def decorator(func: Callable[_P, _R], /) -> Callable[_P, _R]:
            @wraps(func)
            def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
                warn(msg, category=DeprecationWarning, stacklevel=2)
                return func(*args, **kwargs)

            return wrapper

        return decorator


def nfields_from_nspectra(nspectra: int) -> int:
    r"""
    Returns the number of fields for a number of spectra.

    Given the number of spectra *nspectra*, returns the number of
    fields *n* such that ``n * (n + 1) // 2 == nspectra``.
    """
    n = math.floor(math.sqrt(2 * nspectra))
    if n * (n + 1) // 2 != nspectra:
        msg = f"invalid number of spectra: {nspectra}"
        raise ValueError(msg)
    return n


def iternorm(
    k: int,
    cov: Iterable[NDArray[np.float64]],
    size: int | tuple[int, ...] = (),
) -> Generator[tuple[int | None, NDArray[np.float64], NDArray[np.float64]]]:
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
    n = (size,) if isinstance(size, int) else size

    m = np.zeros((*n, k, k))
    a = np.zeros((*n, k))
    s = np.zeros((*n,))
    q = (*n, k + 1)
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
            a = a.reshape(*n, k)

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
    cls: Cls,
    nl: int,
    nf: int,
    nc: int,
) -> Generator[NDArray[np.float64]]:
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
    alm: NDArray[np.complex128],
    bl: NDArray[np.float64],
    *,
    inplace: bool = False,
) -> NDArray[np.complex128]:
    """
    Multiply alm by bl.

    Parameters
    ----------
    alm
        The alm to multiply.
    bl
        The bl to multiply.
    inplace
        Whether to perform the operation in place.

    Returns
    -------
        The product of alm and bl.

    """
    n = len(bl)
    out = np.asanyarray(alm) if inplace else np.copy(alm)
    for m in range(n):
        out[m * n - m * (m - 1) // 2 : (m + 1) * n - m * (m + 1) // 2] *= bl[m:]
    return out


def discretized_cls(
    cls: Cls,
    *,
    lmax: int | None = None,
    ncorr: int | None = None,
    nside: int | None = None,
) -> Cls:
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
    if ncorr is not None:
        n = nfields_from_nspectra(len(cls))
        cls = [
            cls[i * (i + 1) // 2 + j] if j <= ncorr else np.asarray([])
            for i in range(n)
            for j in range(i + 1)
        ]

    if nside is not None:
        pw = hp.pixwin(nside, lmax=lmax)

    gls = []
    for cl in cls:
        if len(cl) > 0:
            if lmax is not None:
                cl = cl[: lmax + 1]  # noqa: PLW2901
            if nside is not None:
                n = min(len(cl), len(pw))
                cl = cl[:n] * pw[:n] ** 2  # noqa: PLW2901
        gls.append(cl)
    return gls


@deprecated("use glass.solve_gaussian_spectra() instead")
def lognormal_gls(
    cls: Cls,
    shift: float = 1.0,
) -> Cls:
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
    gls: Cls,
    nside: int,
    *,
    ncorr: int | None = None,
    rng: np.random.Generator | None = None,
) -> Generator[NDArray[np.float64]]:
    """
    Iteratively sample Gaussian random fields (internal use).

    A generator that iteratively samples HEALPix maps of Gaussian random fields
    with the given angular power spectra ``gls`` and resolution parameter
    ``nside``.

    The optional argument ``ncorr`` can be used to artificially limit now many
    realised fields are correlated. This saves memory, as only `ncorr` previous
    fields need to be kept.

    The ``gls`` array must contain the angular power power spectra of the
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
        rng = np.random.default_rng()

    # number of gls and number of fields
    ngls = len(gls)
    ngrf = nfields_from_nspectra(ngls)

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
    conditional_dist = iternorm(ncorr, cov, size=n)

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


@deprecated("use glass.generate() instead")
def generate_gaussian(
    gls: Cls,
    nside: int,
    *,
    ncorr: int | None = None,
    rng: np.random.Generator | None = None,
) -> Generator[NDArray[np.float64]]:
    """
    Sample Gaussian random fields from Cls iteratively.

    .. deprecated:: 2025.1

       Use :func:`glass.generate()` instead.

    A generator that iteratively samples HEALPix maps of Gaussian random fields
    with the given angular power spectra ``gls`` and resolution parameter
    ``nside``.

    The optional argument ``ncorr`` can be used to artificially limit now many
    realised fields are correlated. This saves memory, as only `ncorr` previous
    fields need to be kept.

    The ``gls`` array must contain the angular power power spectra of the
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
    gls: Cls,
    nside: int,
    shift: float = 1.0,
    *,
    ncorr: int | None = None,
    rng: np.random.Generator | None = None,
) -> Generator[NDArray[np.float64]]:
    """
    Sample lognormal random fields from Gaussian Cls iteratively.

    .. deprecated:: 2025.1

       Use :func:`glass.generate()` instead.

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
    cls: Sequence[NDArray[np.float64] | Sequence[float]],
    i: int,
    j: int,
    lmax: int | None = None,
) -> NDArray[np.float64] | Sequence[float]:
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
        if len(cl) > lmax + 1:
            cl = cl[: lmax + 1]
        else:
            cl = np.pad(cl, (0, lmax + 1 - len(cl)))
    return cl


def enumerate_spectra(
    entries: Iterable[NDArray[Any]],
) -> Iterator[tuple[int, int, NDArray[Any]]]:
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


def spectra_indices(n: int) -> NDArray[np.integer]:
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
    i, j = np.tril_indices(n)
    return np.transpose([i, i - j])


def effective_cls(
    cls: Sequence[NDArray[np.float64] | Sequence[float]],
    weights1: NDArray[np.float64],
    weights2: NDArray[np.float64] | None = None,
    *,
    lmax: int | None = None,
) -> NDArray[np.float64]:
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
    # this is the number of fields
    n = nfields_from_nspectra(len(cls))

    # find lmax if not given
    if lmax is None:
        lmax = max(map(len, cls), default=0) - 1

    # broadcast weights1 such that its shape ends in n
    weights1 = np.asanyarray(weights1)
    weights2 = np.asanyarray(weights2) if weights2 is not None else weights1

    shape1, shape2 = weights1.shape, weights2.shape
    for i, shape in enumerate((shape1, shape2)):
        if not shape or shape[0] != n:
            msg = f"shape mismatch between fields and weights{i + 1}"
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


def gaussian_fields(shells: Sequence[glass.RadialWindow]) -> Sequence[glass.grf.Normal]:
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
    shells: Sequence[glass.RadialWindow],
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


def compute_gaussian_spectra(fields: Fields, spectra: Cls) -> Cls:
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
        gl = glass.grf.compute(cl, fields[i], fields[j])
        gls.append(gl)
    return gls


def solve_gaussian_spectra(fields: Fields, spectra: Cls) -> Cls:
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
    fields: Fields,
    gls: Cls,
    nside: int,
    *,
    ncorr: int | None = None,
    rng: np.random.Generator | None = None,
) -> Iterator[NDArray[Any]]:
    """
    Sample random fields from Gaussian angular power spectra.

    Iteratively sample HEALPix maps of transformed Gaussian random
    fields with the given angular power spectra *gls* and resolution
    parameter *nside*.

    The random fields are sampled from Gaussian random fields using the
    transformations in *fields*.

    The *gls* array must contain the angular power power spectra of the
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


def glass_to_healpix_spectra(spectra: Cls) -> Cls:
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

    comb = [(i, j) for i, j in spectra_indices(n)]
    return [spectra[comb.index((i + k, i))] for k in range(n) for i in range(n - k)]


def healpix_to_glass_spectra(spectra: Cls) -> Cls:
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
    return [spectra[comb.index((i, j))] for i, j in spectra_indices(n)]


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
    return z * (0.008 + z * (0.029 + z * (-0.0079 + z * 0.00065)))


def cov_from_spectra(spectra: Cls, *, lmax: int | None = None) -> NDArray[Any]:
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

    if lmax is None:  # noqa: SIM108
        # maximum length in input spectra
        k = max((cl.size for cl in spectra), default=0)
    else:
        k = lmax + 1

    # this is the covariance matrix of the spectra
    # the leading dimension is k, then it is a n-by-n covariance matrix
    # missing entries are zero, which is the default value
    cov = np.zeros((k, n, n))

    # fill the matrix up by going through the spectra in order
    # skip over entries that are None
    # if the spectra are ragged, some entries at high ell may remain zero
    # only fill the lower triangular part, everything is symmetric
    for i, j, cl in enumerate_spectra(spectra):
        cov[: cl.size, i, j] = cov[: cl.size, j, i] = cl.reshape(-1)[:k]

    return cov


def check_posdef_spectra(spectra: Cls) -> bool:
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
    spectra: Cls,
    *,
    lmax: int | None = None,
    method: Literal["nearest", "clip"] = "nearest",
    **method_kwargs: float | None,
) -> Cls:
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

    """
    # regularise the cov matrix using the chosen method
    cov_method: Callable[..., NDArray[Any]]
    if method == "clip":
        from glass.algorithm import cov_clip as cov_method
    elif method == "nearest":
        from glass.algorithm import cov_nearest as cov_method
    else:
        msg = f"unknown method '{method}'"  # type: ignore[unreachable]
        raise ValueError(msg)

    # get the cov matrix from spectra
    cov = cov_from_spectra(spectra, lmax=lmax)

    # regularise the cov matrix using the chosen method
    cov = cov_method(cov, **method_kwargs)

    # return regularised spectra from cov matrix array
    return [cov[:, i, j] for i, j in spectra_indices(cov.shape[-1])]
