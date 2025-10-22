"""Module for algorithms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import glass._array_api_utils as _utils

if TYPE_CHECKING:
    import numpy as np
    from jaxtyping import Array
    from numpy.typing import NDArray

    from glass._array_api_utils import FloatArray


def nnls(
    a: FloatArray,
    b: FloatArray,
    *,
    tol: float = 0.0,
    maxiter: int | None = None,
) -> FloatArray:
    """
    Compute a non-negative least squares solution.

    Implementation of the algorithm due to [Lawson95]_ as described by
    [Bro97]_.

    Parameters
    ----------
    a
        The matrix.
    b
        The vector.
    tol
        The tolerance for convergence.
    maxiter
        The maximum number of iterations.

    Returns
    -------
        The non-negative least squares solution.

    Raises
    ------
    ValueError
        If ``a`` is not a matrix.
    ValueError
        If ``b`` is not a vector.
    ValueError
        If the shapes of ``a`` and ``b`` do not match.

    """
    xp = _utils.get_namespace(a, b)

    a = xp.asarray(a)
    b = xp.asarray(b)

    if a.ndim != 2:
        msg = "input `a` is not a matrix"
        raise ValueError(msg)
    if b.ndim != 1:
        msg = "input `b` is not a vector"
        raise ValueError(msg)
    if a.shape[0] != b.shape[0]:
        msg = "the shapes of `a` and `b` do not match"
        raise ValueError(msg)

    _, n = a.shape

    if maxiter is None:
        maxiter = 3 * n

    index = xp.arange(n)
    q = xp.full(n, fill_value=False)
    x = xp.zeros(n)
    for _ in range(maxiter):
        if xp.all(q):
            break
        # The sum product over the last axis of arg1 and the second-to-last axis of arg2
        w = xp.sum((b - a @ x)[..., None] * a, axis=-2)

        m = int(index[~q][xp.argmax(w[~q])])
        if w[m] <= tol:
            break
        q[m] = True
        while True:
            # Use `xp.task`` here instead of `a[:,q]` to mask the inner arrays, because
            # array-api requires a masking index to be the sole index, which would
            # return a 1-D array. However, we want to maintain the shape of `a`,
            # i.e. `[[a11],[a12],...]` rather than `[a11,a12,...]`
            aq = xp.take(a, xp.nonzero(q)[0], axis=1)
            xq = x[q]
            sq = xp.linalg.solve(aq.T @ aq, b @ aq)
            t = sq <= 0
            if not xp.any(t):
                break
            alpha = -xp.min(xq[t] / (xq[t] - sq[t]))
            x[q] += alpha * (sq - xq)
            q[x <= 0] = False
        x[q] = sq
        x[~q] = 0
    return x


def cov_clip(
    cov: NDArray[np.float64] | Array,
    rtol: float | None = None,
) -> NDArray[np.float64] | Array:
    """
    Covariance matrix from clipping non-positive eigenvalues.

    The relative tolerance *rtol* is defined as for
    :func:`~array_api.linalg.matrix_rank`.

    Parameter
    ---------
    cov
        A symmetric matrix (or a stack of matrices).
    rtol
        An optional relative tolerance for eigenvalues to be considered
        positive.

    Returns
    -------
        Covariance matrix with negative eigenvalues clipped.

    """
    xp = cov.__array_namespace__()

    # Hermitian eigendecomposition
    w, v = xp.linalg.eigh(cov)

    # get tolerance if not given
    if rtol is None:
        rtol = max(v.shape[-2], v.shape[-1]) * xp.finfo(w.dtype).eps

    # clip negative diagonal values
    w = xp.clip(w, rtol * xp.max(w, axis=-1, keepdims=True), None)

    # put matrix back together
    # enforce symmetry
    v = xp.sqrt(w[..., None, :]) * v
    return xp.matmul(v, xp.matrix_transpose(v))


def nearcorr(
    a: NDArray[np.float64] | Array,
    *,
    tol: float | None = None,
    niter: int = 100,
) -> NDArray[np.float64] | Array:
    """
    Compute the nearest correlation matrix.

    Returns the nearest correlation matrix using the alternating
    projections algorithm of [Higham02]_.

    Parameters
    ----------
    a
        Square matrix (or a stack of square matrices).
    tol
        Tolerance for convergence. Default is dimension times machine
        epsilon.
    niter
        Maximum number of iterations.

    Returns
    -------
        Nearest correlation matrix.

    """
    xp = a.__array_namespace__()

    # shorthand for Frobenius norm
    frob = xp.linalg.matrix_norm

    # get size of the covariance matrix and flatten leading dimensions
    *dim, m, n = a.shape
    if m != n:
        msg = "non-square matrix"
        raise ValueError(msg)

    # default tolerance
    if tol is None:
        tol = n * xp.finfo(a.dtype).eps

    # current result, flatten leading dimensions
    y = xp.reshape(a, (-1, n, n))

    # initial correction is zero
    ds = xp.zeros_like(a)

    # store identity matrix
    diag = xp.eye(n)

    # find the nearest correlation matrix
    for _ in range(niter):
        # apply Dykstra's correction to current result
        r = y - ds

        # project onto positive semi-definite matrices
        x = cov_clip(r)

        # compute Dykstra's correction
        ds = x - r

        # project onto matrices with unit diagonal
        y = (1 - diag) * x + diag

        # check for convergence
        if xp.all(frob(y - x) <= tol * frob(y)):
            break

    # return result in original shape
    return xp.reshape(y, (*dim, n, n))


def cov_nearest(
    cov: NDArray[np.float64] | Array,
    tol: float | None = None,
    niter: int = 100,
) -> NDArray[np.float64] | Array:
    """
    Covariance matrix from nearest correlation matrix.

    Divides *cov* along rows and columns by the square root of the
    diagonal, then computes the nearest valid correlation matrix using
    :func:`nearcorr`, before scaling rows and columns back.  The
    diagonal of the input is hence unchanged.

    Parameters
    ----------
    cov
        A square matrix (or a stack of matrices).
    tol
        Tolerance for convergence, see :func:`nearcorr`.
    niter
        Maximum number of iterations.

    Returns
    -------
        Covariance matrix from nearest correlation matrix.

    """
    xp = cov.__array_namespace__()

    # get the diagonal
    diag = xp.linalg.diagonal(cov)

    # cannot fix negative diagonal
    if xp.any(diag < 0):
        msg = "negative values on the diagonal"
        raise ValueError(msg)

    # store the normalisation of the matrix
    norm = xp.sqrt(diag)
    norm = norm[..., None, :] * norm[..., :, None]

    # find nearest correlation matrix
    corr = cov / xp.where(norm > 0, norm, 1.0)
    return nearcorr(corr, niter=niter, tol=tol) * norm
