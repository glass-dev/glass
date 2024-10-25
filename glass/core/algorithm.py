"""Core module for algorithms."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def nnls(
    a: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
    *,
    tol: float = 0.0,
    maxiter: int | None = None,
) -> npt.NDArray[np.float64]:
    """
    _summary_.

    Parameters
    ----------
    a
        _description_
    b
        _description_
    tol
        _description_
    maxiter
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
    ValueError
        _description_

    """
    a = np.asanyarray(a)
    b = np.asanyarray(b)

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

    index = np.arange(n)
    p = np.full(n, fill_value=False)
    x = np.zeros(n)
    for _ in range(maxiter):
        if np.all(p):
            break
        w = np.dot(b - a @ x, a)
        m = index[~p][np.argmax(w[~p])]
        if w[m] <= tol:
            break
        p[m] = True
        while True:
            ap = a[:, p]
            xp = x[p]
            sp = np.linalg.solve(ap.T @ ap, b @ ap)
            t = sp <= 0
            if not np.any(t):
                break
            alpha = -np.min(xp[t] / (xp[t] - sp[t]))
            x[p] += alpha * (sp - xp)
            p[x <= 0] = False
        x[p] = sp
        x[~p] = 0
    return x
