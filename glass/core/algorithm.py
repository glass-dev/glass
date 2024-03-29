# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''core module for algorithms'''

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def nnls(
    a: ArrayLike,
    b: ArrayLike,
    *,
    tol: float = 0.0,
    maxiter: int | None = None,
) -> ArrayLike:
    """Compute a non-negative least squares solution.

    Implementation of the algorithm due to [1]_ as described in [2]_.

    References
    ----------
    .. [1] Lawson, C. L. and Hanson, R. J. (1995), Solving Least Squares
        Problems. doi: 10.1137/1.9781611971217
    .. [2] Bro, R. and De Jong, S. (1997), A fast
        non-negativity-constrained least squares algorithm. J.
        Chemometrics, 11, 393-401.

    """

    a = np.asanyarray(a)
    b = np.asanyarray(b)

    if a.ndim != 2:
        raise ValueError("input `a` is not a matrix")
    if b.ndim != 1:
        raise ValueError("input `b` is not a vector")
    if a.shape[0] != b.shape[0]:
        raise ValueError("the shapes of `a` and `b` do not match")

    _, n = a.shape

    if maxiter is None:
        maxiter = 3 * n

    index = np.arange(n)
    p = np.full(n, False)
    x = np.zeros(n)
    for i in range(maxiter):
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
            t = (sp <= 0)
            if not np.any(t):
                break
            alpha = -np.min(xp[t]/(xp[t] - sp[t]))
            x[p] += alpha * (sp - xp)
            p[x <= 0] = False
        x[p] = sp
        x[~p] = 0
    return x
