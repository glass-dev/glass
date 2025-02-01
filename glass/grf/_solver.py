from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from transformcl import cltocorr, corrtocl

if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import NDArray

from ._core import Transformation, corr, dcorr, icorr


def _relerr(dx: NDArray[Any], x: NDArray[Any]) -> float:
    """Compute the relative error max(|dx/x|)."""
    q = np.divide(dx, x, where=(dx != 0), out=np.zeros_like(dx))
    return np.fabs(q).max()  # type: ignore[no-any-return]


def solve(
    cl: NDArray[Any],
    t1: Transformation,
    t2: Transformation | None = None,
    *,
    pad: int = 0,
    initial: NDArray[Any] | None = None,
    cltol: float = 1e-5,
    gltol: float = 1e-5,
    maxiter: int = 20,
    monopole: float | None = None,
) -> tuple[NDArray[Any], NDArray[Any], int]:
    """
    Solve for a Gaussian angular power spectrum.

    Parameters
    ----------
    cl : (n,) array
    t1, t2 : :class:`Transformation`

    Returns
    -------
    gl : (n,) array
        Gaussian angular power spectrum solution.
    cl : (n + pad,) array
        Realised transformed angular power spectrum.
    info : {0, 1, 2, 3}
        Indicates success of failure of the solution.  Possible values are

        * ``0``, solution did not converge in *maxiter* iterations;
        * ``1``, solution converged in *cl* relative error;
        * ``2``, solution converged in *gl* relative error;
        * ``3``, solution converged in both *cl* and *gl* relative error.

    See Also
    --------
    glass.grf.compute: Direct computation for band-limited spectra.

    References
    ----------
    .. [1] Tessore N., Loureiro A., Joachimi B., von Wietersheim-Kramsta M.,
           Jeffrey N., 2023, OJAp, 6, 11. doi:10.21105/astro.2302.01942


    """
    if t2 is None:
        t2 = t1

    n = len(cl)
    if pad < 0:
        raise TypeError("pad must be a positive integer")

    if initial is None:
        gl = corrtocl(icorr(t1, t2, cltocorr(cl)))
    else:
        gl = np.zeros(n)
        gl[: len(initial)] = initial[:n]

    if monopole is not None:
        gl[0] = monopole

    gt = cltocorr(np.pad(gl, (0, pad)))
    rl = corrtocl(corr(t1, t2, gt))
    fl = rl[:n] - cl
    if monopole is not None:
        fl[0] = 0
    clerr = _relerr(fl, cl)

    info = 0
    for _ in range(maxiter):
        if clerr <= cltol:
            info |= 1
        if info > 0:
            break

        ft = cltocorr(np.pad(fl, (0, pad)))
        dt = dcorr(t1, t2, gt)
        xl = -corrtocl(ft / dt)[:n]
        if monopole is not None:
            xl[0] = 0

        while True:
            gl_ = gl + xl
            gt_ = cltocorr(np.pad(gl_, (0, pad)))
            rl_ = corrtocl(corr(t1, t2, gt_))
            fl_ = rl_[:n] - cl
            if monopole is not None:
                fl_[0] = 0
            clerr_ = _relerr(fl_, cl)
            if clerr_ <= clerr:
                break
            xl /= 2

        if _relerr(xl, gl) <= gltol:
            info |= 2

        gl, gt, rl, fl, clerr = gl_, gt_, rl_, fl_, clerr_

    return gl, rl, info
