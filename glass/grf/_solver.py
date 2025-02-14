from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from transformcl import cltocorr, corrtocl

import glass.grf

if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import NDArray


def _relerr(dx: NDArray[Any], x: NDArray[Any]) -> float:
    """Compute the relative error max(|dx/x|)."""
    q = np.divide(dx, x, where=(dx != 0), out=np.zeros_like(dx))
    return np.fabs(q).max()  # type: ignore[no-any-return]


def solve(  # noqa: PLR0912, PLR0913
    cl: NDArray[Any],
    t1: glass.grf.Transformation,
    t2: glass.grf.Transformation | None = None,
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

    Given the input angular power spectrum *cl* and a pair of
    transformations *t1* and *t2*, computes a Gaussian angular spectrum
    that reproduces *cl* after the transformations are applied.  This is
    done using the iterative solver proposed in [Tessore23]_.

    The internal padding of the solver is set by *pad*, and the initial
    solution can be passed as *initial*.  The convergence is controlled
    by the relative errors *cltol* and *gltol*, and the maximum number
    of iterations *maxiter*.

    If *monopole* is provided, the monopole of the Gaussian angular
    power spectrum is fixed to that value and ignored by the solver.

    Returns a tuple *gl*, *cl*, *info* where *gl* is the Gaussian
    angular power spectrum solution, *cl* is the realised angular power
    spectrum after transformation, and *info* indicates success of
    failure of the solution.  Possible *info* values are

    * ``0``, solution did not converge in *maxiter* iterations;
    * ``1``, solution converged in *cl* relative error;
    * ``2``, solution converged in *gl* relative error;
    * ``3``, solution converged in both *cl* and *gl* relative error.

    Parameters
    ----------
    cl
        The angular power spectrum after the transformations.
    t1, t2
        Transformations applied to the Gaussian random field(s).
    pad
        Internal padding applied to the transforms.
    initial
        Initial solution. If not provided, uses the result of
        :func:`glass.grf.compute`.
    cltol
        Relative error for convergence in the desired output.
    gltol
        Relative error for convergence in the solution.
    maxiter
        Maximum number of iterations of the solver.
    monopole
        Fix the monopole of the solution to the given value.

    See Also
    --------
    glass.grf.compute : Direct computation for band-limited spectra.

    """
    if t2 is None:
        t2 = t1

    n = len(cl)
    if pad < 0:
        msg = "pad must be a positive integer"
        raise ValueError(msg)

    if initial is None:
        gl = corrtocl(glass.grf.icorr(t1, t2, cltocorr(cl)))
    else:
        gl = np.zeros(n)
        gl[: len(initial)] = initial[:n]

    if monopole is not None:
        gl[0] = monopole

    gt = cltocorr(np.pad(gl, (0, pad)))
    rl = corrtocl(glass.grf.corr(t1, t2, gt))
    fl = rl[:n] - cl
    if monopole is not None:
        fl[0] = 0
    clerr = _relerr(fl, cl)

    # this is a very basic Gauss-Newton solver
    # see https://arxiv.org/pdf/2302.01942 for details
    info = 0
    for _ in range(maxiter):
        if clerr <= cltol:
            info |= 1
        if info > 0:
            break

        ft = cltocorr(np.pad(fl, (0, pad)))
        dt = glass.grf.dcorr(t1, t2, gt)
        xl = -corrtocl(ft / dt)[:n]
        if monopole is not None:
            xl[0] = 0

        # we know the "direction" of the step xl at this point
        # make xl smaller (by half) until we get an improved solution
        while True:
            gl_ = gl + xl
            gt_ = cltocorr(np.pad(gl_, (0, pad)))
            rl_ = corrtocl(glass.grf.corr(t1, t2, gt_))
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
