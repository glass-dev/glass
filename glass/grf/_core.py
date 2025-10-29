from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import transformcl

if TYPE_CHECKING:
    from types import NotImplementedType

    from glass._types import AnyArray


class Transformation(Protocol):
    """Protocol for transformations of Gaussian random fields."""

    def __call__(self, x: AnyArray, var: float, /) -> AnyArray:
        """
        Transform a Gaussian random field *x* with variance *var*.

        Parameters
        ----------
        x
            The Gaussian random field to be transformed.
        var
            Variance of the Gaussian random field.

        Returns
        -------
            The transformed Gaussian random field.

        """

    def corr(
        self, other: Transformation, x: AnyArray, /
    ) -> AnyArray | NotImplementedType:
        """Implementation of the corr function."""

    def icorr(
        self, other: Transformation, x: AnyArray, /
    ) -> AnyArray | NotImplementedType:
        """Implementation of the icorr function."""

    def dcorr(
        self, other: Transformation, x: AnyArray, /
    ) -> AnyArray | NotImplementedType:
        """Implementation of the dcorr function."""


def corr(t1: Transformation, t2: Transformation, x: AnyArray, /) -> AnyArray:
    """
    Transform a Gaussian angular correlation function.

    Parameters
    ----------
    t1, t2
        Transformations of the Gaussian random field.
    x
        The Gaussian angular correlation function.

    Returns
    -------
        The transformed angular correlation function.

    """
    result = t1.corr(t2, x)
    if result is not NotImplemented:
        return result
    result = t2.corr(t1, x)
    if result is not NotImplemented:
        return result
    msg = f"{t1.__class__.__name__} x {t2.__class__.__name__}"
    raise NotImplementedError(msg)


def icorr(t1: Transformation, t2: Transformation, x: AnyArray, /) -> AnyArray:
    """
    Inverse-transform an angular correlation function.

    Parameters
    ----------
    t1, t2
        Transformations of the Gaussian random field.
    x
        The transformed angular correlation function.

    Returns
    -------
        The Gaussian angular correlation function.

    """
    result = t1.icorr(t2, x)
    if result is not NotImplemented:
        return result
    result = t2.icorr(t1, x)
    if result is not NotImplemented:
        return result
    msg = f"{t1.__class__.__name__} x {t2.__class__.__name__}"
    raise NotImplementedError(msg)


def dcorr(t1: Transformation, t2: Transformation, x: AnyArray, /) -> AnyArray:
    """
    Derivative of the angular correlation function transform.

    Parameters
    ----------
    t1, t2
        Transformations of the Gaussian random field.
    x
        The Gaussian angular correlation function.

    Returns
    -------
        The derivative of the transformed angular correlation function.

    """
    result = t1.dcorr(t2, x)
    if result is not NotImplemented:
        return result
    result = t2.dcorr(t1, x)
    if result is not NotImplemented:
        return result
    msg = f"{t1.__class__.__name__} x {t2.__class__.__name__}"
    raise NotImplementedError(msg)


def compute(
    cl: AnyArray, t1: Transformation, t2: Transformation | None = None
) -> AnyArray:
    """
    Compute a band-limited Gaussian angular power spectrum for the
    target spectrum *cl* and the transformations *t1* and *t2*.  If *t2*
    is not given, it is assumed to be the same as *t1*.

    Parameters
    ----------
    cl
        The angular power spectrum after the transformations.
    t1, t2
        Transformations applied to the Gaussian random field(s).

    Returns
    -------
        Gaussian angular power spectrum.

    Examples
    --------
    Compute a Gaussian angular power spectrum ``gl`` for a lognormal
    transformation::

        t = glass.grf.Lognormal()
        gl = glass.grf.compute(cl, t)

    See Also
    --------
    glass.grf.solve: Iterative solver for non-band-limited spectra.

    """
    if t2 is None:
        t2 = t1

    # transform C_l to C(\theta), apply transformation, and transform back
    return transformcl.corrtocl(icorr(t1, t2, transformcl.cltocorr(cl)))
