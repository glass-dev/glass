from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Protocol, TypeVar

import transformcl

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from numpy.typing import NDArray

    TransformationT = TypeVar("TransformationT", bound="Transformation")


class Transformation(Protocol):
    """Protocol for transformations of Gaussian random fields."""

    def __call__(self, x: NDArray[Any], var: float, /) -> NDArray[Any]:
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


class _Dispatch(Protocol):
    """Protocol for the result of dispatch()."""

    def __call__(
        self, t1: Transformation, t2: Transformation, x: NDArray[Any], /
    ) -> NDArray[Any]: ...

    def add(
        self,
        impl: Callable[[TransformationT, TransformationT, NDArray[Any]], NDArray[Any]],
    ) -> Callable[[TransformationT, TransformationT, NDArray[Any]], NDArray[Any]]: ...


def dispatch(
    func: Callable[[Transformation, Transformation, NDArray[Any]], NDArray[Any]],
) -> _Dispatch:
    """Create a simple dispatcher for transformation pairs."""
    outer = functools.singledispatch(func)
    dispatch = outer.dispatch
    register = outer.register

    def add(
        impl: Callable[[TransformationT, TransformationT, NDArray[Any]], NDArray[Any]],
    ) -> Callable[[TransformationT, TransformationT, NDArray[Any]], NDArray[Any]]:
        from inspect import signature  # noqa: PLC0415
        from typing import get_type_hints  # noqa: PLC0415

        sig = signature(impl)
        if len(sig.parameters) != 3:
            msg = "invalid signature"
            raise TypeError(msg)
        par1, par2, _ = sig.parameters.values()
        if par1.annotation is par1.empty or par2.annotation is par2.empty:
            msg = "invalid signature"
            raise TypeError(msg)
        a, b, *_ = get_type_hints(impl).values()

        inner_a = dispatch(a)
        inner_b = dispatch(b)

        if inner_a is func:
            inner_a = register(a, functools.singledispatch(func))
        if inner_b is func:
            inner_b = register(b, functools.singledispatch(func))

        inner_a.register(b, impl)  # type: ignore[attr-defined]
        inner_b.register(a, lambda t2, t1, x: impl(t1, t2, x))  # type: ignore[attr-defined]

        return impl

    @functools.wraps(func)
    def wrapper(
        t1: Transformation,
        t2: Transformation,
        x: NDArray[Any],
    ) -> NDArray[Any]:
        impl = dispatch(type(t1))
        if impl is not func:
            impl = impl.dispatch(type(t2))  # type: ignore[attr-defined]
        return impl(t1, t2, x)

    wrapper.add = add  # type: ignore[attr-defined]
    return wrapper  # type: ignore[return-value]


@dispatch
def corr(t1: Transformation, t2: Transformation, x: NDArray[Any], /) -> NDArray[Any]:
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
    msg = f"{t1.__class__.__name__} x {t2.__class__.__name__}"
    raise NotImplementedError(msg)


@dispatch
def icorr(t1: Transformation, t2: Transformation, x: NDArray[Any], /) -> NDArray[Any]:
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
    msg = f"{t1.__class__.__name__} x {t2.__class__.__name__}"
    raise NotImplementedError(msg)


@dispatch
def dcorr(t1: Transformation, t2: Transformation, x: NDArray[Any], /) -> NDArray[Any]:
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
    msg = f"{t1.__class__.__name__} x {t2.__class__.__name__}"
    raise NotImplementedError(msg)


def compute(
    cl: NDArray[Any],
    t1: Transformation,
    t2: Transformation | None = None,
) -> NDArray[Any]:
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
    return transformcl.corrtocl(icorr(t1, t2, transformcl.cltocorr(cl)))  # type: ignore[no-any-return]
