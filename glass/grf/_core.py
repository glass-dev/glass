from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Protocol, TypeVar

from transformcl import cltocorr, corrtocl

if TYPE_CHECKING:
    from collections.abc import Callable


ArrayT = TypeVar("ArrayT", bound="Array")
ArrayT_co = TypeVar("ArrayT_co", bound="Array", covariant=True)
TransformationT = TypeVar("TransformationT", bound="Transformation")


class ArrayNamespace(Protocol[ArrayT]):
    """Protocol for array namespaces."""

    pi: float

    def arange(self, n: int) -> ArrayT: ...

    def sqrt(self, x: ArrayT) -> ArrayT: ...
    def exp(self, x: ArrayT) -> ArrayT: ...
    def expm1(self, x: ArrayT) -> ArrayT: ...
    def log1p(self, x: ArrayT) -> ArrayT: ...


class Array(Protocol):
    """Protocol for arrays."""

    shape: tuple[int, ...]

    def __array_namespace__(self: ArrayT) -> ArrayNamespace[ArrayT]: ...

    def __add__(self: ArrayT, other: ArrayT | float) -> ArrayT: ...
    def __sub__(self: ArrayT, other: ArrayT | float) -> ArrayT: ...
    def __mul__(self: ArrayT, other: ArrayT | float) -> ArrayT: ...
    def __truediv__(self: ArrayT, other: ArrayT | float) -> ArrayT: ...
    def __pow__(self: ArrayT, other: ArrayT | float) -> ArrayT: ...

    def __radd__(self: ArrayT, other: ArrayT | float) -> ArrayT: ...
    def __rsub__(self: ArrayT, other: ArrayT | float) -> ArrayT: ...
    def __rmul__(self: ArrayT, other: ArrayT | float) -> ArrayT: ...
    def __rtruediv__(self: ArrayT, other: ArrayT | float) -> ArrayT: ...
    def __rpow__(self: ArrayT, other: ArrayT | float) -> ArrayT: ...


class Transformation(Protocol):
    """Protocol for transformations of Gaussian random fields."""

    def __call__(self, x: ArrayT, var: float, /) -> ArrayT:
        """Transform a Gaussian random field *x* with variance *var*."""


class Dispatch(Protocol):
    """Protocol for the result of dispatch()."""

    def __call__(
        self, t1: Transformation, t2: Transformation, x: ArrayT, /
    ) -> ArrayT: ...

    def add(
        self,
        impl: Callable[[TransformationT, TransformationT, ArrayT], ArrayT],
    ) -> Callable[[TransformationT, TransformationT, ArrayT], ArrayT]: ...


def dispatch(
    func: Callable[[Transformation, Transformation, ArrayT], ArrayT],
) -> Dispatch:
    """Create a simple dispatcher for transformation pairs."""
    outer = functools.singledispatch(func)
    dispatch = outer.dispatch
    register = outer.register

    def add(
        impl: Callable[[TransformationT, TransformationT, ArrayT], ArrayT],
    ) -> Callable[[TransformationT, TransformationT, ArrayT], ArrayT]:
        from inspect import signature
        from typing import get_type_hints

        sig = signature(impl)
        if len(sig.parameters) != 3:
            raise TypeError("invalid signature")
        par1, par2, _ = sig.parameters.values()
        if par1.annotation is par1.empty or par2.annotation is par2.empty:
            raise TypeError("invalid signature")
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
    def wrapper(t1: Transformation, t2: Transformation, x: ArrayT) -> ArrayT:
        inner = dispatch(type(t1))
        if inner is not func:
            impl = inner.dispatch(type(t2))  # type: ignore[attr-defined]
        else:
            impl = func
        return impl(t1, t2, x)  # type: ignore[no-any-return]

    wrapper.add = add  # type: ignore[attr-defined]
    return wrapper  # type: ignore[return-value]


@dispatch
def corr(t1: Transformation, t2: Transformation, x: ArrayT, /) -> ArrayT:
    """
    Transform a Gaussian angular correlation function.

    Parameters
    ----------
    t1, t2 : :class:`Transformation`
        Transformations of the Gaussian random field.
    x : array_like
        The Gaussian angular correlation function.

    Returns
    -------
    y : array_like
        The transformed angular correlation function.

    """
    msg = f"{t1.__class__.__name__} x {t2.__class__.__name__}"
    raise NotImplementedError(msg)


@dispatch
def icorr(t1: Transformation, t2: Transformation, x: ArrayT, /) -> ArrayT:
    """
    Inverse-transform an angular correlation function.

    Parameters
    ----------
    t1, t2 : :class:`Transformation`
        Transformations of the Gaussian random field.
    x : array_like
        The transformed angular correlation function.

    Returns
    -------
    y : array_like
        The Gaussian angular correlation function.

    """
    msg = f"{t1.__class__.__name__} x {t2.__class__.__name__}"
    raise NotImplementedError(msg)


@dispatch
def dcorr(t1: Transformation, t2: Transformation, x: ArrayT, /) -> ArrayT:
    """
    Derivative of the angular correlation function transform.

    Parameters
    ----------
    t1, t2 : :class:`Transformation`
        Transformations of the Gaussian random field.
    x : array_like
        The Gaussian angular correlation function.

    Returns
    -------
    y : array_like
        The derivative of the transformed angular correlation function.

    """
    msg = f"{t1.__class__.__name__} x {t2.__class__.__name__}"
    raise NotImplementedError(msg)


def compute(cl: ArrayT, t1: Transformation, t2: Transformation | None = None) -> ArrayT:
    """
    Compute a band-limited Gaussian angular power spectrum for the
    target spectrum *cl* and the transformations *t1* and *t2*.  If *t2*
    is not given, it is assumed to be the same as *t1*.

    Parameters
    ----------
    cl : array_like
        The angular power spectrum after the transformations.
    t1, t2 : :class:`Transformation`
        Transformations applied to the Gaussian random field(s).

    Returns
    -------
    gl : array_like
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
    return corrtocl(icorr(t1, t2, cltocorr(cl)))  # type: ignore[no-any-return]
