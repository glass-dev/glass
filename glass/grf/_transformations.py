from __future__ import annotations

from dataclasses import dataclass

# type imports for NDArray[Any] cannot be moved into TYPE_CHECKING here
# otherwise, the dispatch mechanism cannot resolve the class dynamically
from typing import Any

from numpy.typing import NDArray  # noqa: TC002

from array_api_strict._array_object import Array as AArray  # noqa: TC002

from glass.grf import corr, dcorr, icorr


@dataclass
class Normal:
    """
    Transformation for normal fields.

    .. math::

       t(X) = X

    This is the identity transformation.

    """

    def __call__(
        self,
        x: NDArray[Any] | AArray,
        _var: float,
        /,
    ) -> NDArray[Any] | AArray:
        """Return *x* unchanged."""
        return x


@dataclass
class Lognormal:
    r"""
    Transformation for lognormal fields.

    .. math::

       t(X) = \lambda \bigl[\exp(X - \tfrac{\sigma^2}{2}) - 1\bigr]

    The "lognormal shift" parameter :math:`\lambda` is the scale
    parameter of the distribution, with determines the smallest
    possible value :math:`-\lambda` of the lognormal field.

    Parameters
    ----------
    lamda
        The parameter :math:`\lambda`.

    """

    lamda: float = 1.0

    def __call__(
        self,
        x: NDArray[Any] | AArray,
        var: float,
        /,
    ) -> NDArray[Any] | AArray:
        """Transform *x* into a lognormal field."""
        xp = x.__array_namespace__()
        x = xp.expm1(x - var / 2)
        if self.lamda != 1.0:
            x = self.lamda * x
        return x


@dataclass
class SquaredNormal:
    r"""
    Transformation for squared normal fields as introduced by
    [Tessore25]_.

    .. math::

       t(X) = \lambda \, \bigl[(X - a)^2 - 1\bigr]

    The transformation is characterised by the shape parameter *a*,
    which is related to the variance :math:`\sigma^2` of the Gaussian
    random field, :math:`a = \sqrt{1 - \sigma^2}`.

    The "shift" parameter :math:`\lambda` is the scale parameter of the
    distribution, with determines the smallest possible value
    :math:`-\lambda` of the squared normal field.

    Parameters
    ----------
    a
        The parameter :math:`a`.
    lamda
        The parameter :math:`\lambda`.

    """

    a: float
    lamda: float = 1.0

    def __call__(
        self,
        x: NDArray[Any] | AArray,
        _var: float,
        /,
    ) -> NDArray[Any] | AArray:
        """Transform *x* into a squared normal field."""
        x = (x - self.a) ** 2 - 1
        if self.lamda != 1.0:
            x = self.lamda * x
        return x


########################################################################
# normal x normal
########################################################################


@corr.add
def _(_t1: Normal, _t2: Normal, x: NDArray[Any] | AArray, /) -> NDArray[Any] | AArray:
    return x


@icorr.add
def _(_t1: Normal, _t2: Normal, x: NDArray[Any] | AArray, /) -> NDArray[Any] | AArray:
    return x


@dcorr.add
def _(_t1: Normal, _t2: Normal, x: NDArray[Any] | AArray, /) -> NDArray[Any] | AArray:
    return 1.0 + (0 * x)


########################################################################
# lognormal x lognormal
########################################################################


@corr.add
def _(
    t1: Lognormal,
    t2: Lognormal,
    x: NDArray[Any] | AArray,
    /,
) -> NDArray[Any] | AArray:
    xp = x.__array_namespace__()
    return t1.lamda * t2.lamda * xp.expm1(x)


@icorr.add
def _(
    t1: Lognormal,
    t2: Lognormal,
    x: NDArray[Any] | AArray,
    /,
) -> NDArray[Any] | AArray:
    xp = x.__array_namespace__()
    return xp.log1p(x / (t1.lamda * t2.lamda))


@dcorr.add
def _(
    t1: Lognormal,
    t2: Lognormal,
    x: NDArray[Any] | AArray,
    /,
) -> NDArray[Any] | AArray:
    xp = x.__array_namespace__()
    return t1.lamda * t2.lamda * xp.exp(x)


########################################################################
# lognormal x normal
########################################################################


@corr.add
def _(t1: Lognormal, _t2: Normal, x: NDArray[Any] | AArray, /) -> NDArray[Any] | AArray:
    return t1.lamda * x


@icorr.add
def _(t1: Lognormal, _t2: Normal, x: NDArray[Any] | AArray, /) -> NDArray[Any] | AArray:
    return x / t1.lamda


@dcorr.add
def _(t1: Lognormal, _t2: Normal, x: NDArray[Any] | AArray, /) -> NDArray[Any] | AArray:
    return t1.lamda + (0.0 * x)


########################################################################
# squared normal x squared normal
########################################################################


@corr.add
def _(
    t1: SquaredNormal,
    t2: SquaredNormal,
    x: NDArray[Any] | AArray,
    /,
) -> NDArray[Any] | AArray:
    aa = t1.a * t2.a
    ll = t1.lamda * t2.lamda
    return 2 * ll * x * (x + 2 * aa)


@icorr.add
def _(
    t1: SquaredNormal,
    t2: SquaredNormal,
    x: NDArray[Any] | AArray,
    /,
) -> NDArray[Any] | AArray:
    xp = x.__array_namespace__()
    aa = t1.a * t2.a
    ll = t1.lamda * t2.lamda
    return xp.sqrt(x / (2 * ll) + aa**2) - aa


@dcorr.add
def _(
    t1: SquaredNormal,
    t2: SquaredNormal,
    x: NDArray[Any] | AArray,
    /,
) -> NDArray[Any] | AArray:
    aa = t1.a * t2.a
    ll = t1.lamda * t2.lamda
    return 4 * ll * (x + aa)
