from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import NotImplementedType

    from glass._types import AnyArray
    from glass.grf import Transformation


@dataclass
class Normal:
    """
    Transformation for normal fields.

    .. math::

       t(X) = X

    This is the identity transformation.

    """

    def __call__(self, x: AnyArray, _var: float, /) -> AnyArray:
        """Return *x* unchanged."""
        return x

    def corr(
        self, other: Transformation, x: AnyArray, /
    ) -> AnyArray | NotImplementedType:
        if type(other) is Normal:
            return x
        return NotImplemented

    def icorr(
        self, other: Transformation, x: AnyArray, /
    ) -> AnyArray | NotImplementedType:
        if type(other) is Normal:
            return x
        return NotImplemented

    def dcorr(
        self, other: Transformation, x: AnyArray, /
    ) -> AnyArray | NotImplementedType:
        if type(other) is Normal:
            return 1.0 + (0 * x)
        return NotImplemented


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

    def __call__(self, x: AnyArray, var: float, /) -> AnyArray:
        """Transform *x* into a lognormal field."""
        xp = x.__array_namespace__()
        x = xp.expm1(x - var / 2)
        if self.lamda != 1.0:
            x = self.lamda * x
        return x

    def corr(
        self, other: Transformation, x: AnyArray, /
    ) -> AnyArray | NotImplementedType:
        xp = x.__array_namespace__()

        if type(other) is Lognormal:
            return self.lamda * other.lamda * xp.expm1(x)

        if type(other) is Normal:
            return self.lamda * x

        return NotImplemented

    def icorr(
        self, other: Transformation, x: AnyArray, /
    ) -> AnyArray | NotImplementedType:
        xp = x.__array_namespace__()

        if type(other) is Lognormal:
            return xp.log1p(x / (self.lamda * other.lamda))

        if type(other) is Normal:
            return x / self.lamda

        return NotImplemented

    def dcorr(
        self, other: Transformation, x: AnyArray, /
    ) -> AnyArray | NotImplementedType:
        xp = x.__array_namespace__()

        if type(other) is Lognormal:
            return self.lamda * other.lamda * xp.exp(x)

        if type(other) is Normal:
            return self.lamda + (0.0 * x)

        return NotImplemented


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

    def __call__(self, x: AnyArray, _var: float, /) -> AnyArray:
        """Transform *x* into a squared normal field."""
        x = (x - self.a) ** 2 - 1
        if self.lamda != 1.0:
            x = self.lamda * x
        return x

    def corr(
        self, other: Transformation, x: AnyArray, /
    ) -> AnyArray | NotImplementedType:
        if type(other) is SquaredNormal:
            aa = self.a * other.a
            ll = self.lamda * other.lamda
            return 2 * ll * x * (x + 2 * aa)

        return NotImplemented

    def icorr(
        self, other: Transformation, x: AnyArray, /
    ) -> AnyArray | NotImplementedType:
        xp = x.__array_namespace__()

        if type(other) is SquaredNormal:
            aa = self.a * other.a
            ll = self.lamda * other.lamda
            return xp.sqrt(x / (2 * ll) + aa**2) - aa

        return NotImplemented

    def dcorr(
        self, other: Transformation, x: AnyArray, /
    ) -> AnyArray | NotImplementedType:
        if type(other) is SquaredNormal:
            aa = self.a * other.a
            ll = self.lamda * other.lamda
            return 4 * ll * (x + aa)

        return NotImplemented
