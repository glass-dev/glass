from __future__ import annotations

from dataclasses import dataclass

from ._core import ArrayT, corr, dcorr, icorr


@dataclass
class Normal:
    """Transformation for normal fields."""

    def __call__(self, x: ArrayT, _var: float, /) -> ArrayT:
        """Return *x* unchanged."""
        return x


@dataclass
class Lognormal:
    """Transformation for lognormal fields."""

    lamda: float = 1.0

    def __call__(self, x: ArrayT, var: float, /) -> ArrayT:
        """Transform *x* into a lognormal field."""
        xp = x.__array_namespace__()
        x = xp.expm1(x - var / 2)
        if self.lamda != 1.0:
            x = self.lamda * x
        return x


@dataclass
class SquaredNormal:
    """
    Transformation for squared normal fields.

    References
    ----------
    .. [1] https://arxiv.org/abs/2408.16903

    """

    a: float
    lamda: float = 1.0

    def __call__(self, x: ArrayT, _var: float, /) -> ArrayT:
        """Transform *x* into a squared normal field."""
        x = (x - self.a) ** 2 - 1
        if self.lamda != 1.0:
            x = self.lamda * x
        return x


########################################################################
# normal x normal
########################################################################


@corr.add
def _(_t1: Normal, _t2: Normal, x: ArrayT, /) -> ArrayT:
    return x


@icorr.add
def _(_t1: Normal, _t2: Normal, x: ArrayT, /) -> ArrayT:
    return x


@dcorr.add
def _(_t1: Normal, _t2: Normal, x: ArrayT, /) -> ArrayT:
    return 1.0 + (0 * x)


########################################################################
# lognormal x lognormal
########################################################################


@corr.add
def _(t1: Lognormal, t2: Lognormal, x: ArrayT, /) -> ArrayT:
    xp = x.__array_namespace__()
    return t1.lamda * t2.lamda * xp.expm1(x)


@icorr.add
def _(t1: Lognormal, t2: Lognormal, x: ArrayT, /) -> ArrayT:
    xp = x.__array_namespace__()
    return xp.log1p(x / (t1.lamda * t2.lamda))


@dcorr.add
def _(t1: Lognormal, t2: Lognormal, x: ArrayT, /) -> ArrayT:
    xp = x.__array_namespace__()
    return t1.lamda * t2.lamda * xp.exp(x)


########################################################################
# lognormal x normal
########################################################################


@corr.add
def _(t1: Lognormal, _t2: Normal, x: ArrayT, /) -> ArrayT:
    return t1.lamda * x


@icorr.add
def _(t1: Lognormal, _t2: Normal, x: ArrayT, /) -> ArrayT:
    return x / t1.lamda


@dcorr.add
def _(t1: Lognormal, _t2: Normal, x: ArrayT, /) -> ArrayT:
    return t1.lamda + (0.0 * x)


########################################################################
# squared normal x squared normal
########################################################################


@corr.add
def _(t1: SquaredNormal, t2: SquaredNormal, x: ArrayT, /) -> ArrayT:
    aa = t1.a * t2.a
    ll = t1.lamda * t2.lamda
    return 2 * ll * x * (x + 2 * aa)


@icorr.add
def _(t1: SquaredNormal, t2: SquaredNormal, x: ArrayT, /) -> ArrayT:
    xp = x.__array_namespace__()
    aa = t1.a * t2.a
    ll = t1.lamda * t2.lamda
    return xp.sqrt(x / (2 * ll) + aa**2) - aa


@dcorr.add
def _(t1: SquaredNormal, t2: SquaredNormal, x: ArrayT, /) -> ArrayT:
    aa = t1.a * t2.a
    ll = t1.lamda * t2.lamda
    return 4 * ll * (x + aa)
