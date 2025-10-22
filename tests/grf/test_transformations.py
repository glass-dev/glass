

from typing import TYPE_CHECKING

import numpy as np

import glass.grf

if TYPE_CHECKING:
    import types

    from conftest import UnifiedGenerator


def test_normal(urng: UnifiedGenerator) -> None:
    t = glass.grf.Normal()
    x = urng.standard_normal(10)
    np.testing.assert_array_equal(t(x, 1.0), x)


def test_lognormal(xp: types.ModuleType, urng: UnifiedGenerator) -> None:
    for lam in 1.0, urng.uniform():
        var = urng.uniform()
        t = glass.grf.Lognormal(lam)
        x = urng.standard_normal(10)
        y = lam * xp.expm1(x - var / 2)
        np.testing.assert_array_equal(t(x, var), y)


def test_sqnormal(xp: types.ModuleType, urng: UnifiedGenerator) -> None:
    for lam in 1.0, urng.uniform():
        var = urng.uniform()
        a = xp.sqrt(1 - var)
        t = glass.grf.SquaredNormal(a, lam)
        x = urng.standard_normal(10)
        y = lam * ((x - a) ** 2 - 1)
        np.testing.assert_array_equal(t(x, var), y)


def test_normal_normal(xp: types.ModuleType, urng: UnifiedGenerator) -> None:
    t1 = glass.grf.Normal()
    t2 = glass.grf.Normal()
    x = urng.random(10)
    np.testing.assert_array_equal(glass.grf.corr(t1, t2, x), x)
    np.testing.assert_array_equal(glass.grf.icorr(t1, t2, x), x)
    np.testing.assert_array_equal(glass.grf.dcorr(t1, t2, x), xp.ones_like(x))


def test_lognormal_lognormal(xp: types.ModuleType, urng: UnifiedGenerator) -> None:
    lam1 = urng.uniform()
    t1 = glass.grf.Lognormal(lam1)

    lam2 = urng.uniform()
    t2 = glass.grf.Lognormal(lam2)

    x = urng.random(10)
    y = lam1 * lam2 * xp.expm1(x)
    dy = lam1 * lam2 * xp.exp(x)

    np.testing.assert_array_equal(glass.grf.corr(t1, t2, x), y)
    np.testing.assert_array_almost_equal_nulp(glass.grf.icorr(t1, t2, y), x)
    np.testing.assert_array_equal(glass.grf.dcorr(t1, t2, x), dy)


def test_lognormal_normal(xp: types.ModuleType, urng: UnifiedGenerator) -> None:
    lam1 = urng.uniform()
    t1 = glass.grf.Lognormal(lam1)

    t2 = glass.grf.Normal()

    x = urng.random(10)
    y = lam1 * x
    dy = lam1 * xp.ones_like(x)

    np.testing.assert_array_equal(glass.grf.corr(t1, t2, x), y)
    np.testing.assert_array_almost_equal_nulp(glass.grf.icorr(t1, t2, y), x)
    np.testing.assert_array_equal(glass.grf.dcorr(t1, t2, x), dy)


def test_sqnormal_sqnormal(xp: types.ModuleType, urng: UnifiedGenerator) -> None:
    lam1, var1 = urng.uniform(size=2)
    a1 = xp.sqrt(1 - var1)
    t1 = glass.grf.SquaredNormal(a1, lam1)

    lam2, var2 = urng.uniform(size=2)
    a2 = xp.sqrt(1 - var2)
    t2 = glass.grf.SquaredNormal(a2, lam2)

    # https://arxiv.org/pdf/2408.16903, (E.7)
    x = urng.random(10)
    y = 2 * lam1 * lam2 * x * (x + 2 * a1 * a2)
    dy = 4 * lam1 * lam2 * (x + a1 * a2)

    np.testing.assert_array_equal(glass.grf.corr(t1, t2, x), y)
    np.testing.assert_array_almost_equal_nulp(glass.grf.icorr(t1, t2, y), x, nulp=8)
    np.testing.assert_array_equal(glass.grf.dcorr(t1, t2, x), dy)
