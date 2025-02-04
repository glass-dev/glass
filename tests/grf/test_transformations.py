import numpy as np

import glass.grf


def test_normal(rng):
    t = glass.grf.Normal()
    x = rng.standard_normal(10)
    np.testing.assert_array_equal(t(x, 1.0), x)


def test_lognormal(rng):
    for lam in 1.0, rng.uniform():
        var = rng.uniform()
        t = glass.grf.Lognormal(lam)
        x = rng.standard_normal(10)
        y = lam * np.expm1(x - var / 2)
        np.testing.assert_array_equal(t(x, var), y)


def test_sqnormal(rng):
    for lam in 1.0, rng.uniform():
        var = rng.uniform()
        a = np.sqrt(1 - var)
        t = glass.grf.SquaredNormal(a, lam)
        x = rng.standard_normal(10)
        y = lam * ((x - a) ** 2 - 1)
        np.testing.assert_array_equal(t(x, var), y)


def test_normal_normal(rng):
    t1 = glass.grf.Normal()
    t2 = glass.grf.Normal()
    x = rng.random(10)
    np.testing.assert_array_equal(glass.grf.corr(t1, t2, x), x)
    np.testing.assert_array_equal(glass.grf.icorr(t1, t2, x), x)
    np.testing.assert_array_equal(glass.grf.dcorr(t1, t2, x), np.ones_like(x))


def test_lognormal_lognormal(rng):
    lam1 = rng.uniform()
    t1 = glass.grf.Lognormal(lam1)

    lam2 = rng.uniform()
    t2 = glass.grf.Lognormal(lam2)

    x = rng.random(10)
    y = lam1 * lam2 * np.expm1(x)
    dy = lam1 * lam2 * np.exp(x)

    np.testing.assert_array_equal(glass.grf.corr(t1, t2, x), y)
    np.testing.assert_array_almost_equal_nulp(glass.grf.icorr(t1, t2, y), x)
    np.testing.assert_array_equal(glass.grf.dcorr(t1, t2, x), dy)


def test_lognormal_normal(rng):
    lam1 = rng.uniform()
    t1 = glass.grf.Lognormal(lam1)

    t2 = glass.grf.Normal()

    x = rng.random(10)
    y = lam1 * x
    dy = lam1 * np.ones_like(x)

    np.testing.assert_array_equal(glass.grf.corr(t1, t2, x), y)
    np.testing.assert_array_almost_equal_nulp(glass.grf.icorr(t1, t2, y), x)
    np.testing.assert_array_equal(glass.grf.dcorr(t1, t2, x), dy)


def test_sqnormal_sqnormal(rng):
    lam1, var1 = rng.uniform(size=2)
    a1 = np.sqrt(1 - var1)
    t1 = glass.grf.SquaredNormal(a1, lam1)

    lam2, var2 = rng.uniform(size=2)
    a2 = np.sqrt(1 - var2)
    t2 = glass.grf.SquaredNormal(a2, lam2)

    # https://arxiv.org/pdf/2408.16903, (E.7)
    x = rng.random(10)
    y = 2 * lam1 * lam2 * x * (x + 2 * a1 * a2)
    dy = 4 * lam1 * lam2 * (x + a1 * a2)

    np.testing.assert_array_equal(glass.grf.corr(t1, t2, x), y)
    np.testing.assert_array_almost_equal_nulp(glass.grf.icorr(t1, t2, y), x, nulp=5)
    np.testing.assert_array_equal(glass.grf.dcorr(t1, t2, x), dy)
