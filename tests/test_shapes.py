import numpy as np
import pytest


def test_triaxial_axis_ratio():
    from glass.shapes import triaxial_axis_ratio

    q = triaxial_axis_ratio(0.8, 0.4)
    assert np.isscalar(q)

    q = triaxial_axis_ratio(0.8, 0.4, size=1000)
    assert np.shape(q) == (1000,)

    q = triaxial_axis_ratio(0.8, 0.4, size=(10, 10))
    assert np.shape(q) == (10, 10)

    q1 = triaxial_axis_ratio([0.8, 0.9], 0.4)
    q2 = triaxial_axis_ratio(0.8, [0.4, 0.5])
    assert np.shape(q1) == np.shape(q2) == (2,)

    q = triaxial_axis_ratio([[0.6, 0.7], [0.8, 0.9]], [0.4, 0.5])
    assert np.shape(q) == (2, 2)

    zeta, xi = np.sort(np.random.uniform(0, 1, size=(2, 1000)), axis=0)
    qmin = np.min([zeta, xi, xi / zeta], axis=0)
    qmax = np.max([zeta, xi, xi / zeta], axis=0)
    q = triaxial_axis_ratio(zeta, xi)
    assert np.all((qmax >= q) & (q >= qmin))


def test_ellipticity_ryden04():
    from glass.shapes import ellipticity_ryden04

    e = ellipticity_ryden04(-1.85, 0.89, 0.222, 0.056)
    assert np.isscalar(e)

    e = ellipticity_ryden04(-1.85, 0.89, 0.222, 0.056, size=1000)
    assert np.shape(e) == (1000,)

    e = ellipticity_ryden04(-1.85, 0.89, 0.222, 0.056, size=(10, 10))
    assert np.shape(e) == (10, 10)

    e1 = ellipticity_ryden04(-1.85, 0.89, [0.222, 0.333], 0.056)
    e2 = ellipticity_ryden04(-1.85, 0.89, 0.222, [0.056, 0.067])
    e3 = ellipticity_ryden04([-1.85, -2.85], 0.89, 0.222, 0.056)
    assert np.shape(e1) == np.shape(e2) == np.shape(e3) == (2,)

    e = ellipticity_ryden04(0.0, 1.0, 0.222, 0.056, size=10)
    assert np.all((e.real >= -1.0) & (e.real <= 1.0))

    e = ellipticity_ryden04(0.0, 1.0, 0.0, 1.0, size=10)
    assert np.all((e.real >= -1.0) & (e.real <= 1.0))


def test_ellipticity_gaussian():
    from glass.shapes import ellipticity_gaussian

    n = 1_000_000

    eps = ellipticity_gaussian(n, 0.256)

    assert eps.shape == (n,)

    assert np.all(np.abs(eps) < 1)

    assert np.isclose(np.std(eps.real), 0.256, atol=1e-3, rtol=0)
    assert np.isclose(np.std(eps.imag), 0.256, atol=1e-3, rtol=0)

    eps = ellipticity_gaussian([n, n], [0.128, 0.256])

    assert eps.shape == (2 * n,)

    assert np.all(np.abs(eps) < 1)

    assert np.isclose(np.std(eps.real[:n]), 0.128, atol=1e-3, rtol=0)
    assert np.isclose(np.std(eps.imag[:n]), 0.128, atol=1e-3, rtol=0)
    assert np.isclose(np.std(eps.real[n:]), 0.256, atol=1e-3, rtol=0)
    assert np.isclose(np.std(eps.imag[n:]), 0.256, atol=1e-3, rtol=0)


def test_ellipticity_intnorm():
    from glass.shapes import ellipticity_intnorm

    n = 1_000_000

    eps = ellipticity_intnorm(n, 0.256)

    assert eps.shape == (n,)

    assert np.all(np.abs(eps) < 1)

    assert np.isclose(np.std(eps.real), 0.256, atol=1e-3, rtol=0)
    assert np.isclose(np.std(eps.imag), 0.256, atol=1e-3, rtol=0)

    eps = ellipticity_intnorm([n, n], [0.128, 0.256])

    assert eps.shape == (2 * n,)

    assert np.all(np.abs(eps) < 1)

    assert np.isclose(np.std(eps.real[:n]), 0.128, atol=1e-3, rtol=0)
    assert np.isclose(np.std(eps.imag[:n]), 0.128, atol=1e-3, rtol=0)
    assert np.isclose(np.std(eps.real[n:]), 0.256, atol=1e-3, rtol=0)
    assert np.isclose(np.std(eps.imag[n:]), 0.256, atol=1e-3, rtol=0)

    with pytest.raises(ValueError):
        ellipticity_intnorm(1, 0.71)
