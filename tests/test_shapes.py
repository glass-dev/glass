import numpy as np
import pytest

from glass.shapes import (
    ellipticity_gaussian,
    ellipticity_intnorm,
    ellipticity_ryden04,
    triaxial_axis_ratio,
)


def test_triaxial_axis_ratio(rng):  # type: ignore[no-untyped-def]
    # single axis ratio

    q = triaxial_axis_ratio(0.8, 0.4)  # type: ignore[no-untyped-call]
    assert np.isscalar(q)

    # many axis ratios

    q = triaxial_axis_ratio(0.8, 0.4, size=1000)  # type: ignore[no-untyped-call]
    assert np.shape(q) == (1000,)

    # explicit shape

    q = triaxial_axis_ratio(0.8, 0.4, size=(10, 10))  # type: ignore[no-untyped-call]
    assert np.shape(q) == (10, 10)

    # implicit size

    q1 = triaxial_axis_ratio([0.8, 0.9], 0.4)  # type: ignore[no-untyped-call]
    q2 = triaxial_axis_ratio(0.8, [0.4, 0.5])  # type: ignore[no-untyped-call]
    assert np.shape(q1) == np.shape(q2) == (2,)

    # broadcasting rule

    q = triaxial_axis_ratio([[0.6, 0.7], [0.8, 0.9]], [0.4, 0.5])  # type: ignore[no-untyped-call]
    assert np.shape(q) == (2, 2)

    # random parameters and check that projection is
    # between largest and smallest possible value

    zeta, xi = np.sort(rng.uniform(0, 1, size=(2, 1000)), axis=0)
    qmin = np.min([zeta, xi, xi / zeta], axis=0)
    qmax = np.max([zeta, xi, xi / zeta], axis=0)
    q = triaxial_axis_ratio(zeta, xi)  # type: ignore[no-untyped-call]
    assert np.all((qmax >= q) & (q >= qmin))


def test_ellipticity_ryden04(rng):  # type: ignore[no-untyped-def]
    # single ellipticity

    e = ellipticity_ryden04(-1.85, 0.89, 0.222, 0.056)  # type: ignore[no-untyped-call]
    assert np.isscalar(e)

    # test with rng

    e = ellipticity_ryden04(-1.85, 0.89, 0.222, 0.056, rng=rng)  # type: ignore[no-untyped-call]
    assert np.isscalar(e)

    # many ellipticities

    e = ellipticity_ryden04(-1.85, 0.89, 0.222, 0.056, size=1000)  # type: ignore[no-untyped-call]
    assert np.shape(e) == (1000,)

    # explicit shape

    e = ellipticity_ryden04(-1.85, 0.89, 0.222, 0.056, size=(10, 10))  # type: ignore[no-untyped-call]
    assert np.shape(e) == (10, 10)

    # implicit size

    e1 = ellipticity_ryden04(-1.85, 0.89, [0.222, 0.333], 0.056)  # type: ignore[no-untyped-call]
    e2 = ellipticity_ryden04(-1.85, 0.89, 0.222, [0.056, 0.067])  # type: ignore[no-untyped-call]
    e3 = ellipticity_ryden04([-1.85, -2.85], 0.89, 0.222, 0.056)  # type: ignore[no-untyped-call]
    e4 = ellipticity_ryden04(-1.85, [0.89, 1.001], 0.222, 0.056)  # type: ignore[no-untyped-call]
    assert np.shape(e1) == np.shape(e2) == np.shape(e3) == np.shape(e4) == (2,)

    # broadcasting rule

    e = ellipticity_ryden04([-1.9, -2.9], 0.9, [[0.2, 0.3], [0.4, 0.5]], 0.1)  # type: ignore[no-untyped-call]
    assert np.shape(e) == (2, 2)

    # check that result is in the specified range

    e = ellipticity_ryden04(0.0, 1.0, 0.222, 0.056, size=10)  # type: ignore[no-untyped-call]
    assert np.all((e.real >= -1.0) & (e.real <= 1.0))

    e = ellipticity_ryden04(0.0, 1.0, 0.0, 1.0, size=10)  # type: ignore[no-untyped-call]
    assert np.all((e.real >= -1.0) & (e.real <= 1.0))


def test_ellipticity_gaussian(rng):  # type: ignore[no-untyped-def]
    n = 1_000_000

    eps = ellipticity_gaussian(n, 0.256)

    assert eps.shape == (n,)

    # test with rng

    eps = ellipticity_gaussian(n, 0.256, rng=rng)

    assert eps.shape == (n,)

    np.testing.assert_array_less(np.abs(eps), 1)

    np.testing.assert_allclose(np.std(eps.real), 0.256, atol=1e-3, rtol=0)
    np.testing.assert_allclose(np.std(eps.imag), 0.256, atol=1e-2, rtol=0)

    eps = ellipticity_gaussian([n, n], [0.128, 0.256])

    assert eps.shape == (2 * n,)

    np.testing.assert_array_less(np.abs(eps), 1)

    np.testing.assert_allclose(np.std(eps.real[:n]), 0.128, atol=1e-3, rtol=0)
    np.testing.assert_allclose(np.std(eps.imag[:n]), 0.128, atol=1e-3, rtol=0)
    np.testing.assert_allclose(np.std(eps.real[n:]), 0.256, atol=1e-3, rtol=0)
    np.testing.assert_allclose(np.std(eps.imag[n:]), 0.256, atol=1e-3, rtol=0)


def test_ellipticity_intnorm(rng):  # type: ignore[no-untyped-def]
    n = 1_000_000

    eps = ellipticity_intnorm(n, 0.256)

    assert eps.shape == (n,)

    # test with rng

    eps = ellipticity_intnorm(n, 0.256, rng=rng)

    assert eps.shape == (n,)

    np.testing.assert_array_less(np.abs(eps), 1)

    np.testing.assert_allclose(np.std(eps.real), 0.256, atol=1e-3, rtol=0)
    np.testing.assert_allclose(np.std(eps.imag), 0.256, atol=1e-3, rtol=0)

    eps = ellipticity_intnorm([n, n], [0.128, 0.256])

    assert eps.shape == (2 * n,)

    np.testing.assert_array_less(np.abs(eps), 1)

    np.testing.assert_allclose(np.std(eps.real[:n]), 0.128, atol=1e-3, rtol=0)
    np.testing.assert_allclose(np.std(eps.imag[:n]), 0.128, atol=1e-3, rtol=0)
    np.testing.assert_allclose(np.std(eps.real[n:]), 0.256, atol=1e-3, rtol=0)
    np.testing.assert_allclose(np.std(eps.imag[n:]), 0.256, atol=1e-3, rtol=0)

    with pytest.raises(ValueError, match="sigma must be between"):
        ellipticity_intnorm(1, 0.71)
