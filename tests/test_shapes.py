import numpy as np
import pytest


def test_ellipticity_gaussian():

    from glass.shapes import ellipticity_gaussian

    n = 1_000_000

    eps = ellipticity_gaussian(n, 0.256)

    assert eps.shape == (n,)

    assert np.all(np.abs(eps) < 1)

    assert np.isclose(np.std(eps.real), 0.256, atol=1e-3, rtol=0)
    assert np.isclose(np.std(eps.imag), 0.256, atol=1e-3, rtol=0)

    eps = ellipticity_gaussian([n, n], [0.128, 0.256])

    assert eps.shape == (2*n,)

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

    assert eps.shape == (2*n,)

    assert np.all(np.abs(eps) < 1)

    assert np.isclose(np.std(eps.real[:n]), 0.128, atol=1e-3, rtol=0)
    assert np.isclose(np.std(eps.imag[:n]), 0.128, atol=1e-3, rtol=0)
    assert np.isclose(np.std(eps.real[n:]), 0.256, atol=1e-3, rtol=0)
    assert np.isclose(np.std(eps.imag[n:]), 0.256, atol=1e-3, rtol=0)

    with pytest.raises(ValueError):
        ellipticity_intnorm(1, 0.71)
