import numpy as np


def test_ellipticity_intnorm():

    from glass.shapes import ellipticity_intnorm

    n = 1_000_000

    eps = ellipticity_intnorm(n, 0.256)

    assert eps.size == n

    assert np.all(np.abs(eps) < 1)

    assert np.isclose(np.std(eps.real), 0.256, atol=1e-3, rtol=0)
    assert np.isclose(np.std(eps.imag), 0.256, atol=1e-3, rtol=0)
