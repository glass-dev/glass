import numpy as np


def test_gen_ellip_intnorm():

    from glass.galaxies import gen_ellip_intnorm

    gen = gen_ellip_intnorm(0.256)
    gen.send(None)

    ngal = 1_000_000

    eps = gen.send(ngal)

    assert eps.size == ngal

    assert np.all(np.abs(eps) < 1)

    assert np.isclose(np.std(eps.real), 0.256, atol=1e-3, rtol=0)
    assert np.isclose(np.std(eps.imag), 0.256, atol=1e-3, rtol=0)

    gen.close()
