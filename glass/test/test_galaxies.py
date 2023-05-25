import pytest


def test_redshifts_from_nz():
    import numpy as np
    from glass.galaxies import redshifts_from_nz

    # test sampling

    redshifts = redshifts_from_nz(10, [0, 1, 2, 3, 4], [1, 0, 0, 0, 0])
    assert np.all((0 <= redshifts) & (redshifts <= 1))

    redshifts = redshifts_from_nz(10, [0, 1, 2, 3, 4], [0, 0, 1, 0, 0])
    assert np.all((1 <= redshifts) & (redshifts <= 3))

    redshifts = redshifts_from_nz(10, [0, 1, 2, 3, 4], [0, 0, 0, 0, 1])
    assert np.all((3 <= redshifts) & (redshifts <= 4))

    redshifts = redshifts_from_nz(10, [0, 1, 2, 3, 4], [0, 0, 1, 1, 1])
    assert not np.any(redshifts <= 1)

    # test interface

    # case: no extra dimensions

    count = 10
    z = np.linspace(0, 1, 100)
    nz = z*(1-z)

    redshifts = redshifts_from_nz(count, z, nz)

    assert redshifts.shape == (count,)
    assert np.all((0 <= redshifts) & (redshifts <= 1))

    # case: extra dimensions from count

    count = [10, 20, 30]
    z = np.linspace(0, 1, 100)
    nz = z*(1-z)

    redshifts = redshifts_from_nz(count, z, nz)

    assert np.shape(redshifts) == (60,)

    # case: extra dimensions from nz

    count = 10
    z = np.linspace(0, 1, 100)
    nz = [z*(1-z), (z-0.5)**2]

    redshifts = redshifts_from_nz(count, z, nz)

    assert redshifts.shape == (20,)

    # case: extra dimensions from count and nz

    count = [[10], [20], [30]]
    z = np.linspace(0, 1, 100)
    nz = [z*(1-z), (z-0.5)**2]

    redshifts = redshifts_from_nz(count, z, nz)

    assert redshifts.shape == (120,)

    # case: incompatible input shapes

    count = [10, 20, 30]
    z = np.linspace(0, 1, 100)
    nz = [z*(1-z), (z-0.5)**2]

    with pytest.raises(ValueError):
        redshifts_from_nz(count, z, nz)


def test_gaussian_phz():
    import numpy as np
    from glass.galaxies import gaussian_phz

    # test sampling

    # case: zero variance

    z = np.linspace(0, 1, 100)
    sigma_0 = 0.

    phz = gaussian_phz(z, sigma_0)

    np.testing.assert_array_equal(z, phz)

    # case: truncated normal

    z = 0.
    sigma_0 = np.ones(100)

    phz = gaussian_phz(z, sigma_0)

    assert phz.shape == (100,)
    assert np.all(phz >= 0)

    # test interface

    # case: scalar redshift, scalar sigma_0

    z = 1.
    sigma_0 = 0.

    phz = gaussian_phz(z, sigma_0)

    assert np.ndim(phz) == 0
    assert phz == z

    # case: array redshift, scalar sigma_0

    z = np.linspace(0, 1, 10)
    sigma_0 = 0.

    phz = gaussian_phz(z, sigma_0)

    assert phz.shape == (10,)
    np.testing.assert_array_equal(z, phz)

    # case: scalar redshift, array sigma_0

    z = 1.
    sigma_0 = np.zeros(10)

    phz = gaussian_phz(z, sigma_0)

    assert phz.shape == (10,)
    np.testing.assert_array_equal(z, phz)

    # case: array redshift, array sigma_0

    z = np.linspace(0, 1, 10)
    sigma_0 = np.zeros((11, 1))

    phz = gaussian_phz(z, sigma_0)

    assert phz.shape == (11, 10)
    np.testing.assert_array_equal(np.broadcast_to(z, (11, 10)), phz)
