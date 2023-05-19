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
