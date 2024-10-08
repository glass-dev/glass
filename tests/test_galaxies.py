import numpy as np
import pytest  # type: ignore[import-not-found]

from glass.galaxies import gaussian_phz, redshifts, redshifts_from_nz


def test_redshifts(mocker) -> None:  # type: ignore[no-untyped-def]
    # create a mock radial window function
    w = mocker.Mock()
    w.za = np.linspace(0.0, 1.0, 20)
    w.wa = np.exp(-0.5 * (w.za - 0.5) ** 2 / 0.1**2)

    # sample redshifts (scalar)
    z = redshifts(13, w)
    assert z.shape == (13,)  # type: ignore[union-attr]
    assert z.min() >= 0.0  # type: ignore[union-attr]
    assert z.max() <= 1.0  # type: ignore[union-attr]

    # sample redshifts (array)
    z = redshifts([[1, 2], [3, 4]], w)
    assert z.shape == (10,)  # type: ignore[union-attr]


def test_redshifts_from_nz() -> None:
    # test sampling

    redshifts = redshifts_from_nz(10, [0, 1, 2, 3, 4], [1, 0, 0, 0, 0])
    assert np.all((0 <= redshifts) & (redshifts <= 1))  # type: ignore[operator] # noqa: SIM300

    redshifts = redshifts_from_nz(10, [0, 1, 2, 3, 4], [0, 0, 1, 0, 0])
    assert np.all((1 <= redshifts) & (redshifts <= 3))  # type: ignore[operator] # noqa: SIM300

    redshifts = redshifts_from_nz(10, [0, 1, 2, 3, 4], [0, 0, 0, 0, 1])
    assert np.all((3 <= redshifts) & (redshifts <= 4))  # type: ignore[operator] # noqa: SIM300

    redshifts = redshifts_from_nz(10, [0, 1, 2, 3, 4], [0, 0, 1, 1, 1])
    assert not np.any(redshifts <= 1)  # type: ignore[operator]

    # test interface

    # case: no extra dimensions

    count = 10
    z = np.linspace(0, 1, 100)
    nz = z * (1 - z)

    redshifts = redshifts_from_nz(count, z, nz)

    assert redshifts.shape == (count,)  # type: ignore[union-attr]
    assert np.all((0 <= redshifts) & (redshifts <= 1))  # type: ignore[operator] # noqa: SIM300

    # case: extra dimensions from count

    count = [10, 20, 30]  # type: ignore[assignment]
    z = np.linspace(0, 1, 100)
    nz = z * (1 - z)

    redshifts = redshifts_from_nz(count, z, nz)

    assert np.shape(redshifts) == (60,)

    # case: extra dimensions from nz

    count = 10
    z = np.linspace(0, 1, 100)
    nz = [z * (1 - z), (z - 0.5) ** 2]  # type: ignore[assignment]

    redshifts = redshifts_from_nz(count, z, nz)

    assert redshifts.shape == (20,)  # type: ignore[union-attr]

    # case: extra dimensions from count and nz

    count = [[10], [20], [30]]  # type: ignore[assignment]
    z = np.linspace(0, 1, 100)
    nz = [z * (1 - z), (z - 0.5) ** 2]  # type: ignore[assignment]

    redshifts = redshifts_from_nz(count, z, nz)

    assert redshifts.shape == (120,)  # type: ignore[union-attr]

    # case: incompatible input shapes

    count = [10, 20, 30]  # type: ignore[assignment]
    z = np.linspace(0, 1, 100)
    nz = [z * (1 - z), (z - 0.5) ** 2]  # type: ignore[assignment]

    with pytest.raises(ValueError):
        redshifts_from_nz(count, z, nz)


def test_gaussian_phz() -> None:
    # test sampling

    # case: zero variance

    z = np.linspace(0, 1, 100)
    sigma_0 = 0.0

    phz = gaussian_phz(z, sigma_0)

    np.testing.assert_array_equal(z, phz)

    # case: truncated normal

    z = 0.0  # type: ignore[assignment]
    sigma_0 = np.ones(100)  # type: ignore[assignment]

    phz = gaussian_phz(z, sigma_0)

    assert phz.shape == (100,)  # type: ignore[union-attr]
    assert np.all(phz >= 0)  # type: ignore[operator]

    # case: upper and lower bound

    z = 1.0  # type: ignore[assignment]
    sigma_0 = np.ones(100)  # type: ignore[assignment]

    phz = gaussian_phz(z, sigma_0, lower=0.5, upper=1.5)

    assert phz.shape == (100,)  # type: ignore[union-attr]
    assert np.all(phz >= 0.5)  # type: ignore[operator]
    assert np.all(phz <= 1.5)  # type: ignore[operator]

    # test interface

    # case: scalar redshift, scalar sigma_0

    z = 1.0  # type: ignore[assignment]
    sigma_0 = 0.0

    phz = gaussian_phz(z, sigma_0)

    assert np.ndim(phz) == 0
    assert phz == z

    # case: array redshift, scalar sigma_0

    z = np.linspace(0, 1, 10)
    sigma_0 = 0.0

    phz = gaussian_phz(z, sigma_0)

    assert phz.shape == (10,)  # type: ignore[union-attr]
    np.testing.assert_array_equal(z, phz)

    # case: scalar redshift, array sigma_0

    z = 1.0  # type: ignore[assignment]
    sigma_0 = np.zeros(10)  # type: ignore[assignment]

    phz = gaussian_phz(z, sigma_0)

    assert phz.shape == (10,)  # type: ignore[union-attr]
    np.testing.assert_array_equal(z, phz)

    # case: array redshift, array sigma_0

    z = np.linspace(0, 1, 10)
    sigma_0 = np.zeros((11, 1))  # type: ignore[assignment]

    phz = gaussian_phz(z, sigma_0)

    assert phz.shape == (11, 10)  # type: ignore[union-attr]
    np.testing.assert_array_equal(np.broadcast_to(z, (11, 10)), phz)
