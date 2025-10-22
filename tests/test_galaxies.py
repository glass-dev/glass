from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

import glass

if TYPE_CHECKING:
    import pytest_mock
    from numpy.typing import NDArray


def test_redshifts(mocker: pytest_mock.MockerFixture) -> None:
    # create a mock radial window function
    w = mocker.Mock()
    w.za = np.linspace(0.0, 1.0, 20)
    w.wa = np.exp(-0.5 * (w.za - 0.5) ** 2 / 0.1**2)

    # sample redshifts (scalar)
    z = glass.redshifts(13, w)
    assert z.shape == (13,)
    assert z.min() >= 0.0
    assert z.max() <= 1.0

    # sample redshifts (array)
    z = glass.redshifts(np.array([[1, 2], [3, 4]]), w)
    assert z.shape == (10,)


def test_redshifts_from_nz(rng: np.random.Generator) -> None:
    # test sampling

    redshifts = glass.redshifts_from_nz(
        10,
        np.array([0, 1, 2, 3, 4]),
        np.array([1, 0, 0, 0, 0]),
        warn=False,
    )
    assert np.all((0 <= redshifts) & (redshifts <= 1))  # noqa: SIM300

    redshifts = glass.redshifts_from_nz(
        10,
        np.array([0, 1, 2, 3, 4]),
        np.array([0, 0, 1, 0, 0]),
        warn=False,
    )
    assert np.all((1 <= redshifts) & (redshifts <= 3))  # noqa: SIM300

    redshifts = glass.redshifts_from_nz(
        10,
        np.array([0, 1, 2, 3, 4]),
        np.array([0, 0, 0, 0, 1]),
        warn=False,
    )
    assert np.all((3 <= redshifts) & (redshifts <= 4))  # noqa: SIM300

    redshifts = glass.redshifts_from_nz(
        10,
        np.array([0, 1, 2, 3, 4]),
        np.array([0, 0, 1, 1, 1]),
        warn=False,
    )
    assert not np.any(redshifts <= 1)

    # test with rng

    redshifts = glass.redshifts_from_nz(
        10,
        np.array([0, 1, 2, 3, 4]),
        np.array([0, 0, 1, 1, 1]),
        warn=False,
        rng=rng,
    )
    assert not np.any(redshifts <= 1)

    # test interface

    # case: no extra dimensions

    count: int | NDArray[np.float64] = 10
    z = np.linspace(0, 1, 100)
    nz = z * (1 - z)

    redshifts = glass.redshifts_from_nz(count, z, nz, warn=False)

    assert redshifts.shape == (count,)
    assert np.all((0 <= redshifts) & (redshifts <= 1))  # noqa: SIM300

    # case: extra dimensions from count

    count = np.array([10, 20, 30])
    z = np.linspace(0, 1, 100)
    nz = z * (1 - z)

    redshifts = glass.redshifts_from_nz(count, z, nz, warn=False)

    assert np.shape(redshifts) == (60,)

    # case: extra dimensions from nz

    count = 10
    z = np.linspace(0, 1, 100)
    nz = np.array([z * (1 - z), (z - 0.5) ** 2])

    redshifts = glass.redshifts_from_nz(count, z, nz, warn=False)

    assert redshifts.shape == (20,)

    # case: extra dimensions from count and nz

    count = np.array([[10], [20], [30]])
    z = np.linspace(0, 1, 100)
    nz = np.array([z * (1 - z), (z - 0.5) ** 2])

    redshifts = glass.redshifts_from_nz(count, z, nz, warn=False)

    assert redshifts.shape == (120,)

    # case: incompatible input shapes

    count = np.array([10, 20, 30])
    z = np.linspace(0, 1, 100)
    nz = np.array([z * (1 - z), (z - 0.5) ** 2])

    with pytest.raises(ValueError, match="shape mismatch"):
        glass.redshifts_from_nz(count, z, nz, warn=False)

    with pytest.warns(UserWarning, match="when sampling galaxies"):
        redshifts = glass.redshifts_from_nz(
            10,
            np.array([0, 1, 2, 3, 4]),
            np.array([1, 0, 0, 0, 0]),
        )


def test_galaxy_shear(rng: np.random.Generator) -> None:
    # check shape of the output

    kappa, gamma1, gamma2 = (
        rng.normal(size=(12,)),
        rng.normal(size=(12,)),
        rng.normal(size=(12,)),
    )

    shear = glass.galaxy_shear(
        np.array([]),
        np.array([]),
        np.array([]),
        kappa,
        gamma1,
        gamma2,
    )
    np.testing.assert_equal(shear, [])

    gal_lon, gal_lat, gal_eps = (
        rng.normal(size=(512,)),
        rng.normal(size=(512,)),
        rng.normal(size=(512,)),
    )
    shear = glass.galaxy_shear(gal_lon, gal_lat, gal_eps, kappa, gamma1, gamma2)
    assert np.shape(shear) == (512,)

    # shape with no reduced shear

    shear = glass.galaxy_shear(
        np.array([]),
        np.array([]),
        np.array([]),
        kappa,
        gamma1,
        gamma2,
        reduced_shear=False,
    )
    np.testing.assert_equal(shear, [])

    gal_lon, gal_lat, gal_eps = (
        rng.normal(size=(512,)),
        rng.normal(size=(512,)),
        rng.normal(size=(512,)),
    )
    shear = glass.galaxy_shear(
        gal_lon,
        gal_lat,
        gal_eps,
        kappa,
        gamma1,
        gamma2,
        reduced_shear=False,
    )
    assert np.shape(shear) == (512,)


def test_gaussian_phz(rng: np.random.Generator) -> None:
    # test sampling

    # case: zero variance

    z: float | NDArray[np.float64] = np.linspace(0, 1, 100)
    sigma_0: float | NDArray[np.float64] = 0.0

    phz = glass.gaussian_phz(z, sigma_0)

    np.testing.assert_array_equal(z, phz)

    # test with rng

    phz = glass.gaussian_phz(z, sigma_0, rng=rng)

    np.testing.assert_array_equal(z, phz)

    # case: truncated normal

    z = 0.0
    sigma_0 = np.ones(100)

    phz = glass.gaussian_phz(z, sigma_0)

    assert isinstance(phz, np.ndarray)
    assert phz.shape == (100,)
    assert np.all(phz >= 0)

    # case: upper and lower bound

    z = 1.0
    sigma_0 = np.ones(100)

    phz = glass.gaussian_phz(z, sigma_0, lower=0.5, upper=1.5)

    assert isinstance(phz, np.ndarray)
    assert phz.shape == (100,)
    assert np.all(phz >= 0.5)
    assert np.all(phz <= 1.5)

    # test interface

    # case: scalar redshift, scalar sigma_0

    z = 1.0
    sigma_0 = 0.0

    phz = glass.gaussian_phz(z, sigma_0)

    assert np.ndim(phz) == 0
    assert phz == z

    # case: array redshift, scalar sigma_0

    z = np.linspace(0, 1, 10)
    sigma_0 = 0.0

    phz = glass.gaussian_phz(z, sigma_0)

    assert isinstance(phz, np.ndarray)
    assert phz.shape == (10,)
    np.testing.assert_array_equal(z, phz)

    # case: scalar redshift, array sigma_0

    z = 1.0
    sigma_0 = np.zeros(10)

    phz = glass.gaussian_phz(z, sigma_0)

    assert isinstance(phz, np.ndarray)
    assert phz.shape == (10,)
    np.testing.assert_array_equal(z, phz)

    # case: array redshift, array sigma_0

    z = np.linspace(0, 1, 10)
    sigma_0 = np.zeros((11, 1))

    phz = glass.gaussian_phz(z, sigma_0)

    assert isinstance(phz, np.ndarray)
    assert phz.shape == (11, 10)
    np.testing.assert_array_equal(np.broadcast_to(z, (11, 10)), phz)

    # test resampling

    phz = glass.gaussian_phz(np.array(0.0), np.array(1.0), lower=np.array([0]))
    assert isinstance(phz, float)

    phz = glass.gaussian_phz(np.array(0.0), np.array(1.0), upper=np.array([1]))
    assert isinstance(phz, float)

    # test error

    with pytest.raises(ValueError, match="requires lower < upper"):
        phz = glass.gaussian_phz(z, sigma_0, lower=np.array([1]), upper=np.array([0]))
