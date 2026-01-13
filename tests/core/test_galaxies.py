from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

import glass

if TYPE_CHECKING:
    from types import ModuleType

    from pytest_mock import MockerFixture

    from glass._types import FloatArray, UnifiedGenerator
    from tests.fixtures.helper_classes import Compare


def test_redshifts(mocker: MockerFixture, xp: ModuleType) -> None:
    if xp.__name__ == "jax.numpy":
        pytest.skip("Arrays in redshifts are not immutable, so do not support jax")
    # create a mock radial window function
    w = mocker.Mock()
    w.za = xp.linspace(0.0, 1.0, 20)
    w.wa = xp.exp(-0.5 * (w.za - 0.5) ** 2 / 0.1**2)

    # sample redshifts (scalar)
    z = glass.redshifts(13, w)
    assert z.shape == (13,)
    assert xp.min(z) >= 0.0
    assert xp.max(z) <= 1.0

    # sample redshifts (array)
    z = glass.redshifts(xp.asarray([[1, 2], [3, 4]]), w)
    assert z.shape == (10,)


def test_redshifts_from_nz(urng: UnifiedGenerator, xp: ModuleType) -> None:
    if xp.__name__ == "jax.numpy":
        pytest.skip(
            "Arrays in redshifts_from_nz are not immutable, so do not support jax",
        )

    # test sampling

    redshifts = glass.redshifts_from_nz(
        10,
        xp.asarray([0, 1, 2, 3, 4]),
        xp.asarray([1, 0, 0, 0, 0]),
        warn=False,
    )
    assert xp.all((redshifts >= 0) & (redshifts <= 1))

    redshifts = glass.redshifts_from_nz(
        10,
        xp.asarray([0, 1, 2, 3, 4]),
        xp.asarray([0, 0, 1, 0, 0]),
        warn=False,
    )
    assert xp.all((redshifts >= 1) & (redshifts <= 3))

    redshifts = glass.redshifts_from_nz(
        10,
        xp.asarray([0, 1, 2, 3, 4]),
        xp.asarray([0, 0, 0, 0, 1]),
        warn=False,
    )
    assert xp.all((redshifts >= 3) & (redshifts <= 4))

    redshifts = glass.redshifts_from_nz(
        10,
        xp.asarray([0, 1, 2, 3, 4]),
        xp.asarray([0, 0, 1, 1, 1]),
        warn=False,
    )
    assert not xp.any(redshifts <= 1)

    # test with rng

    redshifts = glass.redshifts_from_nz(
        10,
        xp.asarray([0, 1, 2, 3, 4]),
        xp.asarray([0, 0, 1, 1, 1]),
        warn=False,
        rng=urng,
    )
    assert not xp.any(redshifts <= 1)

    # test interface

    # case: no extra dimensions

    count = 10
    z = xp.linspace(0, 1, 100)
    nz = z * (1 - z)

    redshifts = glass.redshifts_from_nz(count, z, nz, warn=False)

    assert redshifts.shape == (count,)
    assert xp.all((redshifts >= 0) & (redshifts <= 1))

    # case: extra dimensions from count

    count = xp.asarray([10, 20, 30])
    z = xp.linspace(0, 1, 100)
    nz = z * (1 - z)

    redshifts = glass.redshifts_from_nz(count, z, nz, warn=False)

    assert redshifts.shape == (60,)

    # case: extra dimensions from nz

    count = 10
    z = xp.linspace(0, 1, 100)
    nz = xp.stack([z * (1 - z), (z - 0.5) ** 2])

    redshifts = glass.redshifts_from_nz(count, z, nz, warn=False)

    assert redshifts.shape == (20,)

    # case: extra dimensions from count and nz

    count = xp.asarray([[10], [20], [30]])
    z = xp.linspace(0, 1, 100)
    nz = xp.stack([z * (1 - z), (z - 0.5) ** 2])

    redshifts = glass.redshifts_from_nz(count, z, nz, warn=False)

    assert redshifts.shape == (120,)

    # case: incompatible input shapes

    count = xp.asarray([10, 20, 30])
    z = xp.linspace(0, 1, 100)
    nz = xp.stack([z * (1 - z), (z - 0.5) ** 2])

    with pytest.raises(ValueError, match="shape mismatch"):
        glass.redshifts_from_nz(count, z, nz, warn=False)

    with pytest.warns(UserWarning, match="when sampling galaxies"):
        redshifts = glass.redshifts_from_nz(
            10,
            xp.asarray([0, 1, 2, 3, 4]),
            xp.asarray([1, 0, 0, 0, 0]),
        )


def test_galaxy_shear(compare: type[Compare], rng: np.random.Generator) -> None:
    # check shape of the output

    kappa, gamma1, gamma2 = (
        rng.normal(size=(12,)),
        rng.normal(size=(12,)),
        rng.normal(size=(12,)),
    )

    shear = glass.galaxy_shear(
        np.asarray([]),
        np.asarray([]),
        np.asarray([]),
        kappa,
        gamma1,
        gamma2,
    )
    compare.assert_equal(shear, [])

    gal_lon, gal_lat, gal_eps = (
        rng.normal(size=(512,)),
        rng.normal(size=(512,)),
        rng.normal(size=(512,)),
    )
    shear = glass.galaxy_shear(gal_lon, gal_lat, gal_eps, kappa, gamma1, gamma2)
    assert shear.shape == (512,)

    # shape with no reduced shear

    shear = glass.galaxy_shear(
        np.asarray([]),
        np.asarray([]),
        np.asarray([]),
        kappa,
        gamma1,
        gamma2,
        reduced_shear=False,
    )
    compare.assert_equal(shear, [])

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
    assert shear.shape == (512,)


def test_gaussian_phz(
    compare: type[Compare],
    urng: UnifiedGenerator,
    xp: ModuleType,
) -> None:
    # test sampling

    # case: zero variance

    z: float | FloatArray = xp.linspace(0, 1, 100)
    sigma_0: float | FloatArray = 0.0

    phz = glass.gaussian_phz(z, sigma_0)

    compare.assert_array_equal(z, phz)

    # test with rng

    phz = glass.gaussian_phz(z, sigma_0, rng=urng)

    compare.assert_array_equal(z, phz)

    # case: truncated normal

    phz = glass.gaussian_phz(0.0, xp.ones(100))

    assert phz.__array_namespace__() == xp
    assert phz.shape == (100,)
    assert xp.all(phz >= 0)

    # case: upper and lower bound

    phz = glass.gaussian_phz(1.0, xp.ones(100), lower=0.5, upper=1.5)

    assert phz.__array_namespace__() == xp
    assert phz.shape == (100,)
    assert xp.all(phz >= 0.5)
    assert xp.all(phz <= 1.5)

    # test interface

    # case: scalar redshift, scalar sigma_0

    z = 1.0

    phz = glass.gaussian_phz(z, 0.0, xp=xp)

    assert phz.ndim == 0
    assert phz == xp.asarray(z)

    # Pass floats without xp

    with pytest.raises(
        TypeError,
        match="array_namespace requires at least one non-scalar array input",
    ):
        glass.gaussian_phz(1.0, 0.0)

    # case: array redshift, scalar sigma_0

    z = xp.linspace(0, 1, 10)

    phz = glass.gaussian_phz(z, 0.0)

    assert phz.__array_namespace__() == xp
    assert phz.shape == (10,)
    compare.assert_array_equal(z, phz)

    # case: scalar redshift, array sigma_0

    z = 1.0
    sigma_0 = xp.zeros(10)

    phz = glass.gaussian_phz(z, sigma_0)

    assert phz.__array_namespace__() == xp
    assert phz.shape == (10,)
    compare.assert_array_equal(z, phz)

    # case: array redshift, array sigma_0

    z = xp.linspace(0, 1, 10)
    sigma_0 = xp.zeros((11, 1))

    phz = glass.gaussian_phz(z, sigma_0)

    assert phz.__array_namespace__() == xp
    assert phz.shape == (11, 10)
    compare.assert_array_equal(xp.broadcast_to(z, (11, 10)), phz)

    # shape mismatch

    with pytest.raises(
        ValueError,
        match="lower and upper must best scalars or have the same shape as z",
    ):
        glass.gaussian_phz(xp.asarray(0.0), xp.asarray(1.0), lower=xp.asarray([0]))

    with pytest.raises(
        ValueError,
        match="lower and upper must best scalars or have the same shape as z",
    ):
        glass.gaussian_phz(xp.asarray(0.0), xp.asarray(1.0), upper=xp.asarray([1]))

    # test resampling

    phz = glass.gaussian_phz(xp.asarray(0.0), xp.asarray(1.0), lower=0)
    assert phz.ndim == 0

    phz = glass.gaussian_phz(xp.asarray(0.0), xp.asarray(1.0), upper=1)
    assert phz.ndim == 0

    # test error

    with pytest.raises(ValueError, match="requires lower < upper"):
        phz = glass.gaussian_phz(z, sigma_0, lower=xp.asarray(1), upper=xp.asarray(0))
