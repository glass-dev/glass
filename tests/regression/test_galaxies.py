from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import glass

if TYPE_CHECKING:
    from types import ModuleType

    from pytest_benchmark.fixture import BenchmarkFixture

    from glass._types import UnifiedGenerator


@pytest.mark.stable
def test_redshifts_demo(
    benchmark: BenchmarkFixture,
    urng: UnifiedGenerator,
    xp: ModuleType,
) -> None:
    """Regression test for galaxies.redshifts."""
    scale_factor = 1_000
    # create a mock radial window function
    za = xp.linspace(0.0, 1.0, 20 * scale_factor)
    wa = xp.exp(-0.5 * (za - 0.5) ** 2 / 0.1**2)
    w = glass.RadialWindow(za, wa)

    # sample redshifts (scalar)
    z = benchmark(glass.redshifts, 13 * scale_factor, w, rng=urng)
    assert z.shape == (13 * scale_factor,)
    assert xp.min(z) >= 0.0
    assert xp.max(z) <= 1.0


@pytest.mark.stable
def test_redshifts_from_bins(
    benchmark: BenchmarkFixture,
    urng: UnifiedGenerator,
    xp: ModuleType,
) -> None:
    """Regression test for galaxies.redshifts_from_bins."""
    # sample N of redshifts from K bins
    # good values are millions of redshifts, O(10) bins

    # random bins
    bins = xp.astype(urng.uniform(0.0, 10.0, size=1_000_000), int)

    # generate a flat redshift distribution
    # shape does not matter for performance
    z = xp.linspace(0.0, 3.0, 301)
    nz_dict = {k: xp.ones(z.shape) for k in range(10)}

    # sample redshifts (scalar)
    redshifts = benchmark(
        glass.redshifts_from_bins,
        bins,
        z,
        nz_dict,
        rng=urng,
    )

    # ensure that function has run on input size
    assert redshifts.shape == bins.shape


@pytest.mark.stable
def test_redshifts_from_nz(
    benchmark: BenchmarkFixture,
    urng: UnifiedGenerator,
    xp: ModuleType,
) -> None:
    """Regression test for galaxies.redshifts_from_nz."""
    scale_factor = 1_000
    # create a mock radial window function
    za = xp.linspace(0.0, 1.0, 20 * scale_factor)
    wa = xp.exp(-0.5 * (za - 0.5) ** 2 / 0.1**2)

    # sample redshifts (scalar)
    redshifts = benchmark(
        glass.redshifts_from_nz,
        13 * scale_factor,
        za,
        wa,
        rng=urng,
        warn=False,
    )
    assert redshifts.shape == (13 * scale_factor,)
    assert xp.min(redshifts) >= 0.0
    assert xp.max(redshifts) <= 1.0
    assert xp.all((redshifts >= 0) & (redshifts <= 1))


@pytest.mark.unstable
@pytest.mark.parametrize("reduced_shear", [True, False])
def test_galaxy_shear(
    benchmark: BenchmarkFixture,
    urng: UnifiedGenerator,
    reduced_shear: bool,  # noqa: FBT001
) -> None:
    """Regression test for galaxies.galaxy_shear."""
    scale_factor = 100
    size = (12 * scale_factor,)
    kappa = urng.normal(size=size)
    gamma1 = urng.normal(size=size)
    gamma2 = urng.normal(size=size)

    gal_size = (512 * scale_factor,)
    gal_lon = urng.normal(size=gal_size)
    gal_lat = urng.normal(size=gal_size)
    gal_eps = urng.normal(size=gal_size)

    shear = benchmark(
        glass.galaxy_shear,
        gal_lon,
        gal_lat,
        gal_eps,
        kappa,
        gamma1,
        gamma2,
        reduced_shear=reduced_shear,
    )
    assert shear.shape == gal_size


@pytest.mark.stable
def test_gaussian_phz(
    benchmark: BenchmarkFixture,
    urng: UnifiedGenerator,
    xp: ModuleType,
) -> None:
    """Regression tests for galaxies.gaussian_phz."""
    scaled_length = 10_000

    z = xp.linspace(0, 1, scaled_length)
    sigma_0 = xp.ones(scaled_length)

    phz = benchmark(
        glass.gaussian_phz,
        z,
        sigma_0,
        lower=0.5,
        upper=1.5,
        rng=urng,
    )

    assert phz.shape == (scaled_length,)
    assert xp.all(phz >= 0.5)
    assert xp.all(phz <= 1.5)
