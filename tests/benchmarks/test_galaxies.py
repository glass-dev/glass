from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import glass

if TYPE_CHECKING:
    from types import ModuleType

    from pytest_benchmark.fixture import BenchmarkFixture

    from glass._types import UnifiedGenerator


@pytest.mark.stable
def test_redshifts(
    benchmark: BenchmarkFixture,
    urng_benchmarks: UnifiedGenerator,
    xp_benchmarks: ModuleType,
) -> None:
    """Benchmark for galaxies.redshifts."""
    scale_factor = 1_000
    # create a mock radial window function
    za = xp_benchmarks.linspace(0.0, 1.0, 20 * scale_factor)
    wa = xp_benchmarks.exp(-0.5 * (za - 0.5) ** 2 / 0.1**2)
    w = glass.RadialWindow(za, wa)

    # sample redshifts (scalar)
    z = benchmark(glass.redshifts, 13 * scale_factor, w, rng=urng_benchmarks)
    assert z.shape == (13 * scale_factor,)
    assert xp_benchmarks.min(z) >= 0.0
    assert xp_benchmarks.max(z) <= 1.0


@pytest.mark.stable
def test_redshifts_from_nz(
    benchmark: BenchmarkFixture,
    urng_benchmarks: UnifiedGenerator,
    xp_benchmarks: ModuleType,
) -> None:
    """Benchmark for galaxies.redshifts_from_nz."""
    scale_factor = 1_000
    # create a mock radial window function
    za = xp_benchmarks.linspace(0.0, 1.0, 20 * scale_factor)
    wa = xp_benchmarks.exp(-0.5 * (za - 0.5) ** 2 / 0.1**2)

    # sample redshifts (scalar)
    redshifts = benchmark(
        glass.redshifts_from_nz,
        13 * scale_factor,
        za,
        wa,
        rng=urng_benchmarks,
    )
    assert redshifts.shape == (13 * scale_factor,)
    assert xp_benchmarks.min(redshifts) >= 0.0
    assert xp_benchmarks.max(redshifts) <= 1.0
    assert xp_benchmarks.all((0 <= redshifts) & (redshifts <= 1))  # noqa: SIM300


@pytest.mark.stable
@pytest.mark.parametrize("reduced_shear", [True, False])
def test_galaxy_shear(
    benchmark: BenchmarkFixture,
    urng_benchmarks: UnifiedGenerator,
    xp_benchmarks: ModuleType,
    reduced_shear: bool,  # noqa: FBT001
) -> None:
    """Benchmark for galaxies.galaxy_shear."""
    if xp_benchmarks.__name__ == "array_api_strict":
        pytest.skip(f"glass.galaxy_shear not yet ported for {xp_benchmarks.__name__}")

    scale_factor = 100
    size = (12 * scale_factor,)
    kappa = urng_benchmarks.normal(size=size)
    gamma1 = urng_benchmarks.normal(size=size)
    gamma2 = urng_benchmarks.normal(size=size)

    gal_size = (512 * scale_factor,)
    gal_lon = urng_benchmarks.normal(size=gal_size)
    gal_lat = urng_benchmarks.normal(size=gal_size)
    gal_eps = urng_benchmarks.normal(size=gal_size)

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
    urng_benchmarks: UnifiedGenerator,
    xp_benchmarks: ModuleType,
) -> None:
    """Benchmarks for galaxies.gaussian_phz."""
    scaled_length = 10_000

    z = xp_benchmarks.linspace(0, 1, scaled_length)
    sigma_0 = xp_benchmarks.ones(scaled_length)

    phz = benchmark(
        glass.gaussian_phz,
        z,
        sigma_0,
        lower=0.5,
        upper=1.5,
        rng=urng_benchmarks,
    )

    assert phz.shape == (scaled_length,)
    assert xp_benchmarks.all(phz >= 0.5)
    assert xp_benchmarks.all(phz <= 1.5)
