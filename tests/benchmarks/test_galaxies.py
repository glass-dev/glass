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
    urngb: UnifiedGenerator,
    xpb: ModuleType,
) -> None:
    """Benchmark for galaxies.redshifts."""
    scale_factor = 1_000
    # create a mock radial window function
    za = xpb.linspace(0.0, 1.0, 20 * scale_factor)
    wa = xpb.exp(-0.5 * (za - 0.5) ** 2 / 0.1**2)
    w = glass.RadialWindow(za, wa)

    # sample redshifts (scalar)
    z = benchmark(glass.redshifts, 13 * scale_factor, w, rng=urngb)
    assert z.shape == (13 * scale_factor,)
    assert xpb.min(z) >= 0.0
    assert xpb.max(z) <= 1.0


@pytest.mark.stable
def test_redshifts_from_nz(
    benchmark: BenchmarkFixture,
    urngb: UnifiedGenerator,
    xpb: ModuleType,
) -> None:
    """Benchmark for galaxies.redshifts_from_nz."""
    scale_factor = 1_000
    # create a mock radial window function
    za = xpb.linspace(0.0, 1.0, 20 * scale_factor)
    wa = xpb.exp(-0.5 * (za - 0.5) ** 2 / 0.1**2)

    # sample redshifts (scalar)
    redshifts = benchmark(
        glass.redshifts_from_nz,
        13 * scale_factor,
        za,
        wa,
        rng=urngb,
    )
    assert redshifts.shape == (13 * scale_factor,)
    assert xpb.min(redshifts) >= 0.0
    assert xpb.max(redshifts) <= 1.0
    assert xpb.all((redshifts >= 0) & (redshifts <= 1))


@pytest.mark.stable
@pytest.mark.parametrize("reduced_shear", [True, False])
def test_galaxy_shear(
    benchmark: BenchmarkFixture,
    urngb: UnifiedGenerator,
    xpb: ModuleType,
    reduced_shear: bool,  # noqa: FBT001
) -> None:
    """Benchmark for galaxies.galaxy_shear."""
    if xpb.__name__ == "array_api_strict":
        pytest.skip(f"glass.galaxy_shear not yet ported for {xpb.__name__}")

    scale_factor = 100
    size = (12 * scale_factor,)
    kappa = urngb.normal(size=size)
    gamma1 = urngb.normal(size=size)
    gamma2 = urngb.normal(size=size)

    gal_size = (512 * scale_factor,)
    gal_lon = urngb.normal(size=gal_size)
    gal_lat = urngb.normal(size=gal_size)
    gal_eps = urngb.normal(size=gal_size)

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
    urngb: UnifiedGenerator,
    xpb: ModuleType,
) -> None:
    """Benchmarks for galaxies.gaussian_phz."""
    scaled_length = 10_000

    z = xpb.linspace(0, 1, scaled_length)
    sigma_0 = xpb.ones(scaled_length)

    phz = benchmark(
        glass.gaussian_phz,
        z,
        sigma_0,
        lower=0.5,
        upper=1.5,
        rng=urngb,
    )

    assert phz.shape == (scaled_length,)
    assert xpb.all(phz >= 0.5)
    assert xpb.all(phz <= 1.5)
