from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import glass

if TYPE_CHECKING:
    from types import ModuleType

    from pytest_benchmark.fixture import BenchmarkFixture

    from glass._types import UnifiedGenerator


@pytest.mark.stable
def test_ellipticity_ryden04(
    benchmark: BenchmarkFixture,
    urng_benchmarks: UnifiedGenerator,
) -> None:
    """Benchmark for glass.ellipticity_ryden04."""
    size = (1000, 1000)

    # single ellipticity

    mu = urng_benchmarks.random(size) * -1.0
    sigma = urng_benchmarks.random(size)
    gamma = urng_benchmarks.random(size)
    sigma_gamma = urng_benchmarks.random(size)

    e = benchmark(glass.ellipticity_ryden04, mu, sigma, gamma, sigma_gamma, size=size)
    assert e.shape == size


@pytest.mark.stable
def test_ellipticity_gaussian(
    benchmark: BenchmarkFixture,
    xp_benchmarks: ModuleType,
) -> None:
    """Benchmark for glass.ellipticity_guassian."""
    array_length = 10
    n = 1_000_000
    count = xp_benchmarks.full(array_length, fill_value=n)
    sigma = xp_benchmarks.full(array_length, fill_value=0.256)

    eps = benchmark(
        glass.ellipticity_gaussian,
        count,
        sigma,
    )

    assert eps.shape == (n * array_length,)


@pytest.mark.stable
def test_ellipticity_intnorm(
    benchmark: BenchmarkFixture,
    xp_benchmarks: ModuleType,
) -> None:
    """Benchmark for glass.ellipticity_intnorm."""
    array_length = 10
    n = 1_000_000
    count = xp_benchmarks.full(array_length, fill_value=n)
    sigma = xp_benchmarks.full(array_length, fill_value=0.256)

    eps = benchmark(
        glass.ellipticity_intnorm,
        count,
        sigma,
    )

    assert eps.shape == (n * array_length,)
