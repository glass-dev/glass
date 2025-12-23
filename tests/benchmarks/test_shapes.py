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
    urngb: UnifiedGenerator,
) -> None:
    """Benchmark for glass.ellipticity_ryden04."""
    size = (1_000, 1_000)

    # single ellipticity

    mu = urngb.random(size) * -1.0
    sigma = urngb.random(size)
    gamma = urngb.random(size)
    sigma_gamma = urngb.random(size)

    e = benchmark(glass.ellipticity_ryden04, mu, sigma, gamma, sigma_gamma, size=size)
    assert e.shape == size


@pytest.mark.stable
def test_ellipticity_gaussian(
    benchmark: BenchmarkFixture,
    xpb: ModuleType,
) -> None:
    """Benchmark for glass.ellipticity_gaussian."""
    array_length = 10
    n = 1_000_000
    count = xpb.full(array_length, fill_value=n)
    sigma = xpb.full(array_length, fill_value=0.256)

    eps = benchmark(
        glass.ellipticity_gaussian,
        count,
        sigma,
    )

    assert eps.shape == (n * array_length,)


@pytest.mark.stable
def test_ellipticity_intnorm(
    benchmark: BenchmarkFixture,
    xpb: ModuleType,
) -> None:
    """Benchmark for glass.ellipticity_intnorm."""
    array_length = 10
    n = 1_000_000
    count = xpb.full(array_length, fill_value=n)
    sigma = xpb.full(array_length, fill_value=0.256)

    eps = benchmark(
        glass.ellipticity_intnorm,
        count,
        sigma,
    )

    assert eps.shape == (n * array_length,)
