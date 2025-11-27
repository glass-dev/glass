from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import glass.shapes

if TYPE_CHECKING:
    from types import ModuleType

    from pytest_benchmark.fixture import BenchmarkFixture

    from glass._types import UnifiedGenerator


@pytest.mark.stable
def test_ellipticity_ryden04(
    benchmark: BenchmarkFixture,
    urng: UnifiedGenerator,
    xp: ModuleType,
) -> None:
    """Benchmark for glass.shapes.ellipticity_ryden04."""
    if xp.__name__ == "jax.numpy":
        pytest.skip(
            "Arrays in ellipticity_ryden04 are not immutable, so do not support jax",
        )
    size = (1000, 1000)

    # single ellipticity

    mu = urng.random(size) * -1.0
    sigma = urng.random(size)
    gamma = urng.random(size)
    sigma_gamma = urng.random(size)

    e = benchmark(
        glass.shapes.ellipticity_ryden04, mu, sigma, gamma, sigma_gamma, size=size
    )
    assert e.shape == size


@pytest.mark.stable
def test_ellipticity_gaussian(
    benchmark: BenchmarkFixture,
    xp: ModuleType,
) -> None:
    """Benchmark for glass.shapes.ellipticity_guassian."""
    if xp.__name__ == "jax.numpy":
        pytest.skip(
            "Arrays in ellipticity_gaussian are not immutable, so do not support jax",
        )

    array_length = 10
    n = 1_000_000
    count = xp.full(array_length, fill_value=n)
    sigma = xp.full(array_length, fill_value=0.256)

    eps = benchmark(
        glass.ellipticity_gaussian,
        count,
        sigma,
    )

    assert eps.shape == (n * array_length,)


@pytest.mark.stable
def test_ellipticity_intnorm(
    benchmark: BenchmarkFixture,
    xp: ModuleType,
) -> None:
    """Benchmark for glass.shapes.ellipticity_intnorm."""
    if xp.__name__ == "jax.numpy":
        pytest.skip(
            "Arrays in ellipticity_intnorm are not immutable, so do not support jax",
        )

    array_length = 10
    n = 1_000_000
    count = xp.full(array_length, fill_value=n)
    sigma = xp.full(array_length, fill_value=0.256)

    eps = benchmark(
        glass.ellipticity_intnorm,
        count,
        sigma,
    )

    assert eps.shape == (n * array_length,)
