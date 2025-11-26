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
    size = (100, 100)

    # single ellipticity

    mu = urng.random(size) * -1.0
    sigma = urng.random(size)
    gamma = urng.random(size)
    sigma_gamma = urng.random(size)

    e = benchmark(
        glass.shapes.ellipticity_ryden04, mu, sigma, gamma, sigma_gamma, size=size
    )
    assert e.shape == size
