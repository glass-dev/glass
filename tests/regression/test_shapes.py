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
    """Regression test for glass.ellipticity_ryden04."""
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
    """Regression test for glass.ellipticity_gaussian."""
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
    """Regression test for glass.ellipticity_intnorm."""
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


@pytest.mark.stable
@pytest.mark.parametrize(
    ("varg", "vargamma"),
    [(None, None), (0.1, None), (None, 0.1)],
    ids=["novar", "varg", "vargamma"],
)
def test_resample_shapes(
    varg: float | None,
    vargamma: float | None,
    benchmark: BenchmarkFixture,
    xpb: ModuleType,
    urngb: UnifiedGenerator,
) -> None:
    """Regression test for :func:`glass.resample_shapes`."""
    n = 1_000_000

    r = xpb.sqrt(urngb.uniform(0.0, 1.0, n))
    phi = urngb.uniform(0.0, 2 * xpb.pi, n)
    epsilon = r * xpb.exp(1j * phi)

    result = benchmark(
        glass.resample_shapes,
        epsilon,
        varg=varg,
        vargamma=vargamma,
        rng=urngb,
    )

    assert result.shape == epsilon.shape
