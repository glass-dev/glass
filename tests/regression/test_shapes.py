from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import glass

if TYPE_CHECKING:
    from types import ModuleType

    from pytest_benchmark.fixture import BenchmarkFixture

    from glass._types import UnifiedGenerator


@pytest.mark.skipif(
    not hasattr(glass, "ellipticity_ryden04"),
    reason="glass.ellipticity_ryden04 not implemented",
)
@pytest.mark.stable
def test_ellipticity_ryden04(
    benchmark: BenchmarkFixture,
    urng: UnifiedGenerator,
) -> None:
    """Regression test for glass.ellipticity_ryden04."""
    size = (1_000, 1_000)

    # single ellipticity

    mu = urng.random(size) * -1.0
    sigma = urng.random(size)
    gamma = urng.random(size)
    sigma_gamma = urng.random(size)

    e = benchmark(glass.ellipticity_ryden04, mu, sigma, gamma, sigma_gamma, size=size)
    assert e.shape == size


@pytest.mark.skipif(
    not hasattr(glass, "ellipticity_gaussian"),
    reason="glass.ellipticity_gaussian not implemented",
)
@pytest.mark.stable
def test_ellipticity_gaussian(
    benchmark: BenchmarkFixture,
    xp: ModuleType,
) -> None:
    """Regression test for glass.ellipticity_gaussian."""
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


@pytest.mark.skipif(
    not hasattr(glass, "ellipticity_intnorm"),
    reason="glass.ellipticity_intnorm not implemented",
)
@pytest.mark.stable
def test_ellipticity_intnorm(
    benchmark: BenchmarkFixture,
    xp: ModuleType,
) -> None:
    """Regression test for glass.ellipticity_intnorm."""
    array_length = 20
    n = 100_000
    count = xp.full(array_length, fill_value=n)
    sigma = xp.full(array_length, fill_value=0.256)

    eps = benchmark(
        glass.ellipticity_intnorm,
        count,
        sigma,
    )

    assert eps.shape == (n * array_length,)


@pytest.mark.skipif(
    not hasattr(glass, "resample_shapes"),
    reason="glass.resample_shapes not implemented",
)
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
    xp: ModuleType,
    urng: UnifiedGenerator,
) -> None:
    """Regression test for :func:`glass.resample_shapes`."""
    n = 1_000_000

    r = xp.sqrt(urng.uniform(0.0, 1.0, n))
    phi = urng.uniform(0.0, 2 * xp.pi, n)
    epsilon = r * xp.exp(1j * phi)

    result = benchmark(
        glass.resample_shapes,
        epsilon,
        varg=varg,
        vargamma=vargamma,
        rng=urng,
    )

    assert result.shape == epsilon.shape
