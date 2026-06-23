from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest

import glass

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType
    from typing import Any

    from pytest_benchmark.fixture import BenchmarkFixture

    from glass._types import UnifiedGenerator
    from tests.fixtures.helper_classes import (
        DataTransformer,
        GeneratorConsumer,
    )


@pytest.mark.stable
@pytest.mark.parametrize(
    ("bias", "bias_model"),
    [
        (None, lambda x: x),
        (0.8, glass.linear_bias),
    ],
)
@pytest.mark.parametrize("remove_monopole", [True, False])
def test_positions_from_delta(  # noqa: PLR0913
    benchmark: BenchmarkFixture,
    data_transformer: type[DataTransformer],
    generator_consumer: type[GeneratorConsumer],
    xp: ModuleType,
    bias: float,
    bias_model: Callable[[int], int],
    remove_monopole: bool,  # noqa: FBT001
) -> None:
    """Regression tests for glass.positions_from_delta."""
    nside = 48
    npix = 12 * nside * nside

    ngal = xp.asarray([i * 1e-3 for i in range(10)])
    delta = xp.zeros((9, 1, npix))
    vis = xp.ones(npix)

    def function_to_benchmark() -> list[Any]:
        generator = glass.positions_from_delta(
            ngal,
            delta,
            bias,
            vis,
            remove_monopole=remove_monopole,
            bias_model=bias_model,
        )
        return generator_consumer.consume(generator)

    pos = benchmark(function_to_benchmark)

    lon, lat, count = data_transformer.catpos(pos, xp=xp)

    assert isinstance(count, xp.ndarray)
    assert count.shape == (9, 10)
    assert lon.shape == (xp.sum(count),)
    assert lat.shape == (xp.sum(count),)


@pytest.mark.stable
def test_uniform_positions(
    benchmark: BenchmarkFixture,
    data_transformer: type[DataTransformer],
    generator_consumer: type[GeneratorConsumer],
    xp: ModuleType,
) -> None:
    """Regression tests for glass.uniform_positionsuniform_positions."""
    scaling_factor = 12
    shape_ngal = (int(scaling_factor / 2), 2)

    ngal = xp.asarray([[1e-3, 2e-3], [3e-3, 4e-3], [5e-3, 6e-3]])
    ngal = xp.reshape(
        xp.arange(1e-3, 1e-3 * scaling_factor + 1e-3, 1e-3),
        shape=shape_ngal,
    )

    def function_to_benchmark() -> list[Any]:
        generator = glass.uniform_positions(ngal)
        return generator_consumer.consume(generator)

    pos = benchmark(function_to_benchmark)

    lon, lat, count = data_transformer.catpos(pos, xp=xp)
    assert count.__array_namespace__() == xp
    assert count.shape == shape_ngal
    assert lon.shape == lat.shape == (xp.sum(count),)


@pytest.mark.parametrize(
    ("r_to_alpha"),
    [
        # Complex
        (lambda r: r + 0j),
        # Real
        (lambda r: [r, 0]),
    ],
)
@pytest.mark.skipif(
    not hasattr(glass, "displace"),
    reason="test requires glass.displace",
)
def test_displace(
    benchmark: BenchmarkFixture,
    xp: ModuleType,
    r_to_alpha: Callable[[float], complex | list[float]],
) -> None:
    """Regression test for glass.displace with complex values."""
    scale_length = 100_000

    d = 5.0  # deg
    r = d / 180 * math.pi

    # displace the origin so everything is easy
    lon0 = xp.asarray(xp.zeros(scale_length, dtype=xp.float64))
    lat0 = xp.asarray(xp.zeros(scale_length, dtype=xp.float64))
    alpha = xp.asarray(r_to_alpha(r))

    lon, lat = benchmark(
        glass.displace,
        lon0,
        lat0,
        alpha,
    )
    assert lon.shape == (scale_length,)
    assert lat.shape == (scale_length,)


@pytest.mark.stable
@pytest.mark.skipif(
    not hasattr(glass, "displacement"),
    reason="test requires glass.displacement",
)
def test_displacement(
    benchmark: BenchmarkFixture,
    urngb: UnifiedGenerator,
) -> None:
    """Regression test for glass.displacement."""
    scale_factor = 100

    # test on an array
    from_lon = urngb.uniform(-180.0, 180.0, size=(20 * scale_factor, 1))
    from_lat = urngb.uniform(-90.0, 90.0, size=(20 * scale_factor, 1))
    to_lon = urngb.uniform(-180.0, 180.0, size=5 * scale_factor)
    to_lat = urngb.uniform(-90.0, 90.0, size=5 * scale_factor)
    alpha = benchmark(
        glass.displacement,
        from_lon,
        from_lat,
        to_lon,
        to_lat,
    )
    assert alpha.shape == (20 * scale_factor, 5 * scale_factor)
