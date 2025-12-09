from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

import glass
from tests.conftest import xp_available_backends

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType
    from typing import Any

    from pytest_benchmark.fixture import BenchmarkFixture

    from glass._types import UnifiedGenerator
    from tests.conftest import Compare, DataTransformer, GeneratorConsumer


@pytest.mark.stable
@pytest.mark.parametrize(
    ("bias", "bias_model"),
    [
        (None, lambda x: x),
        (0.8, "linear"),
    ],
)
@pytest.mark.parametrize("remove_monopole", [True, False])
def test_positions_from_delta(  # noqa: PLR0913
    benchmark: BenchmarkFixture,
    data_transformer: DataTransformer,
    generator_consumer: GeneratorConsumer,
    xp: ModuleType,
    bias: float,
    bias_model: str | Callable[[int], int],
    remove_monopole: bool,  # noqa: FBT001
) -> None:
    """Benchmarks for glass.positions_from_delta."""
    if xp.__name__ in {"array_api_strict", "jax.numpy"}:
        pytest.skip(
            f"glass.lensing.multi_plane_matrix not yet ported for {xp.__name__}",
        )
    # create maps that saturate the batching in the function
    nside = 128
    npix = 12 * nside * nside

    scaling_length = 1

    ngal = xp.asarray([1e-3, 2e-3])
    delta = xp.zeros((3 * scaling_length, 1, npix))
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

    lon, lat, cnt = data_transformer.catpos(pos, xp=np)

    assert isinstance(cnt, xp.ndarray)
    assert cnt.shape == (3 * scaling_length, 2)
    assert lon.shape == (cnt.sum(),)
    assert lat.shape == (cnt.sum(),)


@pytest.mark.stable
def test_uniform_positions(
    benchmark: BenchmarkFixture,
    data_transformer: DataTransformer,
    generator_consumer: GeneratorConsumer,
    xp: ModuleType,
) -> None:
    """Benchmarks for glass.uniform_positionsuniform_positions."""
    if xp.__name__ in {"jax.numpy"}:
        pytest.skip(
            f"glass.lensing.multi_plane_matrix not yet ported for {xp.__name__}",
        )

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

    lon, lat, cnt = data_transformer.catpos(pos, xp=xp)
    assert not isinstance(cnt, int)
    assert cnt.__array_namespace__() == xp
    assert cnt.shape == shape_ngal
    assert lon.shape == lat.shape == (xp.sum(cnt),)


@pytest.mark.parametrize(
    ("r_to_alpha", "expected_lon", "expected_lat"),
    [
        # Complex
        (lambda r: r + 0j, 0.0, 5.0),
        (lambda r: -r + 0j, 0.0, -5.0),
        (lambda r: 1j * r, -5.0, 0.0),
        (lambda r: -1j * r, 5.0, 0.0),
        # Real
        (lambda r: [r, 0], 0.0, 5.0),
        (lambda r: [-r, 0], 0.0, -5.0),
        (lambda r: [0, r], -5.0, 0.0),
        (lambda r: [0, -r], 5.0, 0.0),
    ],
)
def test_displace(  # noqa: PLR0913
    benchmark: BenchmarkFixture,
    compare: type[Compare],
    xp: ModuleType,
    r_to_alpha: Callable[[float], complex | list[float]],
    expected_lon: float,
    expected_lat: float,
) -> None:
    """Benchmark for glass.displace with complex values."""
    scale_length = 100_000

    d = 5.0  # deg
    r = d / 180 * xp.pi

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
    compare.assert_allclose(lon, expected_lon, atol=1e-15)
    compare.assert_allclose(lat, expected_lat, atol=1e-15)


def _benchmark_displacement(
    benchmark: BenchmarkFixture,
    urng: UnifiedGenerator,
) -> None:
    """Benchmark logic for glass.displacement."""
    scale_factor = 100

    # test on an array
    from_lon = urng.uniform(-180.0, 180.0, size=(20 * scale_factor, 1))
    from_lat = urng.uniform(-90.0, 90.0, size=(20 * scale_factor, 1))
    to_lon = urng.uniform(-180.0, 180.0, size=5 * scale_factor)
    to_lat = urng.uniform(-90.0, 90.0, size=5 * scale_factor)
    alpha = benchmark(
        glass.displacement,
        from_lon,
        from_lat,
        to_lon,
        to_lat,
    )
    assert alpha.shape == (20 * scale_factor, 5 * scale_factor)


@pytest.mark.stable
@pytest.mark.parametrize(
    "xp",
    [xp for name, xp in xp_available_backends.items() if name != "jax.numpy"],
)
def test_displacement(
    benchmark: BenchmarkFixture,
    urng: UnifiedGenerator,
    xp: ModuleType,  # noqa: ARG001
) -> None:
    """Benchmark for glass.displacement with all backends, but jax."""
    _benchmark_displacement(benchmark, urng)


@pytest.mark.unstable
@pytest.mark.parametrize(
    "xp",
    [xp for name, xp in xp_available_backends.items() if name == "jax.numpy"],
)
def test_displacement_jax(
    benchmark: BenchmarkFixture,
    urng: UnifiedGenerator,
    xp: ModuleType,  # noqa: ARG001
) -> None:
    """Benchmark for glass.displacement with jax."""
    _benchmark_displacement(benchmark, urng)
