from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

import glass

if TYPE_CHECKING:
    from collections.abc import Callable, Generator
    from types import ModuleType
    from typing import Any

    from conftest import GeneratorConsumer
    from pytest_benchmark.fixture import BenchmarkFixture

    from glass._types import FloatArray, IntArray


def catpos(
    pos: Generator[
        tuple[
            FloatArray,
            FloatArray,
            IntArray,
        ]
    ],
    *,
    xp: ModuleType,
) -> tuple[
    FloatArray,
    FloatArray,
    IntArray,
]:
    """Concatenate an array of pos into three arrays lon, lat and count."""
    lon = xp.empty(0)
    lat = xp.empty(0)
    cnt: IntArray = 0
    for lo, la, co in pos:
        lon = xp.concat([lon, lo])
        lat = xp.concat([lat, la])
        cnt = cnt + co
    return lon, lat, cnt


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
    generator_consumer: GeneratorConsumer,
    xp: ModuleType,
    bias: float,
    bias_model: str | Callable[[int], int],
    remove_monopole: bool,  # noqa: FBT001
) -> None:
    """Benchmarks for glass.positions_from_delta."""
    if xp.__name__ in {"array_api_strict", "jax.numpy"}:
        pytest.skip(
            f"glass.lensing.multi_plane_matrix not yet ported for {xp.__name__}"
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
        return generator_consumer.consume(generator)  # type: ignore[no-any-return]

    pos = benchmark(function_to_benchmark)

    lon, lat, cnt = catpos(pos, xp=np)

    assert isinstance(cnt, xp.ndarray)
    assert cnt.shape == (3 * scaling_length, 2)
    assert lon.shape == (cnt.sum(),)
    assert lat.shape == (cnt.sum(),)
