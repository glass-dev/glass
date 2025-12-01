from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import glass.arraytools

if TYPE_CHECKING:
    from types import ModuleType

    from pytest_benchmark.fixture import BenchmarkFixture

    from tests.conftest import Compare


@pytest.mark.unstable
def test_broadcast_leading_axes(
    benchmark: BenchmarkFixture,
    xp: ModuleType,
) -> None:
    """Benchmark test for glass.arraytools.broadcast_leading_axes."""
    # Ensure we don't use too much memory
    a_in = 0
    b_shape = (4, 10)
    c_shape = (30, 1, 5, 6)
    b_in = xp.zeros(b_shape)
    c_in = xp.zeros(c_shape)

    dims, *rest = benchmark(
        glass.arraytools.broadcast_leading_axes,
        (a_in, 0),
        (b_in, 1),
        (c_in, 2),
    )
    a_out, b_out, c_out = rest

    assert dims == (c_shape[0], b_shape[0])
    assert a_out.shape == (c_shape[0], b_shape[0])
    assert b_out.shape == (c_shape[0], b_shape[0], b_shape[1])
    assert c_out.shape == (c_shape[0], b_shape[0], c_shape[2], c_shape[3])


@pytest.mark.unstable
def test_cumulative_trapezoid_1d(
    benchmark: BenchmarkFixture,
    compare: type[Compare],
    xp: ModuleType,
) -> None:
    """Benchmark test for glass.arraytools.cumulative_trapezoid."""
    scaled_length = 10_000

    f = xp.arange(scaled_length + 1)[1:]  # [1, 2, 3, 4,...]
    x = xp.arange(scaled_length)  # [0, 1, 2, 3,...]

    expected_first_4_out = [0.0, 1.5, 4.0, 7.5]

    ct = benchmark(glass.arraytools.cumulative_trapezoid, f, x)
    compare.assert_allclose(ct[:4], xp.asarray(expected_first_4_out))


@pytest.mark.unstable
def test_cumulative_trapezoid_2d(
    benchmark: BenchmarkFixture,
    compare: type[Compare],
    xp: ModuleType,
) -> None:
    """Benchmark test for glass.arraytools.cumulative_trapezoid."""
    scaled_length = 5_000

    f = xp.stack(
        [  # [[1, 2, 3, 4,...], [1, 2, 3, 4,...]]
            xp.arange(scaled_length + 1)[1:],
            xp.arange(scaled_length + 1)[1:],
        ]
    )
    x = xp.arange(scaled_length)  # [0, 1, 2, 3,...]

    expected_first_4_out = [0.0, 1.5, 4.0, 7.5]

    ct = benchmark(glass.arraytools.cumulative_trapezoid, f, x)
    compare.assert_allclose(ct[0, :4], xp.asarray(expected_first_4_out))
    compare.assert_allclose(ct[1, :4], xp.asarray(expected_first_4_out))
