from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import glass.arraytools

if TYPE_CHECKING:
    from types import ModuleType

    from pytest_benchmark.fixture import BenchmarkFixture


def test_broadcast_leading_axes(xp: ModuleType, benchmark: BenchmarkFixture) -> None:
    """Benchmark test for glass.arraytools.broadcast_leading_axes."""
    a_in = 0
    b_in = xp.zeros((40, 10))
    c_in = xp.zeros((300, 1, 50, 60))

    dims, *rest = benchmark(
        glass.arraytools.broadcast_leading_axes,
        (a_in, 0),
        (b_in, 1),
        (c_in, 2),
    )
    a_out, b_out, c_out = rest

    assert dims == (300, 40)
    assert a_out.shape == (300, 40)
    assert b_out.shape == (300, 40, 10)
    assert c_out.shape == (300, 40, 50, 60)


def test_cumulative_trapezoid_1d(
    xp: ModuleType,
    benchmark: BenchmarkFixture,
) -> None:
    """Benchmark test for glass.arraytools.cumulative_trapezoid."""
    f = xp.arange(5001)[1:]  # [1, 2, 3, 4,...]
    x = xp.arange(5000)  # [0, 1, 2, 3,...]

    expected_first_4_out = [0.0, 1.5, 4.0, 7.5]

    ct = benchmark(glass.arraytools.cumulative_trapezoid, f, x)
    np.testing.assert_allclose(ct[:4], xp.asarray(expected_first_4_out))


def test_cumulative_trapezoid_2d(
    xp: ModuleType,
    benchmark: BenchmarkFixture,
) -> None:
    """Benchmark test for glass.arraytools.cumulative_trapezoid."""
    f = xp.stack(
        [  # [[1, 2, 3, 4,...], [1, 2, 3, 4,...]]
            xp.arange(2501)[1:],
            xp.arange(2501)[1:],
        ]
    )
    x = xp.arange(2500)  # [0, 1, 2, 3,...]

    expected_first_4_out = [0.0, 1.5, 4.0, 7.5]

    ct = benchmark(glass.arraytools.cumulative_trapezoid, f, x)
    np.testing.assert_allclose(ct[0, :4], xp.asarray(expected_first_4_out))
    np.testing.assert_allclose(ct[1, :4], xp.asarray(expected_first_4_out))
