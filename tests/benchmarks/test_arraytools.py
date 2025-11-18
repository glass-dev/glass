from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

import glass.arraytools

if TYPE_CHECKING:
    from types import ModuleType

    from pytest_benchmark.fixture import BenchmarkFixture


def test_broadcast_leading_axes(xp: ModuleType, benchmark: BenchmarkFixture) -> None:
    """Benchmark test for glass.arraytools.broadcast_leading_axes."""
    a_in = 0
    b_in = xp.zeros((4, 10))
    c_in = xp.zeros((3, 1, 5, 6))

    dims, *rest = benchmark(
        glass.arraytools.broadcast_leading_axes,
        (a_in, 0),
        (b_in, 1),
        (c_in, 2),
    )
    a_out, b_out, c_out = rest

    assert dims == (3, 4)
    assert a_out.shape == (3, 4)
    assert b_out.shape == (3, 4, 10)
    assert c_out.shape == (3, 4, 5, 6)


@pytest.mark.parametrize(
    ("f_in", "x_in", "ct_out"),
    [
        # 1D f and x
        (
            [1, 2, 3, 4],
            [0, 1, 2, 3],
            [0.0, 1.5, 4.0, 7.5],
        ),
        # 2D f and 1D x
        (
            [[1, 4, 9, 16], [2, 3, 5, 7]],
            [0, 1, 2.5, 4],
            [[0.0, 2.5, 12.25, 31.0], [0.0, 2.5, 8.5, 17.5]],
        ),
    ],
)
def test_cumulative_trapezoid(
    xp: ModuleType,
    f_in: list[int],
    x_in: list[int],
    ct_out: list[int],
    benchmark: BenchmarkFixture,
) -> None:
    """Benchmark test for glass.arraytools.cumulative_trapezoid."""
    # 1D f and x

    f = xp.asarray(f_in)
    x = xp.asarray(x_in)

    ct = benchmark(glass.arraytools.cumulative_trapezoid, f, x)
    np.testing.assert_allclose(ct, xp.asarray(ct_out))
