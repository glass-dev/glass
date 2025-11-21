from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import glass._array_comparison as _compare
import glass.arraytools

if TYPE_CHECKING:
    from types import ModuleType

    from pytest_benchmark.fixture import BenchmarkFixture


def test_broadcast_first(xp: ModuleType, benchmark: BenchmarkFixture) -> None:
    """Benchmark test for glass.arraytools.broadcast_first."""
    a = xp.ones((2, 3, 4))
    b = xp.ones((2, 1))

    # arrays with shape ((3, 4, 2)) and ((1, 2)) are passed
    # to np.broadcast_arrays; hence it works
    a_a, b_a = benchmark(glass.arraytools.broadcast_first, a, b)
    assert a_a.shape == (2, 3, 4)
    assert b_a.shape == (2, 3, 4)


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
    ("x_in", "yq_in", "y_out", "shape_out"),
    [
        # test 1d interpolation
        (
            0.5,
            [1.1, 1.2, 1.3, 1.4, 1.5],
            1.15,
            (),
        ),
        (
            [0.5, 1.5, 2.5],
            [1.1, 1.2, 1.3, 1.4, 1.5],
            [1.15, 1.25, 1.35],
            (3,),
        ),
        (
            [[0.5, 1.5], [2.5, 3.5]],
            [1.1, 1.2, 1.3, 1.4, 1.5],
            [[1.15, 1.25], [1.35, 1.45]],
            (2, 2),
        ),
        # test nd interpolation in final axis
        (
            0.5,
            [[1.1, 1.2, 1.3, 1.4, 1.5], [2.1, 2.2, 2.3, 2.4, 2.5]],
            [1.15, 2.15],
            (2,),
        ),
        (
            [0.5, 1.5, 2.5],
            [[1.1, 1.2, 1.3, 1.4, 1.5], [2.1, 2.2, 2.3, 2.4, 2.5]],
            [[1.15, 1.25, 1.35], [2.15, 2.25, 2.35]],
            (2, 3),
        ),
        (
            [[0.5, 1.5], [2.5, 3.5]],
            [[1.1, 1.2, 1.3, 1.4, 1.5], [2.1, 2.2, 2.3, 2.4, 2.5]],
            [[[1.15, 1.25], [1.35, 1.45]], [[2.15, 2.25], [2.35, 2.45]]],
            (2, 2, 2),
        ),
    ],
)
def test_ndinterp_with_default_axis(  # noqa: PLR0913
    xp: ModuleType,
    x_in: int | list[int],
    yq_in: list[int],
    y_out: list[int],
    shape_out: tuple[int, ...],
    benchmark: BenchmarkFixture,
) -> None:
    """Benchmark test for glass.arraytools.ndinterp with the default value for axis."""
    # test 1d interpolation

    xq = xp.asarray([0, 1, 2, 3, 4])
    yq = xp.asarray(yq_in)

    x = xp.asarray(x_in)
    y = benchmark(
        glass.arraytools.ndinterp,
        x,
        xq,
        yq,
    )
    assert y.shape == shape_out
    _compare.assert_allclose(y, y_out, atol=1e-15)


@pytest.mark.parametrize(
    ("x_in", "y_out", "shape_out"),
    [
        (
            0.5,
            [[1.15], [2.15]],
            (2, 1),
        ),
        (
            [0.5, 1.5, 2.5],
            [[[1.15], [1.25], [1.35]], [[2.15], [2.25], [2.35]]],
            (2, 3, 1),
        ),
        (
            [[0.5, 1.5, 2.5, 3.5], [3.5, 2.5, 1.5, 0.5], [0.5, 3.5, 1.5, 2.5]],
            [
                [
                    [[1.15], [1.25], [1.35], [1.45]],
                    [[1.45], [1.35], [1.25], [1.15]],
                    [[1.15], [1.45], [1.25], [1.35]],
                ],
                [
                    [[2.15], [2.25], [2.35], [2.45]],
                    [[2.45], [2.35], [2.25], [2.15]],
                    [[2.15], [2.45], [2.25], [2.35]],
                ],
            ],
            (2, 3, 4, 1),
        ),
    ],
)
def test_ndinterp_nd_interpolation_in_middle_axis(
    xp: ModuleType,
    x_in: int | list[int],
    y_out: list[int],
    shape_out: tuple[int, ...],
    benchmark: BenchmarkFixture,
) -> None:
    """Benchmark test for glass.arraytools.ndinterp setting axis to 1."""
    # test 1d interpolation

    xq = xp.asarray([0, 1, 2, 3, 4])
    yq = xp.asarray(
        [[[1.1], [1.2], [1.3], [1.4], [1.5]], [[2.1], [2.2], [2.3], [2.4], [2.5]]]
    )

    x = xp.asarray(x_in)
    y = benchmark(
        glass.arraytools.ndinterp,
        x,
        xq,
        yq,
        axis=1,
    )
    assert y.shape == shape_out
    _compare.assert_allclose(y, y_out, atol=1e-15)


def test_trapezoid_product(xp: ModuleType, benchmark: BenchmarkFixture) -> None:
    """Benchmark test for glass.arraytools.trapezoid_product."""
    x1 = xp.linspace(0, 2, 100)
    f1 = xp.full_like(x1, 2.0)

    x2 = xp.linspace(1, 2, 10)
    f2 = xp.full_like(x2, 0.5)

    s = benchmark(
        glass.arraytools.trapezoid_product,
        (x1, f1),
        (x2, f2),
    )

    _compare.assert_allclose(s, 1.0)


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
    _compare.assert_allclose(ct, xp.asarray(ct_out))
