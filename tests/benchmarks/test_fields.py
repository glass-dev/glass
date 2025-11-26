from __future__ import annotations

import random
from typing import TYPE_CHECKING

import glass.fields

if TYPE_CHECKING:
    from types import ModuleType

    from conftest import Compare
    from pytest_benchmark.fixture import BenchmarkFixture


def test_getcl_lmax_0(
    benchmark: BenchmarkFixture,
    compare: type[Compare],
    xp: ModuleType,
) -> None:
    """Benchmarks for glass.fields.getcl with lmax of 0."""
    scale_factor = 1_000
    # make a mock Cls array with the index pairs as entries
    cls = [
        xp.asarray([i, j], dtype=xp.float64)
        for i in range(scale_factor)
        for j in range(i, -1, -1)
    ]

    random_i = random.randrange(scale_factor)
    random_j = random.randrange(scale_factor)

    # check slicing
    result = benchmark(
        glass.fields.getcl,
        cls,
        random_i,
        random_j,
        lmax=0,
    )
    expected = xp.asarray([max(random_i, random_j)], dtype=xp.float64)
    assert result.size == 1
    compare.assert_allclose(result, expected)


def test_getcl_lmax_larger_than_cls(
    benchmark: BenchmarkFixture,
    compare: type[Compare],
    xp: ModuleType,
) -> None:
    """Benchmarks for glass.fields.getcl with lmax larger than the length of cl."""
    scale_factor = 1_000
    # make a mock Cls array with the index pairs as entries
    cls = [
        xp.asarray([i, j], dtype=xp.float64)
        for i in range(scale_factor)
        for j in range(i, -1, -1)
    ]

    random_i = random.randrange(scale_factor)
    random_j = random.randrange(scale_factor)

    # check padding
    lmax = scale_factor + 50
    result = benchmark(
        glass.fields.getcl,
        cls,
        random_i,
        random_j,
        lmax=lmax,
    )
    expected = xp.zeros((lmax - 1,), dtype=xp.float64)
    assert result.size == lmax + 1
    compare.assert_allclose(result[2:], expected)
