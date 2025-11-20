from __future__ import annotations

from random import randrange
from typing import TYPE_CHECKING

import numpy as np

import glass.fields

if TYPE_CHECKING:
    from types import ModuleType

    from pytest_benchmark.fixture import BenchmarkFixture


def test_getcl_no_lmax(
    xp: ModuleType,
    benchmark: BenchmarkFixture,
) -> None:
    """Benchmarks for glass.fields.getcl with no value of lmax."""
    range_size = 10
    # make a mock Cls array with the index pairs as entries
    cls = [
        xp.asarray([i, j], dtype=xp.float64)
        for i in range(range_size)
        for j in range(i, -1, -1)
    ]

    random_i = randrange(range_size)
    random_j = randrange(range_size)

    # make sure indices are retrieved correctly
    result = benchmark(
        glass.fields.getcl,
        cls,
        random_i,
        random_j,
    )
    expected = xp.asarray(
        [min(random_i, random_j), max(random_i, random_j)], dtype=xp.float64
    )
    np.testing.assert_allclose(xp.sort(result), expected)


def test_getcl_lmax_0(
    xp: ModuleType,
    benchmark: BenchmarkFixture,
) -> None:
    """Benchmarks for glass.fields.getcl with lmax of 0."""
    range_size = 10
    # make a mock Cls array with the index pairs as entries
    cls = [
        xp.asarray([i, j], dtype=xp.float64)
        for i in range(range_size)
        for j in range(i, -1, -1)
    ]

    random_i = randrange(range_size)
    random_j = randrange(range_size)

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
    np.testing.assert_allclose(result, expected)


def test_getcl_lmax_50(
    xp: ModuleType,
    benchmark: BenchmarkFixture,
) -> None:
    """Benchmarks for glass.fields.getcl with lmax of 50."""
    range_size = 10
    # make a mock Cls array with the index pairs as entries
    cls = [
        xp.asarray([i, j], dtype=xp.float64)
        for i in range(range_size)
        for j in range(i, -1, -1)
    ]

    random_i = randrange(range_size)
    random_j = randrange(range_size)

    # check padding
    result = benchmark(
        glass.fields.getcl,
        cls,
        random_i,
        random_j,
        lmax=50,
    )
    expected = xp.zeros((49,), dtype=xp.float64)
    assert result.size == 51
    np.testing.assert_allclose(result[2:], expected)
