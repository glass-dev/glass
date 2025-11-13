"""Benchmarks for glass.algorithm."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import glass.algorithm

if TYPE_CHECKING:
    from types import ModuleType

    from pytest_benchmark.fixture import BenchmarkFixture


def test_nnls(xp: ModuleType, benchmark: BenchmarkFixture) -> None:
    """
    Benchmark for glass.algorithm.nnls.

    Notes
    -----
        We need a test which covers more of glass.algorithm.nnls.

    """
    if xp.__name__ == "jax.numpy":
        pytest.skip("Arrays in nnls are not immutable, so do not support jax")

    a = xp.reshape(xp.arange(25.0), (-1, 5))
    b = xp.arange(5.0)
    y = a @ b
    res = benchmark(glass.algorithm.nnls, a, y)
    assert xp.linalg.vector_norm((a @ res) - y) < 1e-7
