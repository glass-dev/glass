"""Benchmarks for glass.algorithm."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

import glass.algorithm

if TYPE_CHECKING:
    from types import ModuleType

    from pytest_benchmark.fixture import BenchmarkFixture

    from glass._types import UnifiedGenerator


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


@pytest.mark.parametrize("rtol", [None, 1.0])
def test_cov_clip(
    xp: ModuleType,
    urng: UnifiedGenerator,
    benchmark: BenchmarkFixture,
    rtol: float | None,
) -> None:
    """
    Benchmark test for glass.algorithm.cov_clip.

    Parameterize over rtol to ensure the most coverage possible.
    """
    # prepare a random matrix
    m = urng.random((4, 4))

    # symmetric matrix
    a = (m + m.T) / 2

    # fix by clipping negative eigenvalues
    cov = benchmark(glass.algorithm.cov_clip, a, rtol=rtol)

    # make sure all eigenvalues are positive
    assert xp.all(xp.linalg.eigvalsh(cov) >= 0)

    if rtol is not None:
        h = xp.max(xp.linalg.eigvalsh(a))
        np.testing.assert_allclose(xp.linalg.eigvalsh(cov), h, rtol=1e-6)


@pytest.mark.parametrize("tol", [None, 0.0001])
def test_nearcorr(
    xp: ModuleType,
    benchmark: BenchmarkFixture,
    tol: float | None,
) -> None:
    """
    Benchmark test for glass.algorithm.nearcorr.

    Parameterize over tol to ensure the most coverage possible.
    """
    # from Higham (2002)
    a = xp.asarray(
        [
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
    )
    b = xp.asarray(
        [
            [1.0000, 0.7607, 0.1573],
            [0.7607, 1.0000, 0.7607],
            [0.1573, 0.7607, 1.0000],
        ],
    )

    x = benchmark(glass.algorithm.nearcorr, a, tol=tol)
    np.testing.assert_allclose(x, b, atol=0.0001)


def test_cov_nearest(
    xp: ModuleType,
    urng: UnifiedGenerator,
    benchmark: BenchmarkFixture,
) -> None:
    """Benchmark test for glass.algorithm.cov_nearest."""
    # prepare a random matrix
    m = urng.random((4, 4))

    # symmetric matrix
    a = xp.eye(4) + (m + m.T) / 2

    # compute covariance
    cov = benchmark(glass.algorithm.cov_nearest, a)

    # make sure all eigenvalues are positive
    assert xp.all(xp.linalg.eigvalsh(cov) >= 0)
