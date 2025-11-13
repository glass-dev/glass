from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

import glass.fields

if TYPE_CHECKING:
    import types

    from pytest_benchmark.fixture import BenchmarkFixture


def test_iternorm_no_size(xp: types.ModuleType, benchmark: BenchmarkFixture) -> None:
    """Benchmarks for glass.iternorm with default value for size."""
    # Call jax version of iternorm once jax version is written
    if xp.__name__ == "jax.numpy":
        pytest.skip("Arrays in iternorm are not immutable, so do not support jax")

    # check output shapes and types

    k = 2

    generator = benchmark(
        glass.fields.iternorm,
        k,
        (xp.asarray(x) for x in [1.0, 0.5, 0.5, 0.5, 0.2, 0.1, 0.5, 0.1, 0.2]),
    )
    result = next(generator)

    j, a, s = result

    assert isinstance(j, int)
    assert a.shape == (k,)
    assert s.shape == ()
    assert s.dtype == xp.float64
    assert s.shape == ()


@pytest.mark.parametrize(
    ("k", "size", "array_in", "expected_result"),
    [
        (
            2,
            3,
            [
                [1.0, 0.5, 0.5],
                [0.5, 0.2, 0.1],
                [0.5, 0.1, 0.2],
            ],
            [
                1,
                (3, 2),
                (3,),
            ],
        ),
        (
            2,
            (3,),
            [
                [
                    [1.0, 0.5, 0.5],
                    [0.5, 0.2, 0.1],
                    [0.5, 0.1, 0.2],
                ],
                [
                    [2.0, 1.0, 0.8],
                    [1.0, 0.5, 0.3],
                    [0.8, 0.3, 0.6],
                ],
            ],
            [
                1,
                (3, 2),
                (3,),
            ],
        ),
    ],
)
def test_iternorm_specify_size(  # noqa: PLR0913
    xp: types.ModuleType,
    k: int,
    size: int,
    array_in: list[int],
    expected_result: list[int],
    benchmark: BenchmarkFixture,
) -> None:
    """Benchmarks for glass.iternorm with size specified."""
    # Call jax version of iternorm once jax version is written
    if xp.__name__ == "jax.numpy":
        pytest.skip("Arrays in iternorm are not immutable, so do not support jax")

    # check output shapes and types

    generator = benchmark(
        glass.fields.iternorm,
        k,
        (xp.asarray(arr) for arr in array_in),
        size,
    )
    result1 = next(generator)
    result2 = next(generator)

    assert result1 != result2
    assert isinstance(result1, tuple)
    assert len(result1) == 3

    j, a, s = result1
    assert isinstance(j, int)
    assert j == expected_result[0]
    assert a.shape == expected_result[1]
    assert s.shape == expected_result[2]


def test_iternorm_k_0(xp: types.ModuleType, benchmark: BenchmarkFixture) -> None:
    """Benchmarks for glass.iternorm with k set to 0."""
    # Call jax version of iternorm once jax version is written
    if xp.__name__ == "jax.numpy":
        pytest.skip("Arrays in iternorm are not immutable, so do not support jax")

    # test k = 0

    generator = benchmark(glass.fields.iternorm, 0, (xp.asarray([x]) for x in [1.0]))

    j, a, s = next(generator)
    assert j is None
    assert a.shape == (0,)
    np.testing.assert_allclose(xp.asarray(s), 1.0)
