from __future__ import annotations

import random
from typing import TYPE_CHECKING

import pytest

import glass.fields

if TYPE_CHECKING:
    from collections.abc import Generator
    from types import ModuleType
    from typing import Any

    from conftest import Compare
    from pytest_benchmark.fixture import BenchmarkFixture

    from glass._types import FloatArray, UnifiedGenerator


def _consume_generator(
    generator: Generator[Any],
) -> list[Any]:
    """
    Generate and consume a generator returned by a given functions.

    The resulting generator will be consumed an any ValueError
    exceptions swallowed.
    """
    output = []
    try:
        # Consume in a loop, as we expect users to
        for result in generator:
            output.append(result)  # noqa: PERF402
    except ValueError:
        pass
    return output


@pytest.mark.stable
def test_iternorm_no_size(xp: ModuleType, benchmark: BenchmarkFixture) -> None:
    """Benchmarks for glass.iternorm with default value for size."""
    # Call jax version of iternorm once jax version is written
    if xp.__name__ == "jax.numpy":
        pytest.skip("Arrays in iternorm are not immutable, so do not support jax")

    # check output shapes and types

    k = 2
    array_in = [xp.asarray(x) for x in xp.arange(10_000, dtype=xp.float64)]

    def function_to_benchmark() -> list[tuple[int | None, FloatArray, FloatArray]]:
        generator = glass.fields.iternorm(
            k,
            (x for x in array_in),
        )
        return _consume_generator(generator)

    results = benchmark(function_to_benchmark)
    j, a, s = results[0]

    assert isinstance(j, int)
    assert a.shape == (k,)
    assert s.shape == ()
    assert s.dtype == xp.float64
    assert s.shape == ()


@pytest.mark.stable
@pytest.mark.parametrize("num_dimensions", [1, 2])
def test_iternorm_specify_size(
    xp: ModuleType,
    benchmark: BenchmarkFixture,
    num_dimensions: int,
) -> None:
    """Benchmarks for glass.iternorm with size specified."""
    # Call jax version of iternorm once jax version is written
    if xp.__name__ == "jax.numpy":
        pytest.skip("Arrays in iternorm are not immutable, so do not support jax")

    k = 2
    size = (3,)
    if num_dimensions == 1:
        list_input = [[1.0, 0.5, 0.5] for _ in range(10_000)]
    elif num_dimensions == 2:
        list_input = [
            [
                [1.0, 0.5, 0.5],  # type: ignore[list-item]
                [0.5, 0.2, 0.1],  # type: ignore[list-item]
                [0.5, 0.1, 0.2],  # type: ignore[list-item]
            ]
            for _ in range(10_000)
        ]
    array_in = [xp.asarray(arr, dtype=xp.float64) for arr in list_input]
    expected_result = [
        1,
        (3, 2),
        (3,),
    ]

    def function_to_benchmark() -> list[tuple[int | None, FloatArray, FloatArray]]:
        generator = glass.fields.iternorm(
            k,
            (x for x in array_in),
            size,
        )
        return _consume_generator(generator)

    # check output shapes and types

    results = benchmark(function_to_benchmark)
    result1 = results[0]
    result2 = results[1]

    assert result1 != result2
    assert isinstance(result1, tuple)
    assert len(result1) == 3

    j, a, s = result1
    assert isinstance(j, int)
    assert j == expected_result[0]
    assert a.shape == expected_result[1]
    assert s.shape == expected_result[2]


@pytest.mark.stable
def test_iternorm_k_0(xp: ModuleType, benchmark: BenchmarkFixture) -> None:
    """Benchmarks for glass.iternorm with k set to 0."""
    # Call jax version of iternorm once jax version is written
    if xp.__name__ == "jax.numpy":
        pytest.skip("Arrays in iternorm are not immutable, so do not support jax")

    k = 0
    array_in = [xp.stack([x]) for x in xp.ones(1_000, dtype=xp.float64)]

    def function_to_benchmark() -> list[tuple[int | None, FloatArray, FloatArray]]:
        generator = glass.fields.iternorm(
            k,
            (x for x in array_in),
        )
        return _consume_generator(generator)

    results = benchmark(function_to_benchmark)

    j, a, s = results[0]
    assert j is None
    assert a.shape == (0,)
    np.testing.assert_allclose(xp.asarray(s), 1.0)


@pytest.mark.stable
def test_cls2cov(
    xp: ModuleType,
    benchmark: BenchmarkFixture,
    urng: UnifiedGenerator,
) -> None:
    """Benchmarks for glass.cls2cov."""
    # Call jax version of iternorm once jax version is written
    if xp.__name__ == "jax.numpy":
        pytest.skip("Arrays in cls2cov are not immutable, so do not support jax")

    # check output values and shape

    nl, nf, nc = 3, 2, 2
    array_in = [urng.random(3) for _ in range(1_000)]

    def function_to_benchmark() -> list[tuple[int | None, FloatArray, FloatArray]]:
        generator = glass.fields.cls2cov(
            array_in,
            nl,
            nf,
            nc,
        )
        return _consume_generator(generator)

    covs = benchmark(function_to_benchmark)
    cov = covs[0]

    assert cov.shape == (nl, nc + 1)
    assert cov.dtype == xp.float64

    np.testing.assert_allclose(
        cov[:, 0],
        xp.asarray([0.348684, 0.047089, 0.487811]),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        cov[:, 1],
        [0.38057, 0.393032, 0.064057],
        atol=1e-6,
    )
    np.testing.assert_allclose(cov[:, 2], 0)


@pytest.mark.unstable
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


@pytest.mark.unstable
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
