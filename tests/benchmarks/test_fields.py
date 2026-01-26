from __future__ import annotations

import random
from typing import TYPE_CHECKING

import pytest

import glass
import glass.fields
import glass.healpix as hp

if TYPE_CHECKING:
    from types import ModuleType
    from typing import Any

    from conftest import Compare, GeneratorConsumer
    from pytest_benchmark.fixture import BenchmarkFixture

    from glass._types import AngularPowerSpectra, UnifiedGenerator


@pytest.mark.stable
def test_iternorm_no_size(
    benchmark: BenchmarkFixture,
    generator_consumer: GeneratorConsumer,
    xpb: ModuleType,
) -> None:
    """Benchmarks for glass.iternorm with default value for size."""
    k = 2
    array_in = [xpb.asarray(x) for x in xpb.arange(10_000, dtype=xpb.float64)]

    def function_to_benchmark() -> list[Any]:
        generator = glass.iternorm(k, iter(array_in))
        return generator_consumer.consume(  # type: ignore[no-any-return]
            generator,
            valid_exception="covariance matrix is not positive definite",
        )

    results = benchmark(function_to_benchmark)
    j, a, s = results[0]

    assert isinstance(j, int)
    assert a.shape == (k,)
    assert s.shape == ()
    assert s.dtype == xpb.float64
    assert s.shape == ()


@pytest.mark.stable
@pytest.mark.parametrize("num_dimensions", [1, 2])
def test_iternorm_specify_size(
    benchmark: BenchmarkFixture,
    generator_consumer: GeneratorConsumer,
    xpb: ModuleType,
    num_dimensions: int,
) -> None:
    """Benchmarks for glass.iternorm with size specified."""
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
    array_in = [xpb.asarray(arr, dtype=xpb.float64) for arr in list_input]
    expected_result = [
        1,
        (3, 2),
        (3,),
    ]

    def function_to_benchmark() -> list[Any]:
        generator = glass.iternorm(k, iter(array_in), size)
        return generator_consumer.consume(  # type: ignore[no-any-return]
            generator,
            valid_exception="covariance matrix is not positive definite",
        )

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
def test_iternorm_k_0(
    benchmark: BenchmarkFixture,
    compare: Compare,
    generator_consumer: GeneratorConsumer,
    xpb: ModuleType,
) -> None:
    """Benchmarks for glass.iternorm with k set to 0."""
    k = 0
    array_in = [xpb.stack([x]) for x in xpb.ones(1_000, dtype=xpb.float64)]

    def function_to_benchmark() -> list[Any]:
        generator = glass.iternorm(k, iter(array_in))
        return generator_consumer.consume(generator)  # type: ignore[no-any-return]

    results = benchmark(function_to_benchmark)

    j, a, s = results[0]
    assert j is None
    assert a.shape == (0,)
    compare.assert_allclose(xpb.asarray(s), 1.0)


@pytest.mark.stable
def test_cls2cov(
    benchmark: BenchmarkFixture,
    compare: Compare,
    generator_consumer: GeneratorConsumer,
    xpb: ModuleType,
) -> None:
    """Benchmarks for glass.cls2cov."""
    nl, nf, nc = 3, 2, 2
    array_in = [xpb.arange(i + 1.0, i + 4.0) for i in range(1_000)]

    def function_to_benchmark() -> list[Any]:
        generator = glass.cls2cov(
            array_in,
            nl,
            nf,
            nc,
        )
        return generator_consumer.consume(generator)  # type: ignore[no-any-return]

    covs = benchmark(function_to_benchmark)
    cov = covs[0]

    assert cov.shape == (nl, nc + 1)
    assert cov.dtype == xpb.float64

    compare.assert_allclose(cov[:, 0], xpb.asarray([1.0, 1.5, 2.0]))
    compare.assert_allclose(cov[:, 1], xpb.asarray([1.5, 2.0, 2.5]))
    compare.assert_allclose(cov[:, 2], 0)


@pytest.mark.stable
@pytest.mark.parametrize("use_rng", [False, True])
@pytest.mark.parametrize("ncorr", [None, 1])
def test_generate_grf(
    benchmark: BenchmarkFixture,
    generator_consumer: GeneratorConsumer,
    urngb: UnifiedGenerator,
    use_rng: bool,  # noqa: FBT001
    ncorr: int | None,
) -> None:
    """Benchmarks for glass.fields._generate_grf with positional arguments only."""
    gls: AngularPowerSpectra = [urngb.random(1_000)]
    nside = 4

    def function_to_benchmark() -> list[Any]:
        generator = glass.fields._generate_grf(
            gls,
            nside,
            rng=urngb if use_rng else None,
            ncorr=ncorr,
        )
        return generator_consumer.consume(generator)  # type: ignore[no-any-return]

    gaussian_fields = benchmark(function_to_benchmark)

    assert gaussian_fields[0].shape == (hp.nside2npix(nside),)


@pytest.mark.stable
@pytest.mark.parametrize("ncorr", [None, 1])
def test_generate(
    benchmark: BenchmarkFixture,
    compare: Compare,
    generator_consumer: GeneratorConsumer,
    xpb: ModuleType,
    ncorr: int | None,
) -> None:
    """Benchmarks for glass.generate."""
    if xpb.__name__ == "array_api_strict":
        pytest.skip(f"glass.generate not yet ported for {xpb.__name__}")

    n = 100
    fields = [lambda x, var: x for _ in range(n)]  # noqa: ARG005
    fields[1] = lambda x, var: x**2  # noqa: ARG005
    nth_triangular_number = int((n * (n + 1)) / 2)
    gls = [xpb.ones(10) for _ in range(nth_triangular_number)]
    nside = 16

    def function_to_benchmark() -> list[Any]:
        generator = glass.generate(
            fields,  # type: ignore[arg-type]
            gls,
            nside=nside,
            ncorr=ncorr,
        )
        return generator_consumer.consume(  # type: ignore[no-any-return]
            generator,
            valid_exception="covariance matrix is not positive definite",
        )

    result = benchmark(function_to_benchmark)

    for field in result:
        assert field.shape == (hp.nside2npix(nside),)
    compare.assert_allclose(result[1], result[0] ** 2, atol=1e-05)


@pytest.mark.unstable
def test_getcl_lmax_0(
    benchmark: BenchmarkFixture,
    compare: type[Compare],
    xpb: ModuleType,
) -> None:
    """Benchmarks for glass.getcl with lmax of 0."""
    scale_factor = 1_000
    # make a mock Cls array with the index pairs as entries
    cls = [
        xpb.asarray([i, j], dtype=xpb.float64)
        for i in range(scale_factor)
        for j in range(i, -1, -1)
    ]

    random_i = random.randrange(scale_factor)
    random_j = random.randrange(scale_factor)

    # check slicing
    result = benchmark(
        glass.getcl,
        cls,
        random_i,
        random_j,
        lmax=0,
    )
    expected = xpb.asarray([max(random_i, random_j)], dtype=xpb.float64)
    assert result.size == 1
    compare.assert_allclose(result, expected)


@pytest.mark.unstable
def test_getcl_lmax_larger_than_cls(
    benchmark: BenchmarkFixture,
    compare: type[Compare],
    xpb: ModuleType,
) -> None:
    """Benchmarks for glass.getcl with lmax larger than the length of cl."""
    scale_factor = 1_000
    # make a mock Cls array with the index pairs as entries
    cls = [
        xpb.asarray([i, j], dtype=xpb.float64)
        for i in range(scale_factor)
        for j in range(i, -1, -1)
    ]

    random_i = random.randrange(scale_factor)
    random_j = random.randrange(scale_factor)

    # check padding
    lmax = scale_factor + 50
    result = benchmark(
        glass.getcl,
        cls,
        random_i,
        random_j,
        lmax=lmax,
    )
    expected = xpb.zeros((lmax - 1,), dtype=xpb.float64)
    assert result.size == lmax + 1
    compare.assert_allclose(result[2:], expected)
