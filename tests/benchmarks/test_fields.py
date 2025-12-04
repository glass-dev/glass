from __future__ import annotations

import random
from typing import TYPE_CHECKING

import healpy as hp
import pytest

import glass
import glass.fields

if TYPE_CHECKING:
    from types import ModuleType
    from typing import Any

    from conftest import Compare, GeneratorConsumer
    from pytest_benchmark.fixture import BenchmarkFixture

    from glass._types import UnifiedGenerator


@pytest.mark.stable
def test_iternorm_no_size(
    benchmark: BenchmarkFixture,
    generator_consumer: GeneratorConsumer,
    xp: ModuleType,
) -> None:
    """Benchmarks for glass.iternorm with default value for size."""
    # Call jax version of iternorm once jax version is written
    if xp.__name__ == "jax.numpy":
        pytest.skip("Arrays in iternorm are not immutable, so do not support jax")

    # check output shapes and types

    k = 2
    array_in = [xp.asarray(x) for x in xp.arange(10_000, dtype=xp.float64)]

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
    assert s.dtype == xp.float64
    assert s.shape == ()


@pytest.mark.stable
@pytest.mark.parametrize("num_dimensions", [1, 2])
def test_iternorm_specify_size(
    benchmark: BenchmarkFixture,
    generator_consumer: GeneratorConsumer,
    xp: ModuleType,
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
    xp: ModuleType,
) -> None:
    """Benchmarks for glass.iternorm with k set to 0."""
    # Call jax version of iternorm once jax version is written
    if xp.__name__ == "jax.numpy":
        pytest.skip("Arrays in iternorm are not immutable, so do not support jax")

    k = 0
    array_in = [xp.stack([x]) for x in xp.ones(1_000, dtype=xp.float64)]

    def function_to_benchmark() -> list[Any]:
        generator = glass.iternorm(k, iter(array_in))
        return generator_consumer.consume(generator)  # type: ignore[no-any-return]

    results = benchmark(function_to_benchmark)

    j, a, s = results[0]
    assert j is None
    assert a.shape == (0,)
    compare.assert_allclose(xp.asarray(s), 1.0)


@pytest.mark.stable
def test_cls2cov(
    benchmark: BenchmarkFixture,
    compare: Compare,
    generator_consumer: GeneratorConsumer,
    urng: UnifiedGenerator,
    xp: ModuleType,
) -> None:
    """Benchmarks for glass.cls2cov."""
    # Call jax version of iternorm once jax version is written
    if xp.__name__ == "jax.numpy":
        pytest.skip("Arrays in cls2cov are not immutable, so do not support jax")

    # check output values and shape

    nl, nf, nc = 3, 2, 2
    array_in = [urng.random(3) for _ in range(1_000)]

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
    assert cov.dtype == xp.float64

    compare.assert_allclose(
        cov[:, 0],
        xp.asarray([0.348684, 0.047089, 0.487811]),
        atol=1e-6,
    )
    compare.assert_allclose(
        cov[:, 1],
        [0.38057, 0.393032, 0.064057],
        atol=1e-6,
    )
    compare.assert_allclose(cov[:, 2], 0)


@pytest.mark.stable
@pytest.mark.parametrize("use_rng", [False, True])
@pytest.mark.parametrize("ncorr", [None, 1])
def test_generate_grf(  # noqa: PLR0913
    xp: ModuleType,
    benchmark: BenchmarkFixture,
    generator_consumer: GeneratorConsumer,
    urng: UnifiedGenerator,
    use_rng: bool,  # noqa: FBT001
    ncorr: int | None,
) -> None:
    """Benchmarks for glass.fields._generate_grf with positional arguments only."""
    if xp.__name__ in {"array_api_strict", "jax.numpy"}:
        pytest.skip(f"glass.fields._generate_grf not yet ported for {xp.__name__}")

    gls = [urng.random(1_000)]
    nside = 4

    def function_to_benchmark() -> list[Any]:
        generator = glass.fields._generate_grf(
            gls,
            nside,
            rng=urng if use_rng else None,  # type: ignore[arg-type]
            ncorr=ncorr,
        )
        return generator_consumer.consume(generator)  # type: ignore[no-any-return]

    gaussian_fields = benchmark(function_to_benchmark)

    assert gaussian_fields[0].shape == (hp.nside2npix(nside),)


@pytest.mark.stable
@pytest.mark.parametrize(
    ("ncorr", "expected_len"),
    [
        (None, 4),
        (1, 2),
    ],
)
def test_generate(  # noqa: PLR0913
    benchmark: BenchmarkFixture,
    compare: Compare,
    generator_consumer: GeneratorConsumer,
    xp: ModuleType,
    expected_len: int,
    ncorr: int | None,
) -> None:
    """Benchmarks for glass.generate."""
    if xp.__name__ in {"array_api_strict", "jax.numpy"}:
        pytest.skip(f"glass.generate not yet ported for {xp.__name__}")

    n = 100
    fields = [lambda x, var: x for _ in range(n)]  # noqa: ARG005
    fields[1] = lambda x, var: x**2  # noqa: ARG005
    nth_triangular_number = int((n * (n + 1)) / 2)
    gls = [xp.ones(10) for _ in range(nth_triangular_number)]
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

    assert len(result) == expected_len
    for field in result:
        assert field.shape == (hp.nside2npix(nside),)
    compare.assert_allclose(result[1], result[0] ** 2, atol=1e-05)


@pytest.mark.unstable
def test_getcl_lmax_0(
    benchmark: BenchmarkFixture,
    compare: type[Compare],
    xp: ModuleType,
) -> None:
    """Benchmarks for glass.getcl with lmax of 0."""
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
        glass.getcl,
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
    """Benchmarks for glass.getcl with lmax larger than the length of cl."""
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
        glass.getcl,
        cls,
        random_i,
        random_j,
        lmax=lmax,
    )
    expected = xp.zeros((lmax - 1,), dtype=xp.float64)
    assert result.size == lmax + 1
    compare.assert_allclose(result[2:], expected)
