from __future__ import annotations

import random
from typing import TYPE_CHECKING

import pytest

import array_api_extra as xpx

import glass
import glass.fields
import glass.healpix as hp

if TYPE_CHECKING:
    from types import ModuleType
    from typing import Any

    from pytest_benchmark.fixture import BenchmarkFixture

    from glass._types import AngularPowerSpectra, UnifiedGenerator
    from tests.fixtures.helper_classes import GeneratorConsumer


@pytest.mark.stable
def test_iternorm_no_size(
    benchmark: BenchmarkFixture,
    generator_consumer: type[GeneratorConsumer],
    xpb: ModuleType,
) -> None:
    """Regression tests for glass.iternorm with k=2."""
    array_in = [xpb.asarray([1.0, 0.5, 0.1])] * 10_000

    def function_to_benchmark() -> list[Any]:
        generator = glass.iternorm(array_in)
        return generator_consumer.consume(generator)

    result = benchmark(function_to_benchmark)

    assert len(result) == len(array_in)


@pytest.mark.stable
@pytest.mark.parametrize("num_dimensions", [1, 2])
def test_iternorm_specify_size(
    benchmark: BenchmarkFixture,
    generator_consumer: type[GeneratorConsumer],
    xpb: ModuleType,
    num_dimensions: int,
) -> None:
    """Regression tests for glass.iternorm with k=2 and size=(3,)."""
    if num_dimensions == 1:
        array_in = [xpb.asarray([1.0, 0.5, 0.1])] * 10_000
    elif num_dimensions == 2:
        array_in = [xpb.asarray([[1.0, 0.5, 0.1]] * 3)] * 10_000

    def function_to_benchmark() -> list[Any]:
        generator = glass.iternorm(array_in)
        return generator_consumer.consume(generator)

    result = benchmark(function_to_benchmark)

    assert len(result) == len(array_in)


@pytest.mark.stable
def test_iternorm_k_0(
    benchmark: BenchmarkFixture,
    generator_consumer: type[GeneratorConsumer],
    xpb: ModuleType,
) -> None:
    """Regression tests glass.iternorm with k set to 0."""
    array_in = [xpb.asarray([x]) for x in xpb.ones(1_000, dtype=xpb.float64)]

    def function_to_benchmark() -> list[Any]:
        generator = glass.iternorm(array_in)
        return generator_consumer.consume(generator)

    result = benchmark(function_to_benchmark)

    assert len(result) == len(array_in)


@pytest.mark.stable
def test_cls2cov(
    benchmark: BenchmarkFixture,
    generator_consumer: type[GeneratorConsumer],
    xpb: ModuleType,
) -> None:
    """Regression tests for glass.cls2cov."""
    nl, nf, nc = 3, 2, 2
    array_in = [xpb.arange(i + 1.0, i + 4.0) for i in range(1_000)]

    def function_to_benchmark() -> list[Any]:
        generator = glass.cls2cov(
            array_in,
            nl,
            nf,
            nc,
        )
        return generator_consumer.consume(generator)

    covs = benchmark(function_to_benchmark)
    cov = covs[0]

    assert cov.shape == (nl, nc + 1)
    assert cov.dtype == xpb.float64

    xpx.testing.assert_equal(cov[:, 0], xpb.asarray([1.0, 1.5, 2.0]))
    xpx.testing.assert_equal(cov[:, 1], xpb.asarray([1.5, 2.0, 2.5]))
    xpx.testing.assert_equal(cov[:, 2], xpb.asarray(0.0), check_shape=False)


@pytest.mark.unstable
@pytest.mark.parametrize("use_rng", [False, True])
@pytest.mark.parametrize("ncorr", [None, 1])
def test_generate_grf(  # noqa: PLR0913
    benchmark: BenchmarkFixture,
    generator_consumer: type[GeneratorConsumer],
    ncorr: int | None,
    urngb: UnifiedGenerator,
    use_rng: bool,  # noqa: FBT001
    xpb: ModuleType,
) -> None:
    """Regression tests of glass.fields._generate_grf with positional arguments only."""
    n = 10
    gls: AngularPowerSpectra = [
        xpb.ones(n) if i == j else xpb.zeros(n)
        for i in range(n)
        for j in range(i, -1, -1)
    ]
    nside = 32

    def function_to_benchmark() -> list[Any]:
        generator = glass.fields._generate_grf(
            gls,
            nside,
            rng=urngb if use_rng else None,
            ncorr=ncorr,
        )
        return generator_consumer.consume(generator)

    gaussian_fields = benchmark(function_to_benchmark)

    assert gaussian_fields[0].shape == (hp.nside2npix(nside),)


@pytest.mark.unstable
@pytest.mark.parametrize("ncorr", [None, 1])
def test_generate(
    benchmark: BenchmarkFixture,
    generator_consumer: type[GeneratorConsumer],
    xpb: ModuleType,
    ncorr: int | None,
) -> None:
    """Regression tests for glass.generate."""
    n = 10
    fields = [lambda x, var: x for _ in range(n)]  # noqa: ARG005
    fields[1] = lambda x, var: x**2  # noqa: ARG005
    gls: AngularPowerSpectra = [
        xpb.ones(n) if i == j else xpb.zeros(n)
        for i in range(n)
        for j in range(i, -1, -1)
    ]
    nside = 32

    def function_to_benchmark() -> list[Any]:
        generator = glass.generate(
            fields,
            gls,
            nside=nside,
            ncorr=ncorr,
        )
        return generator_consumer.consume(generator)

    result = benchmark(function_to_benchmark)

    for field in result:
        assert field.shape == (hp.nside2npix(nside),)


@pytest.mark.unstable
def test_getcl_lmax_0(
    benchmark: BenchmarkFixture,
    xpb: ModuleType,
) -> None:
    """Regression tests for glass.getcl with lmax of 0."""
    scale_factor = 1_000
    # make a mock Cls array with the index pairs as entries
    cls: AngularPowerSpectra = [
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
    assert result.shape[0] == 1
    xpx.testing.assert_equal(result, expected)


@pytest.mark.unstable
def test_getcl_lmax_larger_than_cls(
    benchmark: BenchmarkFixture,
    xpb: ModuleType,
) -> None:
    """Regression tests for glass.getcl with lmax larger than the length of cl."""
    scale_factor = 1_000
    # make a mock Cls array with the index pairs as entries
    cls: AngularPowerSpectra = [
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
    assert result.shape[0] == lmax + 1
    xpx.testing.assert_equal(result[2:], expected)
