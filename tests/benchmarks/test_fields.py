from __future__ import annotations

from typing import TYPE_CHECKING

import healpy as hp
import numpy as np
import pytest

import glass.fields

if TYPE_CHECKING:
    from types import ModuleType

    from pytest_benchmark.fixture import BenchmarkFixture

    from glass._types import UnifiedGenerator


def test_iternorm_no_size(xp: ModuleType, benchmark: BenchmarkFixture) -> None:
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
    xp: ModuleType,
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


def test_iternorm_k_0(xp: ModuleType, benchmark: BenchmarkFixture) -> None:
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


def test_cls2cov(xp: ModuleType, benchmark: BenchmarkFixture) -> None:
    """Benchmarks for glass.cls2cov."""
    # Call jax version of iternorm once jax version is written
    if xp.__name__ == "jax.numpy":
        pytest.skip("Arrays in cls2cov are not immutable, so do not support jax")

    # check output values and shape

    nl, nf, nc = 3, 2, 2

    generator = benchmark(
        glass.fields.cls2cov,
        [xp.asarray([1.0, 0.5, 0.3]), None, xp.asarray([0.7, 0.6, 0.1])],
        nl,
        nf,
        nc,
    )
    cov = next(generator)

    assert cov.shape == (nl, nc + 1)
    assert cov.dtype == xp.float64

    np.testing.assert_allclose(cov[:, 0], xp.asarray([0.5, 0.25, 0.15]))
    np.testing.assert_allclose(cov[:, 1], 0)
    np.testing.assert_allclose(cov[:, 2], 0)


def test_cls2cov_multiple_iterations(
    xp: ModuleType, benchmark: BenchmarkFixture
) -> None:
    """Benchmarks for glass.cls2cov with inputs causing multiple iterations."""
    # Call jax version of iternorm once jax version is written
    if xp.__name__ == "jax.numpy":
        pytest.skip("Arrays in cls2cov are not immutable, so do not support jax")

    # test multiple iterations

    nl, nf, nc = 3, 3, 2

    generator = benchmark(
        glass.fields.cls2cov,
        [
            xp.asarray(arr)
            for arr in [
                [1.0, 0.5, 0.3],
                [0.8, 0.4, 0.2],
                [0.7, 0.6, 0.1],
                [0.9, 0.5, 0.3],
                [0.6, 0.3, 0.2],
                [0.8, 0.7, 0.4],
            ]
        ],
        nl,
        nf,
        nc,
    )

    cov1 = xp.asarray(next(generator), copy=True)
    cov2 = xp.asarray(next(generator), copy=True)
    cov3 = next(generator)

    assert cov1.shape == (nl, nc + 1)
    assert cov2.shape == (nl, nc + 1)
    assert cov3.shape == (nl, nc + 1)

    assert cov1.dtype == xp.float64
    assert cov2.dtype == xp.float64
    assert cov3.dtype == xp.float64

    np.testing.assert_raises(AssertionError, np.testing.assert_allclose, cov1, cov2)
    np.testing.assert_raises(AssertionError, np.testing.assert_allclose, cov2, cov3)


@pytest.mark.parametrize(
    ("alm_in", "bl_in", "inplace", "expected_result"),
    [
        (
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [2.0, 0.5, 1.0],
            True,
            [2.0, 1.0, 1.5, 4.0, 5.0, 6.0],
        ),
        (
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [1.0, 1.0, 1.0],
            False,
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ),
        (
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [0.0, 1.0, 0.0],
            False,
            [0.0, 2.0, 3.0, 0.0, 0.0, 0.0],
        ),
        (
            [],
            [],
            False,
            [],
        ),
    ],
)
def test_multalm(
    xp: ModuleType,
    alm_in: list[float],
    bl_in: list[float],
    inplace: bool,  # noqa: FBT001
    expected_result: list[float],
) -> None:
    """Benchmarks for glass.fields._multalm."""
    # Call jax version of iternorm once jax version is written
    if xp.__name__ == "jax.numpy":
        pytest.skip("Arrays in multalm are not immutable, so do not support jax")

    # check output values and shapes

    alm = xp.asarray(alm_in)
    bl = xp.asarray(bl_in)
    alm_copy = xp.asarray(alm, copy=True)

    result = glass.fields._multalm(alm, bl, inplace=inplace)

    np.testing.assert_allclose(result, xp.asarray(expected_result))

    # If inplace alm is updated but the copy is not
    if inplace:
        np.testing.assert_allclose(result, alm)
        np.testing.assert_raises(
            AssertionError,
            np.testing.assert_allclose,
            alm_copy,
            result,
        )


def test_discretized_cls_lmax_provided(xp: ModuleType) -> None:
    """Benchmarks for glass.fields.discretized_cls with lmax provided."""
    if xp.__name__ == "array_api_strict":
        pytest.skip("glass.fields._multalm has not yet been ported to the array-api")
    # power spectra truncated at lmax + 1 if lmax provided
    cls = [xp.arange(10), xp.arange(10), xp.arange(10)]
    result = glass.fields.discretized_cls(cls, lmax=5)

    for cl in result:
        assert len(cl) == 6


def test_discretized_cls_ncorr_provided(
    xp: ModuleType, benchmark: BenchmarkFixture
) -> None:
    """Benchmarks for glass.fields.discretized_cls with ncorr provided."""
    if xp.__name__ == "array_api_strict":
        pytest.skip("glass.fields._multalm has not yet been ported to the array-api")
    cls = [xp.arange(10), xp.arange(10), xp.arange(10)]
    ncorr = 0
    result = benchmark(glass.fields.discretized_cls, cls, ncorr=ncorr)

    assert len(result[0]) == 10
    assert len(result[1]) == 10
    assert len(result[2]) == 0  # third correlation should be removed


def test_discretized_cls_nside_provided(
    xp: ModuleType, benchmark: BenchmarkFixture
) -> None:
    """Benchmarks for glass.fields.discretized_cls with nside provided."""
    if xp.__name__ == "array_api_strict":
        pytest.skip("glass.fields._multalm has not yet been ported to the array-api")
    # check if pixel window function was applied correctly with nside not None
    cls = [[], xp.ones(10), xp.ones(10)]
    nside = 4

    pw = hp.pixwin(nside, lmax=7)

    result = benchmark(glass.fields.discretized_cls, cls, nside=nside)

    for cl in result:
        n = min(len(cl), len(pw))
        expected = xp.ones(n) * pw[:n] ** 2
        np.testing.assert_allclose(cl[:n], expected)


@pytest.mark.parametrize(
    ("cls_size", "weights1_in", "expected_shape"),
    [
        (
            0,
            [],
            (0,),
        ),
        (
            3,
            [[1.0], [1.0]],
            (1, 1, 15),
        ),
    ],
)
def test_effective_cls(
    xp: ModuleType,
    cls_size: int,
    weights1_in: list[int],
    expected_shape: tuple[int, ...],
    benchmark: BenchmarkFixture,
) -> None:
    """Benchmark for glass.fields.effective_cls."""
    # Call jax version of iternorm once jax version is written
    if xp.__name__ == "jax.numpy":
        pytest.skip("Arrays in effective_cls are not immutable, so do not support jax")

    # empty cls
    cls = [xp.arange(15.0) for _ in range(cls_size)]
    weights1 = xp.asarray(weights1_in)

    result = benchmark(glass.fields.effective_cls, cls, weights1)
    assert result.shape == expected_shape


def test_effective_cls_provided_lmax(
    xp: ModuleType, benchmark: BenchmarkFixture
) -> None:
    """Benchmark for glass.fields.effective_cls with lmax provided."""
    # Call jax version of iternorm once jax version is written
    if xp.__name__ == "jax.numpy":
        pytest.skip("Arrays in effective_cls are not immutable, so do not support jax")

    # check truncation if lmax provided
    cls = [xp.arange(15.0) for _ in range(3)]
    weights1 = xp.ones((2, 1))

    result = benchmark(glass.fields.effective_cls, cls, weights1, lmax=5)

    assert result.shape == (1, 1, 6)
    np.testing.assert_allclose(result[..., 6:], 0)


def test_effective_cls_weights2_equal_weights1(
    xp: ModuleType, benchmark: BenchmarkFixture
) -> None:
    """Benchmark for glass.fields.effective_cls with weight2 equal to weights1."""
    # Call jax version of iternorm once jax version is written
    if xp.__name__ == "jax.numpy":
        pytest.skip("Arrays in effective_cls are not immutable, so do not support jax")

    # check with weights1 and weights2 and weights1 is weights2
    cls = [xp.arange(15.0) for _ in range(3)]
    weights1 = xp.ones((2, 1))

    result = benchmark(glass.fields.effective_cls, cls, weights1, weights2=weights1)
    assert result.shape == (1, 1, 15)


def test_generate_grf_positional_args_only(xp: ModuleType) -> None:
    """Benchmarks for glass.fields._generate_grf with positional arguments only."""
    if xp.__name__ == "array_api_strict":
        pytest.skip("glass.fields._multalm has not yet been ported to the array-api")
    if xp.__name__ == "jax.numpy":
        pytest.skip("Arrays in effective_cls are not immutable, so do not support jax")

    gls = [xp.asarray([1.0, 0.5, 0.1])]
    nside = 4

    gaussian_fields = list(glass.fields._generate_grf(gls, nside))

    assert gaussian_fields[0].shape == (hp.nside2npix(nside),)


def test_generate_grf_with_rng(
    xp: ModuleType,
    urng: UnifiedGenerator,
    benchmark: BenchmarkFixture,
) -> None:
    """Benchmarks for glass.fields._generate_grf with positional arguments only."""
    if xp.__name__ in "array_api_strict":
        pytest.skip("glass.fields._multalm has not yet been ported to the array-api")
    if xp.__name__ == "jax.numpy":
        pytest.skip("Arrays in effective_cls are not immutable, so do not support jax")

    gls = [xp.asarray([1.0, 0.5, 0.1])]
    nside = 4

    # requires resetting the RNG for reproducibility
    gaussian_fields = list(
        benchmark(
            glass.fields._generate_grf,
            gls,
            nside,
            rng=urng,
        )
    )

    assert gaussian_fields[0].shape == (hp.nside2npix(nside),)


def test_generate_grf_with_ncorr_and_rng(
    xp: ModuleType,
    urng: UnifiedGenerator,
    benchmark: BenchmarkFixture,
) -> None:
    """Benchmarks for glass.fields._generate_grf with positional arguments only."""
    if xp.__name__ == "array_api_strict":
        pytest.skip("glass.fields._multalm has not yet been ported to the array-api")
    if xp.__name__ == "jax.numpy":
        pytest.skip("Arrays in effective_cls are not immutable, so do not support jax")

    gls = [xp.asarray([1.0, 0.5, 0.1])]
    nside = 4
    ncorr = 1

    # requires resetting the RNG for reproducibility
    new_gaussian_fields = list(
        benchmark(
            glass.fields._generate_grf,
            gls,
            nside,
            ncorr=ncorr,
            rng=urng,
        ),
    )

    assert new_gaussian_fields[0].shape == (hp.nside2npix(nside),)
