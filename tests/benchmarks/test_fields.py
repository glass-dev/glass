from __future__ import annotations

from random import randrange
from typing import TYPE_CHECKING

import healpy as hp
import numpy as np
import pytest

import glass.fields

if TYPE_CHECKING:
    from types import ModuleType

    from pytest_benchmark.fixture import BenchmarkFixture
    from pytest_mock import MockerFixture

    from glass._types import UnifiedGenerator


def _nth_triangular_number(n: int) -> int:
    """Return the nth triangular number."""
    return int(n * (n + 1) * 0.5)


def test_discretized_cls_lmax_provided(
    xp: ModuleType,
    benchmark: BenchmarkFixture,
    benchmark_scale_factor: int,
) -> None:
    """Benchmarks for glass.fields.discretized_cls with lmax provided."""
    if xp.__name__ == "array_api_strict":
        pytest.skip("glass.fields.discretized_cls has not been ported to the array-api")
    # power spectra truncated at lmax + 1 if lmax provided
    cls = [
        xp.arange(10) for _ in range(_nth_triangular_number(benchmark_scale_factor * 3))
    ]
    result = benchmark(glass.fields.discretized_cls, cls, lmax=5)

    for cl in result:
        assert len(cl) == 6


def test_discretized_cls_ncorr_provided(
    xp: ModuleType,
    benchmark: BenchmarkFixture,
    benchmark_scale_factor: int,
) -> None:
    """Benchmarks for glass.fields.discretized_cls with ncorr provided."""
    if xp.__name__ == "array_api_strict":
        pytest.skip("glass.fields.discretized_cls has not been ported to the array-api")
    cls = [
        xp.arange(10) for _ in range(_nth_triangular_number(benchmark_scale_factor * 3))
    ]
    ncorr = 0
    result = benchmark(glass.fields.discretized_cls, cls, ncorr=ncorr)

    assert len(result[0]) == 10
    assert len(result[1]) == 10
    assert len(result[2]) == 0  # third correlation should be removed


def test_discretized_cls_nside_provided(
    xp: ModuleType,
    benchmark: BenchmarkFixture,
    benchmark_scale_factor: int,
) -> None:
    """Benchmarks for glass.fields.discretized_cls with nside provided."""
    if xp.__name__ == "array_api_strict":
        pytest.skip("glass.fields.discretized_cls has not been ported to the array-api")
    # check if pixel window function was applied correctly with nside not None
    cls = [xp.ones(10) for _ in range(_nth_triangular_number(benchmark_scale_factor))]
    cls[0] = []
    nside = 4

    pw = hp.pixwin(nside, lmax=7)

    result = benchmark(glass.fields.discretized_cls, cls, nside=nside)

    for cl in result:
        n = min(len(cl), len(pw))
        expected = xp.ones(n) * pw[:n] ** 2
        np.testing.assert_allclose(cl[:n], expected)


def test_effective_cls(
    xp: ModuleType,
    benchmark: BenchmarkFixture,
    benchmark_scale_factor: int,
) -> None:
    """Benchmark for glass.fields.effective_cls."""
    # Call jax version of iternorm once jax version is written
    if xp.__name__ == "jax.numpy":
        pytest.skip("Arrays in effective_cls are not immutable, so do not support jax")

    # empty cls
    cls = [
        xp.arange(15.0) for _ in range(_nth_triangular_number(benchmark_scale_factor))
    ]
    weights1 = xp.ones((benchmark_scale_factor, 1))

    result = benchmark(glass.fields.effective_cls, cls, weights1)
    assert result.shape == (1, 1, 15)


def test_effective_cls_provided_lmax(
    xp: ModuleType,
    benchmark: BenchmarkFixture,
    benchmark_scale_factor: int,
) -> None:
    """Benchmark for glass.fields.effective_cls with lmax provided."""
    # Call jax version of iternorm once jax version is written
    if xp.__name__ == "jax.numpy":
        pytest.skip("Arrays in effective_cls are not immutable, so do not support jax")

    # check truncation if lmax provided
    cls = [
        xp.arange(15.0) for _ in range(_nth_triangular_number(benchmark_scale_factor))
    ]
    weights1 = xp.ones((benchmark_scale_factor, 1))

    lmax = 20
    result = benchmark(glass.fields.effective_cls, cls, weights1, lmax=lmax)

    assert result.shape == (1, 1, lmax + 1)
    np.testing.assert_allclose(result[..., lmax + 1 :], 0)


def test_effective_cls_weights2_equal_weights1(
    xp: ModuleType,
    benchmark: BenchmarkFixture,
    benchmark_scale_factor: int,
) -> None:
    """Benchmark for glass.fields.effective_cls with weight2 equal to weights1."""
    # Call jax version of iternorm once jax version is written
    if xp.__name__ == "jax.numpy":
        pytest.skip("Arrays in effective_cls are not immutable, so do not support jax")

    # check with weights1 and weights2 and weights1 is weights2
    cls = [
        xp.arange(15.0) for _ in range(_nth_triangular_number(benchmark_scale_factor))
    ]
    weights1 = xp.ones((benchmark_scale_factor, 1))

    result = benchmark(glass.fields.effective_cls, cls, weights1, weights2=weights1)
    assert result.shape == (1, 1, 15)


def test_getcl_no_lmax(
    xp: ModuleType,
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
    result = glass.fields.getcl(cls, random_i, random_j)
    expected = xp.asarray(
        [min(random_i, random_j), max(random_i, random_j)], dtype=xp.float64
    )
    np.testing.assert_allclose(xp.sort(result), expected)


def test_getcl_lmax_0(
    xp: ModuleType,
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
    result = glass.fields.getcl(cls, random_i, random_j, lmax=0)
    expected = xp.asarray([max(random_i, random_j)], dtype=xp.float64)
    assert result.size == 1
    np.testing.assert_allclose(result, expected)


def test_getcl_lmax_50(
    xp: ModuleType,
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
    result = glass.fields.getcl(cls, random_i, random_j, lmax=50)
    expected = xp.zeros((49,), dtype=xp.float64)
    assert result.size == 51
    np.testing.assert_allclose(result[2:], expected)


@pytest.mark.parametrize(
    ("input_index", "expected_output"),
    [
        (1, [[0, 0]]),
        (2, [[0, 0], [1, 1], [1, 0]]),
        (3, [[0, 0], [1, 1], [1, 0], [2, 2], [2, 1], [2, 0]]),
    ],
)
def test_spectra_indices(
    xp: ModuleType,
    input_index: int,
    expected_output: list[int],
    benchmark: BenchmarkFixture,
) -> None:
    """Benchmarks for glass.fields.spectra_indices."""
    np.testing.assert_array_equal(
        benchmark(glass.fields.spectra_indices, input_index),
        xp.asarray(expected_output),
    )


def test_spectra_indices_input_of_zero(
    xp: ModuleType,
    benchmark: BenchmarkFixture,
) -> None:
    """Benchmarks for glass.fields.spectra_indices with input of zero."""
    np.testing.assert_array_equal(
        benchmark(glass.fields.spectra_indices, 0), xp.zeros((0, 2))
    )


def test_gaussian_fields(
    xp: ModuleType,
    benchmark: BenchmarkFixture,
) -> None:
    """Benchmarks for glass.fields.gaussian_fields."""
    shells = [
        glass.RadialWindow(xp.asarray([]), xp.asarray([]), 1.0),
        glass.RadialWindow(xp.asarray([]), xp.asarray([]), 2.0),
    ]
    fields = benchmark(glass.fields.gaussian_fields, shells)
    assert len(fields) == len(shells)
    assert all(isinstance(f, glass.grf.Normal) for f in fields)


def test_lognormal_fields(
    xp: ModuleType,
    benchmark: BenchmarkFixture,
) -> None:
    """Benchmarks for glass.fields.lognormal_fields."""
    shells = [
        glass.RadialWindow(xp.asarray([]), xp.asarray([]), 1),
        glass.RadialWindow(xp.asarray([]), xp.asarray([]), 2),
        glass.RadialWindow(xp.asarray([]), xp.asarray([]), 3),
    ]

    fields = benchmark(glass.fields.lognormal_fields, shells)
    assert len(fields) == len(shells)
    assert all(isinstance(f, glass.grf.Lognormal) for f in fields)
    assert [f.lamda for f in fields] == [1.0, 1.0, 1.0]


def test_compute_gaussian_spectra(
    xp: ModuleType,
    mocker: MockerFixture,
    benchmark: BenchmarkFixture,
) -> None:
    """Benchmarks for glass.fields.compute_gaussian_spectra."""
    mock = mocker.patch("glass.grf.compute")

    fields = [glass.grf.Normal() for _ in range(10)]
    spectra = [xp.zeros(10) for _ in range(55)]

    gls = benchmark(
        glass.fields.compute_gaussian_spectra,
        fields,
        spectra,
    )

    assert mock.call_args_list[0] == mocker.call(spectra[0], fields[0], fields[0])
    assert mock.call_args_list[1] == mocker.call(spectra[1], fields[1], fields[1])
    assert mock.call_args_list[2] == mocker.call(spectra[2], fields[1], fields[0])
    assert gls == [mock.return_value for _ in range(55)]


def test_solve_gaussian_spectra(
    xp: ModuleType,
    mocker: MockerFixture,
    benchmark: BenchmarkFixture,
) -> None:
    """Benchmarks for glass.fields.solve_gaussian_spectra."""
    mock = mocker.patch("glass.grf.solve")

    result = mock.return_value

    mock.return_value = (result, None, 3)

    fields = [glass.grf.Normal() for _ in range(10)]
    spectra = [xp.zeros(5 * (i + 1)) for i in range(55)]

    gls = benchmark(
        glass.fields.solve_gaussian_spectra,
        fields,
        spectra,
    )

    assert mock.call_args_list[0] == mocker.call(
        spectra[0],
        fields[0],
        fields[0],
        pad=10,
        monopole=0.0,
    )
    assert mock.call_args_list[1] == mocker.call(
        spectra[1],
        fields[1],
        fields[1],
        pad=20,
        monopole=0.0,
    )
    assert mock.call_args_list[2] == mocker.call(
        spectra[2],
        fields[1],
        fields[0],
        pad=30,
        monopole=0.0,
    )
    assert gls == [result for _ in range(len(spectra))]


def test_glass_to_healpix_spectra(
    xp: ModuleType,
    benchmark: BenchmarkFixture,
) -> None:
    """Benchmarks for glass.fields.glass_to_healpix_spectra."""
    if xp.__name__ in {"array_api_strict", "jax.numpy"}:
        pytest.skip(
            "glass.fields.glass_to_healpix_spectra has not been ported to the array-api"
        )
    n = 100
    # inp structure [11, 22, 21, 33, 32, 31, 44, 43, 42, 41,...]
    inp = [(10 * i) + i - j for i in range(n) for j in range(i)]
    out = benchmark(glass.fields.glass_to_healpix_spectra, inp)
    # out structure [11, 22, 33, 44, 21, 32, 43, 31, 42, 41,...]
    expected_out = [(10 * i) + (i - d) for d in range(n - 1) for i in range(d + 1, n)]
    np.testing.assert_array_equal(out, expected_out)


def test_healpix_to_glass_spectra(
    xp: ModuleType,
    benchmark: BenchmarkFixture,
) -> None:
    """Benchmarks for glass.fields.healpix_to_glass_spectra."""
    if xp.__name__ in {"array_api_strict", "jax.numpy"}:
        pytest.skip(
            "glass.fields.healpix_to_glass_spectra has not been ported to the array-api"
        )
    n = 100
    # inp structure [11, 22, 33, 44, 21, 32, 43, 31, 42, 41,...]
    inp = [(10 * i) + (i - d) for d in range(n - 1) for i in range(d + 1, n)]
    out = benchmark(glass.fields.healpix_to_glass_spectra, inp)
    # out structure [11, 22, 21, 33, 32, 31, 44, 43, 42, 41,...]
    expected_out = [(10 * i) + i - j for i in range(n) for j in range(i)]
    np.testing.assert_array_equal(out, expected_out)


def test_glass_to_healpix_alm(
    xp: ModuleType,
    benchmark: BenchmarkFixture,
) -> None:
    """Benchmarks for glass.fields._glass_to_healpix_alm."""
    if xp.__name__ in {"array_api_strict"}:
        pytest.skip(
            "glass.fields._glass_to_healpix_alm has not been ported to the array-api"
        )
    n = 100
    # inp structure [00, 10, 11, 20, 21, 22, 30, 31, 32, 33,...]
    inp = xp.asarray([(10 * i) + (j) for i in range(n) for j in range(i + 1)])
    out = benchmark(glass.fields._glass_to_healpix_alm, inp)
    # out structure [00, 10, 20, 30, 11, 21, 31, 22, 32, 33,...]
    expected_out = xp.asarray(
        [(10 * (j)) + (10 * i) + i for i in range(n) for j in range(n - i)]
    )
    np.testing.assert_array_equal(
        out,
        expected_out,
    )


@pytest.mark.parametrize(
    ("lmax", "expected_output"),
    [
        (
            None,
            [
                [
                    [110, 210, 310],
                    [210, 220, 320],
                    [310, 320, 330],
                ],
                [
                    [111, 211, 311],
                    [211, 221, 321],
                    [311, 321, 331],
                ],
                [
                    [112, 212, 312],
                    [212, 222, 322],
                    [312, 322, 332],
                ],
                [
                    [113, 213, 313],
                    [213, 223, 323],
                    [313, 323, 333],
                ],
            ],
        ),
        (
            1,
            [
                [
                    [110, 210, 310],
                    [210, 220, 320],
                    [310, 320, 330],
                ],
                [
                    [111, 211, 311],
                    [211, 221, 321],
                    [311, 321, 331],
                ],
            ],
        ),
        (
            4,
            [
                [
                    [110, 210, 310],
                    [210, 220, 320],
                    [310, 320, 330],
                ],
                [
                    [111, 211, 311],
                    [211, 221, 321],
                    [311, 321, 331],
                ],
                [
                    [112, 212, 312],
                    [212, 222, 322],
                    [312, 322, 332],
                ],
                [
                    [113, 213, 313],
                    [213, 223, 323],
                    [313, 323, 333],
                ],
                [
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                ],
            ],
        ),
    ],
)
def test_cov_from_spectra(
    xp: ModuleType,
    benchmark: BenchmarkFixture,
    lmax: int,
    expected_output: list[int],
) -> None:
    """Benchmarks for glass.fields.cov_from_spectra."""
    if xp.__name__ in {"array_api_strict"}:
        pytest.skip(
            "glass.fields.cov_from_spectra has not been ported to the array-api"
        )
    spectra = xp.asarray(
        [
            [110, 111, 112, 113],
            [220, 221, 222, 223],
            [210, 211, 212, 213],
            [330, 331, 332, 333],
            [320, 321, 322, 323],
            [310, 311, 312, 313],
        ],
    )

    np.testing.assert_array_equal(
        benchmark(
            glass.fields.cov_from_spectra,
            spectra,
            lmax=lmax,
        ),
        expected_output,
    )


@pytest.mark.parametrize(
    "input_array",
    [
        # posdef spectra
        [
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.9, 0.9, 0.9],
        ],
        # semidef spectra
        [
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.9, 1.0, 0.0],
        ],
    ],
)
def test_check_posdef_spectra_true_result(
    xp: ModuleType,
    benchmark: BenchmarkFixture,
    input_array: list[int],
) -> None:
    """Benchmarks for glass.fields.check_posdef_spectra where the result is True."""
    if xp.__name__ in {"array_api_strict"}:
        pytest.skip(
            "glass.fields.cov_from_spectra has not been ported to the array-api"
        )
    assert benchmark(
        glass.fields.check_posdef_spectra,
        xp.asarray(input_array),
    )


def test_check_posdef_spectra_false_result(
    xp: ModuleType,
    benchmark: BenchmarkFixture,
) -> None:
    """Benchmarks for glass.fields.check_posdef_spectra where the result is False."""
    if xp.__name__ in {"array_api_strict"}:
        pytest.skip(
            "glass.fields.cov_from_spectra has not been ported to the array-api"
        )
    # indef spectra
    assert not benchmark(
        glass.fields.check_posdef_spectra,
        xp.asarray(
            [
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.1, 1.1, 1.1],
            ],
        ),
    )


def test_regularized_spectra(
    xp: ModuleType,
    mocker: MockerFixture,
    urng: UnifiedGenerator,
    benchmark: BenchmarkFixture,
) -> None:
    """Benchmarks for glass.fields.regularized_spectra wiuth the clip method."""
    if xp.__name__ in {"array_api_strict"}:
        pytest.skip(
            "glass.fields.regularized_spectra has not been ported to the array-api"
        )
    spectra = urng.random(size=(6, 101))

    # test method "nearest"
    cov_nearest = mocker.spy(glass.algorithm, "cov_clip")
    benchmark(
        glass.fields.regularized_spectra,
        spectra,
        method="clip",
    )
    cov_nearest.assert_called()


def test_regularized_spectra_nearest(
    xp: ModuleType,
    mocker: MockerFixture,
    urng: UnifiedGenerator,
    benchmark: BenchmarkFixture,
) -> None:
    """Benchmarks for glass.fields.regularized_spectra with the nearest method."""
    if xp.__name__ in {"array_api_strict"}:
        pytest.skip(
            "glass.fields.regularized_spectra has not been ported to the array-api"
        )
    spectra = urng.random(size=(6, 101))

    # test method "nearest"
    cov_nearest = mocker.spy(glass.algorithm, "cov_nearest")
    with pytest.warns(UserWarning, match="Nearest correlation matrix not found"):
        # we don't care about convergence here, only that the correct
        # method is called so this is to suppress the warning
        benchmark(
            glass.fields.regularized_spectra,
            spectra,
            method="nearest",
        )
    cov_nearest.assert_called()
