from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

import numpy as np
import pytest

import glass
import glass.fields
import glass.healpix as hp
from glass import _rng

if TYPE_CHECKING:
    from types import ModuleType

    from pytest_mock import MockerFixture

    from glass._types import AngularPowerSpectra
    from tests.fixtures.helper_classes import Compare

HAVE_JAX = importlib.util.find_spec("jax") is not None


@pytest.fixture(scope="session")
def not_triangle_numbers() -> list[int]:
    return [2, 4, 5, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 20]


def test_iternorm(xp: ModuleType) -> None:
    # Call jax version of iternorm once jax version is written
    if xp.__name__ == "jax.numpy":
        pytest.skip("Arrays in iternorm are not immutable, so do not support jax")

    # check output shapes and types

    k = 2

    generator = glass.iternorm(
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

    # specify size

    size = 3

    generator = glass.iternorm(
        k,
        (
            xp.asarray(arr)
            for arr in [
                [1.0, 0.5, 0.5],
                [0.5, 0.2, 0.1],
                [0.5, 0.1, 0.2],
            ]
        ),
        size,
    )
    result = next(generator)

    j, a, s = result

    assert isinstance(j, int)
    assert a.shape == (size, k)
    assert s.shape == (size,)

    # test shape mismatch error

    with pytest.raises(TypeError, match="covariance row 0: shape"):
        list(
            glass.iternorm(
                k,
                (
                    xp.asarray(arr)
                    for arr in [
                        [1.0, 0.5],
                        [0.5, 0.2],
                    ]
                ),
            ),
        )

    # test positive definite error

    with pytest.raises(ValueError, match="covariance matrix is not positive definite"):
        list(
            glass.iternorm(
                k,
                (xp.asarray(x) for x in [1.0, 0.5, 0.9, 0.5, 0.2, 0.4, 0.9, 0.4, -1.0]),
            ),
        )

    # test multiple iterations

    size = (3,)

    generator = glass.iternorm(
        k,
        (
            xp.asarray(arr)
            for arr in [
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
            ]
        ),
        size,
    )

    result1 = next(generator)
    result2 = next(generator)

    assert result1 != result2
    assert isinstance(result1, tuple)
    assert len(result1) == 3
    assert isinstance(result2, tuple)
    assert len(result2) == 3

    # test k = 0

    generator = glass.iternorm(0, xp.asarray([1.0]))

    j, a, s = result

    assert j == 1
    assert a.shape == (3, 2)
    assert s.shape == (3,)


@pytest.mark.skipif(not HAVE_JAX, reason="test requires jax")
def test_cls2cov_jax(compare: type[Compare], jnp: ModuleType) -> None:
    nl, nf, nc = 3, 3, 2

    generator = glass.cls2cov(
        [
            jnp.asarray(arr)
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

    cov1 = jnp.asarray(next(generator), copy=False)
    cov2 = jnp.asarray(next(generator), copy=False)
    cov3 = next(generator)

    assert cov1.shape == (nl, nc + 1)
    assert cov2.shape == (nl, nc + 1)
    assert cov3.shape == (nl, nc + 1)

    assert cov1.dtype == jnp.float64
    assert cov2.dtype == jnp.float64
    assert cov3.dtype == jnp.float64

    # cov1 has the expected value for the first iteration (different to cov1_copy)
    compare.assert_allclose(cov1[:, 0], jnp.asarray([0.5, 0.25, 0.15]))

    # The copies should not be equal
    with pytest.raises(AssertionError, match="Not equal to tolerance"):
        compare.assert_allclose(cov1, cov2)

    with pytest.raises(AssertionError, match="Not equal to tolerance"):
        compare.assert_allclose(cov2, cov3)


def test_cls2cov_no_jax(compare: type[Compare], xpb: ModuleType) -> None:
    # check output values and shape

    nl, nf, nc = 3, 2, 2

    generator = glass.cls2cov(
        [xpb.asarray([1.0, 0.5, 0.3]), None, xpb.asarray([0.7, 0.6, 0.1])],
        nl,
        nf,
        nc,
    )
    cov = next(generator)

    assert cov.shape == (nl, nc + 1)
    assert cov.dtype == xpb.float64

    compare.assert_allclose(cov[:, 0], xpb.asarray([0.5, 0.25, 0.15]))
    compare.assert_allclose(cov[:, 1], 0)
    compare.assert_allclose(cov[:, 2], 0)

    # test negative value error

    generator = glass.cls2cov(
        [
            xpb.asarray(arr)
            for arr in [
                [-1.0, 0.5, 0.3],
                [0.8, 0.4, 0.2],
                [0.7, 0.6, 0.1],
            ]
        ],
        nl,
        nf,
        nc,
    )
    with pytest.raises(ValueError, match="negative values in cl"):
        next(generator)

    # test multiple iterations

    nl, nf, nc = 3, 3, 2

    generator = glass.cls2cov(
        [
            xpb.asarray(arr)
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

    cov1 = xpb.asarray(next(generator), copy=False)
    cov1_copy = xpb.asarray(cov1, copy=True)
    cov2 = xpb.asarray(next(generator), copy=False)
    cov2_copy = xpb.asarray(cov2, copy=True)
    cov3 = next(generator)

    assert cov1.shape == (nl, nc + 1)
    assert cov2.shape == (nl, nc + 1)
    assert cov3.shape == (nl, nc + 1)

    assert cov1.dtype == xpb.float64
    assert cov2.dtype == xpb.float64
    assert cov3.dtype == xpb.float64

    # cov1|2|3 reuse the same data, so should all equal the third result
    compare.assert_allclose(cov1[:, 0], xpb.asarray([0.45, 0.25, 0.15]))
    compare.assert_allclose(cov1, cov2)
    compare.assert_allclose(cov2, cov3)

    # cov1 has the expected value for the first iteration (different to cov1_copy)
    compare.assert_allclose(cov1_copy[:, 0], xpb.asarray([0.5, 0.25, 0.15]))

    # The copies should not be equal
    with pytest.raises(AssertionError, match="Not equal to tolerance"):
        compare.assert_allclose(cov1_copy, cov2_copy)

    with pytest.raises(AssertionError, match="Not equal to tolerance"):
        compare.assert_allclose(cov2_copy, cov3)


def test_lognormal_gls() -> None:
    shift = 2

    # empty cls

    assert glass.lognormal_gls([], shift) == []

    # check output shape

    assert len(glass.lognormal_gls([np.linspace(1, 5, 5)], shift)) == 1
    assert len(glass.lognormal_gls([np.linspace(1, 5, 5)], shift)[0]) == 5

    inp = [np.linspace(1, 6, 5), np.linspace(1, 5, 4), np.linspace(1, 4, 3)]
    out = glass.lognormal_gls(inp, shift)

    assert len(out) == 3
    assert len(out[0]) == 5
    assert len(out[1]) == 4
    assert len(out[2]) == 3


def test_discretized_cls(compare: type[Compare]) -> None:
    # empty cls

    result = glass.discretized_cls(np.asarray([]))
    assert result == []

    # power spectra truncated at lmax + 1 if lmax provided

    result = glass.discretized_cls(
        [np.arange(10), np.arange(10), np.arange(10)],
        lmax=5,
    )

    for cl in result:
        assert len(cl) == 6

    # check ValueError for triangle number

    with pytest.raises(ValueError, match="invalid number of spectra:"):
        glass.discretized_cls([np.arange(10), np.arange(10)], ncorr=1)

    # ncorr not None

    cls: AngularPowerSpectra = [np.arange(10), np.arange(10), np.arange(10)]
    ncorr = 0
    result = glass.discretized_cls(cls, ncorr=ncorr)

    assert len(result[0]) == 10
    assert len(result[1]) == 10
    assert len(result[2]) == 0  # third correlation should be removed

    # check if pixel window function was applied correctly with nside not None

    nside = 4

    pw = hp.pixwin(nside, lmax=7, xp=np)

    result = glass.discretized_cls(
        [np.asarray([]), np.ones(10), np.ones(10)],
        nside=nside,
    )

    for cl in result:
        n = min(len(cl), len(pw))
        expected = np.ones(n) * pw[:n] ** 2  # type: ignore[operator]
        compare.assert_allclose(cl[:n], expected)


def test_effective_cls(compare: type[Compare], xp: ModuleType) -> None:
    # Call jax version of iternorm once jax version is written
    if xp.__name__ == "jax.numpy":
        pytest.skip("Arrays in effective_cls are not immutable, so do not support jax")

    # empty cls

    result = glass.effective_cls([], xp.asarray([]))
    assert result.shape == (0,)

    # check ValueError for triangle number

    with pytest.raises(ValueError, match="invalid number of spectra:"):
        glass.effective_cls([xp.arange(10), xp.arange(10)], xp.ones((2, 1)))

    # check ValueError for triangle number

    with pytest.raises(ValueError, match="shape mismatch between fields and weights1"):
        glass.effective_cls([], xp.ones((3, 1)))

    # check with only weights1

    cls: AngularPowerSpectra = [xp.arange(15.0) for _ in range(3)]
    weights1 = xp.ones((2, 1))

    result = glass.effective_cls(cls, weights1)
    assert result.shape == (1, 1, 15)

    # check truncation if lmax provided

    result = glass.effective_cls(cls, weights1, lmax=5)

    assert result.shape == (1, 1, 6)
    compare.assert_allclose(result[..., 6:], 0)

    # check with weights1 and weights2 and weights1 is weights2

    result = glass.effective_cls(cls, weights1, weights2=weights1)
    assert result.shape == (1, 1, 15)


def test_generate_grf(compare: type[Compare], xp: ModuleType) -> None:
    gls: AngularPowerSpectra = [xp.asarray([1.0, 0.5, 0.1])]
    nside = 4
    ncorr = 1

    gaussian_fields = list(glass.fields._generate_grf(gls, nside))

    assert gaussian_fields[0].shape == (hp.nside2npix(nside),)

    # requires resetting the RNG for reproducibility
    rng = _rng.rng_dispatcher(xp=xp)
    gaussian_fields = list(glass.fields._generate_grf(gls, nside, rng=rng))

    assert gaussian_fields[0].shape == (hp.nside2npix(nside),)

    # requires resetting the RNG for reproducibility
    rng = _rng.rng_dispatcher(xp=xp)
    new_gaussian_fields = list(
        glass.fields._generate_grf(gls, nside, ncorr=ncorr, rng=rng),
    )

    assert new_gaussian_fields[0].shape == (hp.nside2npix(nside),)

    compare.assert_allclose(new_gaussian_fields[0], gaussian_fields[0])

    with pytest.raises(
        TypeError,
        match="array_namespace requires at least one non-scalar array input",
    ):
        list(glass.fields._generate_grf([], nside))


def test_generate_gaussian(xp: ModuleType) -> None:
    with pytest.deprecated_call():
        glass.generate_gaussian([xp.asarray([1.0, 0.5, 0.1])], 4)


def test_generate_lognormal(xp: ModuleType) -> None:
    with pytest.deprecated_call():
        glass.generate_lognormal([xp.asarray([1.0, 0.5, 0.1])], 4)


def test_generate(compare: type[Compare]) -> None:
    # shape mismatch error

    fields = [lambda x, var: x, lambda x, var: x]  # noqa: ARG005

    with pytest.raises(ValueError, match="mismatch between number of fields and gls"):
        list(glass.generate(fields, [np.ones(10), np.ones(10)], nside=16))

    # check output shape

    nside = 16
    npix = hp.nside2npix(nside)
    gls: AngularPowerSpectra = [np.ones(10), np.ones(10), np.ones(10)]

    result = list(glass.generate(fields, gls, nside=nside))

    assert len(result) == 2
    for field in result:
        assert field.shape == (npix,)

    # check ncorr behavior

    result_default = list(glass.generate(fields, gls, nside=nside, ncorr=None))
    assert len(result_default) == 2

    # ncorr = 1 (forcing only one previous correlation)
    result_limited = list(glass.generate(fields, gls, nside=nside, ncorr=1))
    assert len(result_limited) == 2

    # non-identity field

    fields = [lambda x, var: x, lambda x, var: x**2]  # noqa: ARG005

    result = list(glass.generate(fields, gls, nside=nside))

    compare.assert_allclose(result[1], result[0] ** 2, atol=1e-05)


def test_getcl(compare: type[Compare], xp: ModuleType) -> None:
    # make a mock Cls array with the index pairs as entries
    cls: AngularPowerSpectra = [
        xp.asarray([i, j], dtype=xp.float64)
        for i in range(10)
        for j in range(i, -1, -1)
    ]
    # make sure indices are retrieved correctly
    for i in range(10):
        for j in range(10):
            result = glass.getcl(cls, i, j)
            expected = xp.asarray([min(i, j), max(i, j)], dtype=xp.float64)
            compare.assert_allclose(xp.sort(result), expected)

            # check slicing
            result = glass.getcl(cls, i, j, lmax=0)
            expected = xp.asarray([max(i, j)], dtype=xp.float64)
            assert result.size == 1
            compare.assert_allclose(result, expected)

            # check padding
            result = glass.getcl(cls, i, j, lmax=50)
            expected = xp.zeros((49,), dtype=xp.float64)
            assert result.size == 51
            compare.assert_allclose(result[2:], expected)


def test_is_inv_triangle_number(not_triangle_numbers: list[int]) -> None:
    for n in range(10_000):
        assert glass.fields._inv_triangle_number(n * (n + 1) // 2) == n

    for t in not_triangle_numbers:
        with pytest.raises(ValueError, match=f"not a triangle number: {t}"):
            glass.fields._inv_triangle_number(t)


def test_nfields_from_nspectra(not_triangle_numbers: list[int]) -> None:
    for n in range(10_000):
        assert glass.nfields_from_nspectra(n * (n + 1) // 2) == n

    for t in not_triangle_numbers:
        with pytest.raises(ValueError, match=f"invalid number of spectra: {t}"):
            glass.nfields_from_nspectra(t)


def test_enumerate_spectra() -> None:
    n = 100
    tn = n * (n + 1) // 2

    # create mock spectra with 1 element counting to tn
    spectra: AngularPowerSpectra = np.arange(tn).reshape(tn, 1)

    # this is the expected order of indices
    indices = [(i, j) for i in range(n) for j in range(i, -1, -1)]

    # iterator that will enumerate the spectra for checking
    it = glass.enumerate_spectra(spectra)

    # go through expected indices and values and compare
    for k, (i, j) in enumerate(indices):
        assert next(it) == (i, j, k)

    # make sure iterator is exhausted
    with pytest.raises(StopIteration):
        next(it)


def test_spectra_indices(compare: type[Compare], xp: ModuleType) -> None:
    compare.assert_array_equal(glass.spectra_indices(0), xp.zeros((0, 2)))
    compare.assert_array_equal(glass.spectra_indices(1), xp.asarray([[0, 0]]))
    compare.assert_array_equal(
        glass.spectra_indices(2),
        xp.asarray([[0, 0], [1, 1], [1, 0]]),
    )
    compare.assert_array_equal(
        glass.spectra_indices(3),
        xp.asarray([[0, 0], [1, 1], [1, 0], [2, 2], [2, 1], [2, 0]]),
    )


def test_gaussian_fields(xp: ModuleType) -> None:
    shells = [
        glass.RadialWindow(xp.asarray([]), xp.asarray([]), 1.0),
        glass.RadialWindow(xp.asarray([]), xp.asarray([]), 2.0),
    ]
    fields = glass.gaussian_fields(shells)
    assert len(fields) == len(shells)
    assert all(isinstance(f, glass.grf.Normal) for f in fields)


def test_lognormal_fields(xp: ModuleType) -> None:
    shells = [
        glass.RadialWindow(xp.asarray([]), xp.asarray([]), 1),
        glass.RadialWindow(xp.asarray([]), xp.asarray([]), 2),
        glass.RadialWindow(xp.asarray([]), xp.asarray([]), 3),
    ]

    fields = glass.lognormal_fields(shells)
    assert len(fields) == len(shells)
    assert all(isinstance(f, glass.grf.Lognormal) for f in fields)
    assert [f.lamda for f in fields] == [1.0, 1.0, 1.0]

    fields = glass.lognormal_fields(shells, lambda z: z**2)
    assert [f.lamda for f in fields] == [1, 4, 9]


def test_compute_gaussian_spectra(mocker: MockerFixture, xp: ModuleType) -> None:
    mock = mocker.patch("glass.grf.compute")

    fields = [glass.grf.Normal(), glass.grf.Normal()]
    spectra: AngularPowerSpectra = [xp.zeros(10), xp.zeros(10), xp.zeros(10)]

    gls = glass.compute_gaussian_spectra(fields, spectra)

    assert mock.call_count == 3
    assert mock.call_args_list[0] == mocker.call(spectra[0], fields[0], fields[0])
    assert mock.call_args_list[1] == mocker.call(spectra[1], fields[1], fields[1])
    assert mock.call_args_list[2] == mocker.call(spectra[2], fields[1], fields[0])
    assert gls == [mock.return_value, mock.return_value, mock.return_value]

    # spectra size mismatch
    with pytest.raises(ValueError, match="fields and spectra"):
        glass.compute_gaussian_spectra(fields, spectra[:2])


def test_compute_gaussian_spectra_gh639(mocker: MockerFixture, xp: ModuleType) -> None:
    """Test compute_gaussian_spectra() with an empty input."""
    mock = mocker.patch("glass.grf.compute")

    fields = [glass.grf.Normal(), glass.grf.Normal()]
    spectra: AngularPowerSpectra = [xp.zeros(10), xp.zeros(10), xp.zeros(0)]

    gls = glass.compute_gaussian_spectra(fields, spectra)

    assert mock.call_count == 2
    assert mock.call_args_list[0] == mocker.call(spectra[0], fields[0], fields[0])
    assert mock.call_args_list[1] == mocker.call(spectra[1], fields[1], fields[1])
    assert gls[:2] == [mock.return_value, mock.return_value]
    assert gls[2].size == 0


def test_solve_gaussian_spectra(mocker: MockerFixture, xp: ModuleType) -> None:
    mock = mocker.patch("glass.grf.solve")

    result = mock.return_value

    mock.return_value = (result, None, 3)

    fields = [glass.grf.Normal(), glass.grf.Normal()]
    spectra: AngularPowerSpectra = [xp.zeros(5), xp.zeros(10), xp.zeros(15)]

    gls = glass.solve_gaussian_spectra(fields, spectra)

    assert mock.call_count == 3
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
    assert gls == [result, result, result]

    # spectra size mismatch
    with pytest.raises(ValueError, match="fields and spectra"):
        glass.solve_gaussian_spectra(fields, spectra[:2])


def test_glass_to_healpix_spectra(compare: type[Compare]) -> None:
    inp = [11, 22, 21, 33, 32, 31, 44, 43, 42, 41]
    out = glass.glass_to_healpix_spectra(inp)
    compare.assert_array_equal(out, [11, 22, 33, 44, 21, 32, 43, 31, 42, 41])


def test_healpix_to_glass_spectra(compare: type[Compare]) -> None:
    inp = [11, 22, 33, 44, 21, 32, 43, 31, 42, 41]
    out = glass.healpix_to_glass_spectra(inp)
    compare.assert_array_equal(out, [11, 22, 21, 33, 32, 31, 44, 43, 42, 41])


def test_glass_to_healpix_alm(compare: type[Compare], xp: ModuleType) -> None:
    inp = xp.asarray([00, 10, 11, 20, 21, 22, 30, 31, 32, 33])
    out = glass.fields._glass_to_healpix_alm(inp)
    compare.assert_array_equal(
        out,
        xp.asarray([00, 10, 20, 30, 11, 21, 31, 22, 32, 33]),
    )


def test_lognormal_shift_hilbert2011(compare: type[Compare]) -> None:
    zs = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    shifts = [glass.lognormal_shift_hilbert2011(z) for z in zs]

    # computed by hand
    check = [0.0103031, 0.02975, 0.0538781, 0.0792, 0.103203, 0.12435, 0.142078, 0.1568]

    compare.assert_allclose(shifts, check, atol=1e-4, rtol=1e-4)


def test_cov_from_spectra(compare: type[Compare]) -> None:
    spectra: AngularPowerSpectra = np.asarray(
        [
            [110, 111, 112, 113],
            [220, 221, 222, 223],
            [210, 211, 212, 213],
            [330, 331, 332, 333],
            [320, 321, 322, 323],
            [310, 311, 312, 313],
        ],
    )

    compare.assert_array_equal(
        glass.cov_from_spectra(spectra),
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
    )

    compare.assert_array_equal(
        glass.cov_from_spectra(spectra, lmax=1),
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
    )

    compare.assert_array_equal(
        glass.cov_from_spectra(spectra, lmax=4),
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
    )


def test_check_posdef_spectra() -> None:
    # posdef spectra
    assert glass.check_posdef_spectra(
        np.asarray(
            [
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.9, 0.9, 0.9],
            ],
        ),
    )
    # semidef spectra
    assert glass.check_posdef_spectra(
        np.asarray(
            [
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 0.0],
                [0.9, 1.0, 0.0],
            ],
        ),
    )
    # indef spectra
    assert not glass.check_posdef_spectra(
        np.asarray(
            [
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.1, 1.1, 1.1],
            ],
        ),
    )


def test_regularized_spectra(
    mocker: MockerFixture,
    rng: np.random.Generator,
) -> None:
    spectra: AngularPowerSpectra = rng.random(size=(6, 101))

    # test method "nearest"
    cov_nearest = mocker.spy(glass.algorithm, "cov_nearest")
    with pytest.warns(UserWarning, match="Nearest correlation matrix not found"):
        # we don't care about convergence here, only that the correct
        # method is called so this is to suppress the warning
        glass.regularized_spectra(spectra, method="nearest")
    cov_nearest.assert_called_once()

    # test method "clip"
    cov_clip = mocker.spy(glass.algorithm, "cov_clip")
    glass.regularized_spectra(spectra, method="clip")
    cov_clip.assert_called_once()

    # invalid method
    with pytest.raises(ValueError, match="unknown method"):
        glass.regularized_spectra(spectra, method="unknown")
