import healpy as hp
import numpy as np
import pytest

import glass
from glass import (
    cls2cov,
    discretized_cls,
    effective_cls,
    generate_gaussian,
    generate_lognormal,
    getcl,
    iternorm,
    lognormal_gls,
    multalm,
    transform_cls,
)


def test_iternorm() -> None:
    # check output shapes and types

    k = 2

    generator = iternorm(k, np.array([1.0, 0.5, 0.5, 0.5, 0.2, 0.1, 0.5, 0.1, 0.2]))
    result = next(generator)

    j, a, s = result

    assert isinstance(j, int)
    assert a.shape == (k,)
    assert isinstance(s, float)  # type: ignore[unreachable]
    assert s.shape == ()  # type: ignore[unreachable]

    # specify size

    size = 3

    generator = iternorm(
        k,
        np.array(
            [
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
        list(iternorm(k, np.array([[1.0, 0.5], [0.5, 0.2]])))

    # test positive definite error

    with pytest.raises(ValueError, match="covariance matrix is not positive definite"):
        list(iternorm(k, np.array([1.0, 0.5, 0.9, 0.5, 0.2, 0.4, 0.9, 0.4, -1.0])))

    # test multiple iterations

    size = (3,)

    generator = iternorm(
        k,
        np.array(
            [
                [[1.0, 0.5, 0.5], [0.5, 0.2, 0.1], [0.5, 0.1, 0.2]],
                [[2.0, 1.0, 0.8], [1.0, 0.5, 0.3], [0.8, 0.3, 0.6]],
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

    generator = iternorm(0, np.array([1.0]))

    j, a, s = result

    assert j == 1
    assert a.shape == (3, 2)
    assert isinstance(s, np.ndarray)
    assert s.shape == (3,)


def test_cls2cov() -> None:
    # check output values and shape

    nl, nf, nc = 3, 2, 2

    generator = cls2cov(
        [np.array([1.0, 0.5, 0.3]), None, np.array([0.7, 0.6, 0.1])],  # type: ignore[list-item]
        nl,
        nf,
        nc,
    )
    cov = next(generator)

    assert cov.shape == (nl, nc + 1)

    np.testing.assert_array_equal(cov[:, 0], np.array([0.5, 0.25, 0.15]))
    np.testing.assert_array_equal(cov[:, 1], 0)
    np.testing.assert_array_equal(cov[:, 2], 0)

    # test negative value error

    generator = cls2cov(
        np.array(  # type: ignore[arg-type]
            [
                [-1.0, 0.5, 0.3],
                [0.8, 0.4, 0.2],
                [0.7, 0.6, 0.1],
            ]
        ),
        nl,
        nf,
        nc,
    )
    with pytest.raises(ValueError, match="negative values in cl"):
        next(generator)

    # test multiple iterations

    nl, nf, nc = 3, 3, 2

    generator = cls2cov(
        np.array(  # type: ignore[arg-type]
            [
                [1.0, 0.5, 0.3],
                [0.8, 0.4, 0.2],
                [0.7, 0.6, 0.1],
                [0.9, 0.5, 0.3],
                [0.6, 0.3, 0.2],
                [0.8, 0.7, 0.4],
            ]
        ),
        nl,
        nf,
        nc,
    )

    cov1 = np.copy(next(generator))
    cov2 = np.copy(next(generator))
    cov3 = next(generator)

    assert cov1.shape == (nl, nc + 1)
    assert cov2.shape == (nl, nc + 1)
    assert cov3.shape == (nl, nc + 1)

    assert not np.array_equal(cov1, cov2)
    assert not np.array_equal(cov2, cov3)


def test_multalm() -> None:
    # check output values and shapes

    alm = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    bl = np.array([2.0, 0.5, 1.0])
    alm_copy = np.copy(alm)

    result = multalm(alm, bl, inplace=True)

    assert np.array_equal(result, alm)  # in-place
    expected_result = np.array([2.0, 1.0, 3.0, 2.0, 5.0, 6.0])
    np.testing.assert_allclose(result, expected_result)
    assert not np.array_equal(alm_copy, result)

    # multiple with 1s

    bl = np.ones(3)

    result = multalm(alm, bl, inplace=False)
    np.testing.assert_array_equal(result, alm)

    # multiple with 0s

    bl = np.array([0.0, 1.0, 0.0])

    result = multalm(alm, bl, inplace=False)

    expected_result = np.array([0.0, 1.0, 0.0, 2.0, 0.0, 0.0])
    np.testing.assert_allclose(result, expected_result)

    # empty arrays

    alm = np.array([])
    bl = np.array([])

    result = multalm(alm, bl, inplace=False)
    np.testing.assert_array_equal(result, alm)


def test_transform_cls() -> None:
    tfm = "lognormal"
    pars = [2]
    sub_cls = np.array([1, 2, 3, 4, 5])

    # empty cls

    assert transform_cls([], tfm, pars) == []

    # check output shape

    assert len(transform_cls([sub_cls], tfm, pars)) == 1
    assert len(transform_cls([sub_cls], tfm, pars)[0]) == 5

    assert len(transform_cls([sub_cls, sub_cls], tfm, pars)) == 2
    assert len(transform_cls([sub_cls, sub_cls], tfm, pars)[0]) == 5
    assert len(transform_cls([sub_cls, sub_cls], tfm, pars)[1]) == 5

    # one sequence of empty cls

    assert transform_cls([[], sub_cls], tfm, pars)[0] == []

    # monopole behavior

    assert transform_cls([sub_cls, np.linspace(0, 5, 6)], tfm, pars)[1][0] == 0


def test_lognormal_gls() -> None:
    shift = 2

    # empty cls

    assert lognormal_gls([], shift) == []

    # check output shape

    assert len(lognormal_gls([np.linspace(1, 5, 5)], shift)) == 1
    assert len(lognormal_gls([np.linspace(1, 5, 5)], shift)[0]) == 5

    assert len(lognormal_gls([np.linspace(1, 5, 5), np.linspace(1, 5, 5)], shift)) == 2
    assert (
        len(lognormal_gls([np.linspace(1, 5, 5), np.linspace(1, 5, 5)], shift)[0]) == 5
    )
    assert (
        len(lognormal_gls([np.linspace(1, 5, 5), np.linspace(1, 5, 5)], shift)[1]) == 5
    )


def test_discretized_cls() -> None:
    # empty cls

    result = discretized_cls([])
    assert result == []

    # power spectra truncated at lmax + 1 if lmax provided

    result = discretized_cls([np.arange(10), np.arange(10), np.arange(10)], lmax=5)

    for cl in result:
        assert len(cl) == 6

    # check ValueError for triangle number

    with pytest.raises(
        ValueError, match="length of cls array is not a triangle number"
    ):
        discretized_cls([np.arange(10), np.arange(10)], ncorr=1)

    # ncorr not None

    cls = [np.arange(10), np.arange(10), np.arange(10)]
    ncorr = 0
    result = discretized_cls(cls, ncorr=ncorr)

    assert len(result[0]) == 10
    assert len(result[1]) == 10
    assert len(result[2]) == 0  # third correlation should be removed

    # check if pixel window function was applied correctly with nside not None

    nside = 4

    pw = hp.pixwin(nside, lmax=7)

    result = discretized_cls([[], np.ones(10), np.ones(10)], nside=nside)

    for cl in result:
        n = min(len(cl), len(pw))
        expected = np.ones(n) * pw[:n] ** 2
        np.testing.assert_allclose(cl[:n], expected)


def test_effective_cls() -> None:
    # empty cls

    result = effective_cls([], np.array([]))
    assert result.shape == (0,)

    # check ValueError for triangle number

    with pytest.raises(ValueError, match="length of cls is not a triangle number"):
        effective_cls([np.arange(10), np.arange(10)], np.ones((2, 1)))

    # check ValueError for triangle number

    with pytest.raises(ValueError, match="shape mismatch between fields and weights1"):
        effective_cls([], np.ones((3, 1)))

    # check with only weights1

    cls = [np.arange(15), np.arange(15), np.arange(15)]
    weights1 = np.ones((2, 1))

    result = effective_cls(cls, weights1)
    assert result.shape == (1, 1, 15)

    # check truncation if lmax provided

    result = effective_cls(cls, weights1, lmax=5)

    assert result.shape == (1, 1, 6)
    np.testing.assert_allclose(result[..., 6:], 0)

    # check with weights1 and weights2 and weights1 is weights2

    result = effective_cls(cls, weights1, weights2=weights1)
    assert result.shape == (1, 1, 15)


def test_generate_gaussian() -> None:
    gls = [np.array([1.0, 0.5, 0.1])]
    nside = 4
    ncorr = 1

    gaussian_fields = list(generate_gaussian(gls, nside))

    assert gaussian_fields[0].shape == (hp.nside2npix(nside),)

    # requires resetting the RNG for reproducibility
    rng = np.random.default_rng(seed=42)
    gaussian_fields = list(generate_gaussian(gls, nside, rng=rng))

    assert gaussian_fields[0].shape == (hp.nside2npix(nside),)

    # requires resetting the RNG for reproducibility
    rng = np.random.default_rng(seed=42)
    new_gaussian_fields = list(generate_gaussian(gls, nside, ncorr=ncorr, rng=rng))

    assert new_gaussian_fields[0].shape == (hp.nside2npix(nside),)

    np.testing.assert_allclose(new_gaussian_fields[0], gaussian_fields[0])

    with pytest.raises(ValueError, match="all gls are empty"):
        list(generate_gaussian([], nside))


def test_generate_lognormal(rng: np.random.Generator) -> None:
    gls = [np.array([1.0, 0.5, 0.1])]
    nside = 4

    # check shape and values

    lognormal_fields = list(generate_lognormal(gls, nside, shift=1.0, rng=rng))

    assert len(lognormal_fields) == len(gls)
    for m in lognormal_fields:
        assert m.shape == (192,)
        assert np.all(m >= -1)

    # check shift

    # requires resetting the RNG to obtain exact the same result multiplied by 2 (shift)
    rng = np.random.default_rng(seed=42)
    lognormal_fields = list(generate_lognormal(gls, nside, shift=1.0, rng=rng))

    rng = np.random.default_rng(seed=42)
    new_lognormal_fields = list(generate_lognormal(gls, nside, shift=2.0, rng=rng))

    for ms, mu in zip(new_lognormal_fields, lognormal_fields, strict=False):
        np.testing.assert_allclose(ms, mu * 2.0)


def test_getcl() -> None:
    # make a mock Cls array with the index pairs as entries
    cls = [
        np.array([i, j], dtype=np.float64) for i in range(10) for j in range(i, -1, -1)
    ]
    # make sure indices are retrieved correctly
    for i in range(10):
        for j in range(10):
            result = getcl(cls, i, j)
            expected = np.array([min(i, j), max(i, j)], dtype=np.float64)
            np.testing.assert_array_equal(np.sort(result), expected)

            # check slicing
            result = getcl(cls, i, j, lmax=0)
            expected = np.array([max(i, j)], dtype=np.float64)
            assert len(result) == 1
            np.testing.assert_array_equal(result, expected)

            # check padding
            result = getcl(cls, i, j, lmax=50)
            expected = np.zeros((49,), dtype=np.float64)
            assert len(result) == 51
            np.testing.assert_array_equal(result[2:], expected)


def test_inv_triangle_number():
    from glass.fields import inv_triangle_number

    for n in range(10_000):
        assert inv_triangle_number(n * (n + 1) // 2) == n

    for t in 2, 4, 5, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 20:
        with pytest.raises(ValueError, match="not a triangle number"):
            inv_triangle_number(t)


def test_enumerate_spectra():
    n = 100
    tn = n * (n + 1) // 2

    # create mock spectra with 1 element counting to tn
    spectra = np.arange(tn).reshape(tn, 1)

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


def test_spectra_indices():
    np.testing.assert_array_equal(glass.spectra_indices(0), np.zeros((0, 2)))
    np.testing.assert_array_equal(glass.spectra_indices(1), [[0, 0]])
    np.testing.assert_array_equal(glass.spectra_indices(2), [[0, 0], [1, 1], [1, 0]])
    np.testing.assert_array_equal(
        glass.spectra_indices(3),
        [[0, 0], [1, 1], [1, 0], [2, 2], [2, 1], [2, 0]],
    )


def test_gaussian_fields():
    shells = [
        glass.RadialWindow([], [], 1.0),
        glass.RadialWindow([], [], 2.0),
    ]
    fields = glass.gaussian_fields(shells)
    assert len(fields) == len(shells)
    assert all(isinstance(f, glass.grf.Normal) for f in fields)


def test_lognormal_fields():
    shells = [
        glass.RadialWindow([], [], 1),
        glass.RadialWindow([], [], 2),
        glass.RadialWindow([], [], 3),
    ]

    fields = glass.lognormal_fields(shells)
    assert len(fields) == len(shells)
    assert all(isinstance(f, glass.grf.Lognormal) for f in fields)
    assert [f.lamda for f in fields] == [1.0, 1.0, 1.0]

    fields = glass.lognormal_fields(shells, lambda z: z**2)
    assert [f.lamda for f in fields] == [1, 4, 9]


def test_compute_gaussian_spectra(mocker):
    mock = mocker.patch("glass.grf.compute")

    fields = [glass.grf.Normal(), glass.grf.Normal()]
    spectra = [np.zeros(10), np.zeros(10), np.zeros(10)]

    gls = glass.compute_gaussian_spectra(fields, spectra)

    assert mock.call_count == 3
    assert mock.call_args_list[0] == mocker.call(spectra[0], fields[0], fields[0])
    assert mock.call_args_list[1] == mocker.call(spectra[1], fields[1], fields[1])
    assert mock.call_args_list[2] == mocker.call(spectra[2], fields[1], fields[0])
    assert gls == [mock.return_value, mock.return_value, mock.return_value]

    # spectra size mismatch
    with pytest.raises(ValueError, match="fields and spectra"):
        glass.compute_gaussian_spectra(fields, spectra[:2])


def test_solve_gaussian_spectra(mocker):
    mock = mocker.patch("glass.grf.solve")

    result = mock.return_value

    mock.return_value = (result, None, 3)

    fields = [glass.grf.Normal(), glass.grf.Normal()]
    spectra = [np.zeros(5), np.zeros(10), np.zeros(15)]

    gls = glass.solve_gaussian_spectra(fields, spectra)

    assert mock.call_count == 3
    assert mock.call_args_list[0] == mocker.call(
        spectra[0], fields[0], fields[0], pad=10, monopole=0.0
    )
    assert mock.call_args_list[1] == mocker.call(
        spectra[1], fields[1], fields[1], pad=20, monopole=0.0
    )
    assert mock.call_args_list[2] == mocker.call(
        spectra[2], fields[1], fields[0], pad=30, monopole=0.0
    )
    assert gls == [result, result, result]

    # spectra size mismatch
    with pytest.raises(ValueError, match="fields and spectra"):
        glass.solve_gaussian_spectra(fields, spectra[:2])
