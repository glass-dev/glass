import healpy as hp
import numpy as np
import pytest

import glass


@pytest.fixture(scope="session")
def not_triangle_numbers() -> list[int]:
    return [2, 4, 5, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 20]


def test_iternorm() -> None:
    # check output shapes and types

    k = 2

    generator = glass.iternorm(
        k, np.array([1.0, 0.5, 0.5, 0.5, 0.2, 0.1, 0.5, 0.1, 0.2])
    )
    result = next(generator)

    j, a, s = result

    assert isinstance(j, int)
    assert a.shape == (k,)
    assert isinstance(s, float)  # type: ignore[unreachable]
    assert s.shape == ()  # type: ignore[unreachable]

    # specify size

    size = 3

    generator = glass.iternorm(
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
        list(glass.iternorm(k, np.array([[1.0, 0.5], [0.5, 0.2]])))

    # test positive definite error

    with pytest.raises(ValueError, match="covariance matrix is not positive definite"):
        list(
            glass.iternorm(k, np.array([1.0, 0.5, 0.9, 0.5, 0.2, 0.4, 0.9, 0.4, -1.0]))
        )

    # test multiple iterations

    size = (3,)

    generator = glass.iternorm(
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

    generator = glass.iternorm(0, np.array([1.0]))

    j, a, s = result

    assert j == 1
    assert a.shape == (3, 2)
    assert isinstance(s, np.ndarray)
    assert s.shape == (3,)


def test_cls2cov() -> None:
    # check output values and shape

    nl, nf, nc = 3, 2, 2

    generator = glass.cls2cov(
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

    generator = glass.cls2cov(
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

    generator = glass.cls2cov(
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

    result = glass.fields._multalm(alm, bl, inplace=True)

    assert np.array_equal(result, alm)  # in-place
    expected_result = np.array([2.0, 1.0, 1.5, 4.0, 5.0, 6.0])
    np.testing.assert_allclose(result, expected_result)
    assert not np.array_equal(alm_copy, result)

    # multiple with 1s

    alm = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    bl = np.ones(3)

    result = glass.fields._multalm(alm, bl, inplace=False)
    np.testing.assert_array_equal(result, alm)

    # multiple with 0s

    bl = np.array([0.0, 1.0, 0.0])

    result = glass.fields._multalm(alm, bl, inplace=False)

    expected_result = np.array([0.0, 2.0, 3.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(result, expected_result)

    # empty arrays

    alm = np.array([])
    bl = np.array([])

    result = glass.fields._multalm(alm, bl, inplace=False)
    np.testing.assert_array_equal(result, alm)


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


def test_discretized_cls() -> None:
    # empty cls

    result = glass.discretized_cls([])
    assert result == []

    # power spectra truncated at lmax + 1 if lmax provided

    result = glass.discretized_cls(
        [np.arange(10), np.arange(10), np.arange(10)], lmax=5
    )

    for cl in result:
        assert len(cl) == 6

    # check ValueError for triangle number

    with pytest.raises(ValueError, match="invalid number of spectra:"):
        glass.discretized_cls([np.arange(10), np.arange(10)], ncorr=1)

    # ncorr not None

    cls = [np.arange(10), np.arange(10), np.arange(10)]
    ncorr = 0
    result = glass.discretized_cls(cls, ncorr=ncorr)

    assert len(result[0]) == 10
    assert len(result[1]) == 10
    assert len(result[2]) == 0  # third correlation should be removed

    # check if pixel window function was applied correctly with nside not None

    nside = 4

    pw = hp.pixwin(nside, lmax=7)

    result = glass.discretized_cls([[], np.ones(10), np.ones(10)], nside=nside)

    for cl in result:
        n = min(len(cl), len(pw))
        expected = np.ones(n) * pw[:n] ** 2
        np.testing.assert_allclose(cl[:n], expected)


def test_effective_cls() -> None:
    # empty cls

    result = glass.effective_cls([], np.array([]))
    assert result.shape == (0,)

    # check ValueError for triangle number

    with pytest.raises(ValueError, match="invalid number of spectra:"):
        glass.effective_cls([np.arange(10), np.arange(10)], np.ones((2, 1)))

    # check ValueError for triangle number

    with pytest.raises(ValueError, match="shape mismatch between fields and weights1"):
        glass.effective_cls([], np.ones((3, 1)))

    # check with only weights1

    cls = [np.arange(15), np.arange(15), np.arange(15)]
    weights1 = np.ones((2, 1))

    result = glass.effective_cls(cls, weights1)
    assert result.shape == (1, 1, 15)

    # check truncation if lmax provided

    result = glass.effective_cls(cls, weights1, lmax=5)

    assert result.shape == (1, 1, 6)
    np.testing.assert_allclose(result[..., 6:], 0)

    # check with weights1 and weights2 and weights1 is weights2

    result = glass.effective_cls(cls, weights1, weights2=weights1)
    assert result.shape == (1, 1, 15)


def test_generate_grf() -> None:
    gls = [np.array([1.0, 0.5, 0.1])]
    nside = 4
    ncorr = 1

    gaussian_fields = list(glass.fields._generate_grf(gls, nside))

    assert gaussian_fields[0].shape == (hp.nside2npix(nside),)

    # requires resetting the RNG for reproducibility
    rng = np.random.default_rng(seed=42)
    gaussian_fields = list(glass.fields._generate_grf(gls, nside, rng=rng))

    assert gaussian_fields[0].shape == (hp.nside2npix(nside),)

    # requires resetting the RNG for reproducibility
    rng = np.random.default_rng(seed=42)
    new_gaussian_fields = list(
        glass.fields._generate_grf(gls, nside, ncorr=ncorr, rng=rng)
    )

    assert new_gaussian_fields[0].shape == (hp.nside2npix(nside),)

    np.testing.assert_allclose(new_gaussian_fields[0], gaussian_fields[0])

    with pytest.raises(ValueError, match="all gls are empty"):
        list(glass.fields._generate_grf([], nside))


def test_generate_gaussian() -> None:
    with pytest.deprecated_call():
        glass.generate_gaussian([np.array([1.0, 0.5, 0.1])], 4)


def test_generate_lognormal() -> None:
    with pytest.deprecated_call():
        glass.generate_lognormal([np.array([1.0, 0.5, 0.1])], 4)


def test_generate():
    # shape mismatch error

    fields = [lambda x, var: x, lambda x, var: x]  # noqa: ARG005

    with pytest.raises(ValueError, match="mismatch between number of fields and gls"):
        list(glass.generate(fields, [np.ones(10), np.ones(10)], nside=16))

    # check output shape

    nside = 16
    npix = hp.nside2npix(nside)
    gls = [np.ones(10), np.ones(10), np.ones(10)]

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

    np.testing.assert_allclose(result[1], result[0] ** 2, atol=1e-05)


def test_getcl() -> None:
    # make a mock Cls array with the index pairs as entries
    cls = [
        np.array([i, j], dtype=np.float64) for i in range(10) for j in range(i, -1, -1)
    ]
    # make sure indices are retrieved correctly
    for i in range(10):
        for j in range(10):
            result = glass.getcl(cls, i, j)
            expected = np.array([min(i, j), max(i, j)], dtype=np.float64)
            np.testing.assert_array_equal(np.sort(result), expected)

            # check slicing
            result = glass.getcl(cls, i, j, lmax=0)
            expected = np.array([max(i, j)], dtype=np.float64)
            assert len(result) == 1
            np.testing.assert_array_equal(result, expected)

            # check padding
            result = glass.getcl(cls, i, j, lmax=50)
            expected = np.zeros((49,), dtype=np.float64)
            assert len(result) == 51
            np.testing.assert_array_equal(result[2:], expected)


def test_is_inv_triangle_number(not_triangle_numbers: list[int]):
    for n in range(10_000):
        assert glass.fields._inv_triangle_number(n * (n + 1) // 2) == n

    for t in not_triangle_numbers:
        with pytest.raises(ValueError, match=f"not a triangle number: {t}"):
            glass.fields._inv_triangle_number(t)


def test_nfields_from_nspectra(not_triangle_numbers: list[int]):
    for n in range(10_000):
        assert glass.nfields_from_nspectra(n * (n + 1) // 2) == n

    for t in not_triangle_numbers:
        with pytest.raises(ValueError, match=f"invalid number of spectra: {t}"):
            glass.nfields_from_nspectra(t)


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


def test_compute_gaussian_spectra_gh639(mocker):
    """Test compute_gaussian_spectra() with an empty input."""
    mock = mocker.patch("glass.grf.compute")

    fields = [glass.grf.Normal(), glass.grf.Normal()]
    spectra = [np.zeros(10), np.zeros(10), np.zeros(0)]

    gls = glass.compute_gaussian_spectra(fields, spectra)

    assert mock.call_count == 2
    assert mock.call_args_list[0] == mocker.call(spectra[0], fields[0], fields[0])
    assert mock.call_args_list[1] == mocker.call(spectra[1], fields[1], fields[1])
    assert gls[:2] == [mock.return_value, mock.return_value]
    assert gls[2].size == 0


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


def test_glass_to_healpix_spectra():
    inp = [11, 22, 21, 33, 32, 31, 44, 43, 42, 41]
    out = glass.glass_to_healpix_spectra(inp)
    np.testing.assert_array_equal(out, [11, 22, 33, 44, 21, 32, 43, 31, 42, 41])


def test_healpix_to_glass_spectra():
    inp = [11, 22, 33, 44, 21, 32, 43, 31, 42, 41]
    out = glass.healpix_to_glass_spectra(inp)
    np.testing.assert_array_equal(out, [11, 22, 21, 33, 32, 31, 44, 43, 42, 41])


def test_glass_to_healpix_alm():
    inp = np.array([00, 10, 11, 20, 21, 22, 30, 31, 32, 33])
    out = glass.fields._glass_to_healpix_alm(inp)
    np.testing.assert_array_equal(
        out, np.array([00, 10, 20, 30, 11, 21, 31, 22, 32, 33])
    )


def test_lognormal_shift_hilbert2011():
    zs = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    shifts = [glass.lognormal_shift_hilbert2011(z) for z in zs]

    # computed by hand
    check = [0.0103031, 0.02975, 0.0538781, 0.0792, 0.103203, 0.12435, 0.142078, 0.1568]

    np.testing.assert_allclose(shifts, check, atol=1e-4, rtol=1e-4)


def test_cov_from_spectra():
    spectra = np.array(
        [
            [110, 111, 112, 113],
            [220, 221, 222, 223],
            [210, 211, 212, 213],
            [330, 331, 332, 333],
            [320, 321, 322, 323],
            [310, 311, 312, 313],
        ]
    )

    np.testing.assert_array_equal(
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

    np.testing.assert_array_equal(
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

    np.testing.assert_array_equal(
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


def test_check_posdef_spectra():
    # posdef spectra
    assert glass.fields.check_posdef_spectra(
        np.array(
            [
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.9, 0.9, 0.9],
            ]
        )
    )
    # semidef spectra
    assert glass.fields.check_posdef_spectra(
        np.array(
            [
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 0.0],
                [0.9, 1.0, 0.0],
            ]
        )
    )
    # indef spectra
    assert not glass.fields.check_posdef_spectra(
        np.array(
            [
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.1, 1.1, 1.1],
            ]
        )
    )


def test_regularized_spectra(mocker, rng):
    spectra = rng.random(size=(6, 101))

    # test method "nearest"
    cov_nearest = mocker.spy(glass.algorithm, "cov_nearest")
    glass.fields.regularized_spectra(spectra, method="nearest")
    cov_nearest.assert_called_once()

    # test method "clip"
    cov_clip = mocker.spy(glass.algorithm, "cov_clip")
    glass.fields.regularized_spectra(spectra, method="clip")
    cov_clip.assert_called_once()

    # invalid method
    with pytest.raises(ValueError, match="unknown method"):
        glass.fields.regularized_spectra(spectra, method="unknown")
