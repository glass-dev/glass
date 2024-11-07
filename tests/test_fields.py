import numpy as np
import pytest

from glass.fields import cls2cov, getcl, iternorm, multalm


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
