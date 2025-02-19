import numpy as np
import pytest

import glass.algorithm


def test_nnls(rng: np.random.Generator) -> None:
    # check output

    a = np.arange(25.0).reshape(-1, 5)
    b = np.arange(5.0)
    y = a @ b
    res = glass.algorithm.nnls(a, y)
    assert np.linalg.norm((a @ res) - y) < 1e-7

    a = rng.uniform(low=-10, high=10, size=[50, 10])
    b = np.abs(rng.uniform(low=-2, high=2, size=[10]))
    b[::2] = 0
    x = a @ b
    res = glass.algorithm.nnls(a, x, tol=500 * np.linalg.norm(a, 1) * np.spacing(1.0))
    np.testing.assert_allclose(res, b, rtol=0.0, atol=1e-10)

    # check matrix and vector's shape

    a = rng.standard_normal((100, 20))
    b = rng.standard_normal((100,))

    with pytest.raises(ValueError, match="input `a` is not a matrix"):
        glass.algorithm.nnls(b, a)
    with pytest.raises(ValueError, match="input `b` is not a vector"):
        glass.algorithm.nnls(a, a)
    with pytest.raises(ValueError, match="the shapes of `a` and `b` do not match"):
        glass.algorithm.nnls(a.T, b)


def test_cov_clip(rng):
    # prepare a random matrix
    m = rng.random((4, 4))

    # symmetric matrix
    a = (m + m.T) / 2

    # fix by clipping negative eigenvalues
    cov = glass.algorithm.cov_clip(a)

    # make sure all eigenvalues are positive
    assert np.all(np.linalg.eigvalsh(cov) >= 0)

    # fix by clipping negative eigenvalues
    cov = glass.algorithm.cov_clip(a, rtol=1.0)

    # make sure all eigenvalues are positive
    h = np.linalg.eigvalsh(a).max()
    np.testing.assert_allclose(np.linalg.eigvalsh(cov), h)


def test_nearcorr():
    # from Higham (2002)
    a = np.array(
        [
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
    )
    b = np.array(
        [
            [1.0000, 0.7607, 0.1573],
            [0.7607, 1.0000, 0.7607],
            [0.1573, 0.7607, 1.0000],
        ],
    )

    x = glass.algorithm.nearcorr(a)
    np.testing.assert_allclose(x, b, atol=0.0001)

    # explicit tolerance
    x = glass.algorithm.nearcorr(a, tol=1e-10)
    np.testing.assert_allclose(x, b, atol=0.0001)

    # no iterations
    x = glass.algorithm.nearcorr(a, niter=0)
    np.testing.assert_allclose(x, a)

    # non-square matrix should raise
    with pytest.raises(ValueError, match="non-square matrix"):
        glass.algorithm.nearcorr(np.zeros((4, 3)))


def test_cov_nearest(rng, mocker):
    # prepare a random matrix
    m = rng.random((4, 4))

    # symmetric matrix
    a = np.eye(4) + (m + m.T) / 2

    # spy on the call to nearcorr
    nearcorr = mocker.spy(glass.algorithm, "nearcorr")

    # compute covariance
    cov = glass.algorithm.cov_nearest(a)

    # make sure all eigenvalues are positive
    assert np.all(np.linalg.eigvalsh(cov) >= 0)

    # get normalisation
    sq_d = np.sqrt(a.diagonal())
    norm = np.outer(sq_d, sq_d)

    # make sure nearcorr was called with correct input
    nearcorr.assert_called_once()
    np.testing.assert_array_almost_equal_nulp(
        nearcorr.call_args_list[0].args[0],
        np.divide(a, norm, where=(norm > 0), out=np.zeros_like(a)),
    )

    # cannot deal with negative variances
    with pytest.raises(ValueError, match="negative values"):
        glass.algorithm.cov_nearest(np.diag([1, 1, -1]))
