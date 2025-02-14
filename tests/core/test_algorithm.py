import numpy as np
import pytest

import glass.core.algorithm


def test_nnls(rng: np.random.Generator) -> None:
    # check output

    a = np.arange(25.0).reshape(-1, 5)
    b = np.arange(5.0)
    y = a @ b
    res = glass.core.algorithm.nnls(a, y)
    assert np.linalg.norm((a @ res) - y) < 1e-7

    a = rng.uniform(low=-10, high=10, size=[50, 10])
    b = np.abs(rng.uniform(low=-2, high=2, size=[10]))
    b[::2] = 0
    x = a @ b
    res = glass.core.algorithm.nnls(
        a, x, tol=500 * np.linalg.norm(a, 1) * np.spacing(1.0)
    )
    np.testing.assert_allclose(res, b, rtol=0.0, atol=1e-10)

    # check matrix and vector's shape

    a = rng.standard_normal((100, 20))
    b = rng.standard_normal((100,))

    with pytest.raises(ValueError, match="input `a` is not a matrix"):
        glass.core.algorithm.nnls(b, a)
    with pytest.raises(ValueError, match="input `b` is not a vector"):
        glass.core.algorithm.nnls(a, a)
    with pytest.raises(ValueError, match="the shapes of `a` and `b` do not match"):
        glass.core.algorithm.nnls(a.T, b)


def test_cov_clip(rng):
    # prepare a random matrix
    m = rng.random((4, 4))

    # symmetric matrix
    a = (m + m.T) / 2

    # fix by clipping negative eigenvalues
    cov = glass.core.algorithm.cov_clip(a)

    # make sure all eigenvalues are positive
    assert np.all(np.linalg.eigvalsh(cov) >= 0)


def test_nearcorr():
    # from Higham (2002)
    a = np.array(
        [
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
    )
    x = glass.core.algorithm.nearcorr(a)
    np.testing.assert_allclose(
        x,
        [
            [1.0000, 0.7607, 0.1573],
            [0.7607, 1.0000, 0.7607],
            [0.1573, 0.7607, 1.0000],
        ],
        atol=0.0001,
    )


def test_cov_nearest(rng, mocker):
    # prepare a random matrix
    m = rng.random((4, 4))

    # symmetric matrix
    a = np.eye(4) + (m + m.T) / 2

    # spy on the call to nearcorr
    nearcorr = mocker.spy(glass.core.algorithm, "nearcorr")

    # compute covariance
    cov = glass.core.algorithm.cov_nearest(a)

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
